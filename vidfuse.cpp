#include "cppfuse.hpp"
#include "cppfuse/transparent_fuse.hpp"

#include <iostream>
#include <sstream>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
using namespace std;

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfiltergraph.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>

#include <alloca.h>
#undef av_err2str
#define av_err2str(errnum) \
	av_make_error_string((char*)__builtin_alloca(AV_ERROR_MAX_STRING_SIZE), \
	AV_ERROR_MAX_STRING_SIZE, errnum) 
}

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;

struct VidParams {
	unsigned int bitrate;
	bool hardsubbing;
	bool scaling;
	int scaleMode;
	double scaleFactor;
	size_t scaleMaxPix;
	double scaleMaxWidth;
	size_t dumpsize;
	std::string filepath;
	std::string fullpath;

	VidParams() : bitrate(1500), hardsubbing(true), scaling(true), scaleMaxWidth(1280), dumpsize(0) {}
};


class TranscoderError : public std::exception {
public:
	TranscoderError(int errnum) : errnum(errnum) {
		av_strerror(errnum, errstr, AV_ERROR_MAX_STRING_SIZE);
	}

	TranscoderError(int errnum, const char *log) : errnum(errnum) {
		av_strerror(errnum, errstr, AV_ERROR_MAX_STRING_SIZE);
		av_log(NULL, AV_LOG_ERROR, "%s%s", log, errstr);
	}

	virtual const char *what() const throw() {
		return errstr;
	}

	int errnum;
	char errstr[AV_ERROR_MAX_STRING_SIZE];
};

struct TranscoderErrorOpenInput : TranscoderError {TranscoderErrorOpenInput(int errnum) : TranscoderError(errnum) {}};
struct TranscoderErrorFindStreamInfo : TranscoderError {TranscoderErrorFindStreamInfo(int errnum) : TranscoderError(errnum) {}};
struct TranscoderErrorCreateOutputContext : TranscoderError {TranscoderErrorCreateOutputContext(int errnum) : TranscoderError(errnum) {}};
struct TranscoderErrorWriteHeader : TranscoderError {TranscoderErrorWriteHeader(int errnum) : TranscoderError(errnum) {}};

class Transcoder {
public:
	Transcoder(std::shared_ptr<VidParams> params)
		: params(params),
		  ifmtCtx(nullptr),
		  ofmtCtx(nullptr),
		  transStream(),
		  muxStream(),
		  videoStreamIndex(NO_STREAM),
		  subsStreamIndex(NO_STREAM)
	{
		initEncoding();
	}

	~Transcoder() {
		deinitEncoding();
	}

	std::shared_ptr<VidParams> getParams() {
		return params;
	}

	int read(char *buf, size_t size, off_t offset) {
		off_t remaining = bigBufferSize - offset;
		if (remaining <= 0)
			return 0;
		if ((size_t)remaining < size)
			size = remaining;
//		if (offset < writeBufferPos && (size + offset) > writeBufferPos)
//			size = writeBufferPos - offset;
		std::unique_lock<std::mutex> lk(bufferMutex);
		bufferSignal.wait(lk, [&]{return ((size + offset) <= this->writeBufferPos);});
		//av_log(NULL, AV_LOG_ERROR, "Transcoder::read(%p, %lu, %ld)\n", buf, size, offset);
		std::memcpy(buf, bigBuffer.get() + offset, size);
		return size;
	}

private:
	struct FilterContext {
		AVFilterContext *bufferSinkCtx;
		AVFilterContext *bufferSrcCtx;
		deleted_unique_ptr<AVFilterGraph> filterGraph;
		FilterContext() :
			bufferSinkCtx(nullptr),
			bufferSrcCtx(nullptr),
			filterGraph(nullptr, [&](AVFilterGraph *g){avfilter_graph_free(&g);})
		{}
	};

	static int write_packet(void *opaque, uint8_t *buffer, int count) {
		return ((Transcoder*)opaque)->write_packet(buffer, count);
	}
	static int64_t seek(void *opaque, int64_t offset, int whence) {
		//av_log(NULL, AV_LOG_ERROR, "avio seek(%p, %li, %d)\n", opaque, offset, whence);
		return ((Transcoder*)opaque)->seek(offset, whence);
	}

	int write_packet(uint8_t *buffer, int count) {
		off_t remaining = bigBufferSize - writeBufferPos;
		if (remaining >= count) {
			std::memcpy(bigBuffer.get() + writeBufferPos, buffer, count);
			//av_log(NULL, AV_LOG_ERROR, "avio write_packet(%d, %ld)\n", count, writeBufferPos);
			writeBufferPos += count;
		}
		bufferSignal.notify_one();
		return 0;
	}

	int seek(int64_t offset, int whence) {
		writeBufferPos = offset + ((whence & SEEK_CUR) ? writeBufferPos : 0);
		return 0;
	}

	void initEncoding() {
		int ret;

		cout << params->fullpath.c_str() << endl;
		if ((ret = avformat_open_input(&ifmtCtx, params->fullpath.c_str(), NULL, NULL)) < 0)
		{
			av_log(NULL, AV_LOG_ERROR, "Cannot open input file\n");
			throw TranscoderErrorOpenInput(ret);
		}

		if ((ret = avformat_find_stream_info(ifmtCtx, NULL)) < 0) {
			av_log(NULL, AV_LOG_ERROR, "Cannot find stream information\n");
			avformat_close_input(&ifmtCtx);
			throw TranscoderErrorFindStreamInfo(ret);
		}

		transStream.assign(ifmtCtx->nb_streams, false);
		muxStream.assign(ifmtCtx->nb_streams, false);
		mappedStream.assign(ifmtCtx->nb_streams, NO_STREAM);
		videoStreamIndex = NO_STREAM;
		subsStreamIndex = NO_STREAM;
		subsStreamCount = 0;
		subsStreamNum = NO_STREAM;

		size_t outStreamCount = 0;

		for (unsigned int i = 0; i < ifmtCtx->nb_streams; i++) {
			analyseStream(i);
			if (muxStream[i])
				outStreamCount++;
		}

		// enforce transcoding for hardsubbing
		if (subsStreamIndex != NO_STREAM) {
			if (videoStreamIndex != NO_STREAM)
				transStream[videoStreamIndex] = true;
			else {
				av_log(NULL, AV_LOG_ERROR, "No video stream to draw subtitles on were found\n");
				subsStreamIndex = NO_STREAM;
			}
		}

		avformat_alloc_output_context2(&ofmtCtx, NULL, "Matroska", NULL);
		//avformat_alloc_output_context2(&ofmtCtx, NULL, NULL, ".mp4");
		if (!ofmtCtx) {
			av_log(NULL, AV_LOG_ERROR, "Could not create output context\n");
			throw TranscoderError(AVERROR_UNKNOWN);
		}

		filter.resize(outStreamCount);

		for (unsigned int i = 0, o = 0; i < ifmtCtx->nb_streams; i++) {
			if (!muxStream[i])
				continue;

			AVStream *outStream = avformat_new_stream(ofmtCtx, NULL);
			AVStream *inStream = ifmtCtx->streams[i];
			mappedStream[i] = outStream->index;

			if (transStream[i])
				initStreamTranscode(i);

			if (!transStream[i]) {
				// could have changed if initStreamTranscode failed
				outStream->time_base = inStream->time_base;
				ret = avcodec_copy_context(outStream->codec, inStream->codec);
				if (ret < 0) {
					muxStream[i] = false;
					mappedStream[i] = NO_STREAM;
					av_log(NULL, AV_LOG_INFO, "Copying stream context of stream %d failed\n", i);
				}
			}

			if (ofmtCtx->oformat->flags & AVFMT_GLOBALHEADER)
				outStream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

			av_dict_copy(&outStream->metadata, inStream->metadata, 0);

			o++;
		}
	
		av_dump_format(ifmtCtx, 0, NULL, 0);
		av_dump_format(ofmtCtx, 0, NULL, 1);

		smallBuffer.reset(new unsigned char[smallBufferSize]);
		bigBuffer.reset(new unsigned char[bigBufferSize]);
		writeBufferPos = 0;
		lastReadPos = 0;
		ofmtCtx->pb = avio_alloc_context(smallBuffer.get(), smallBufferSize, AVIO_FLAG_WRITE, this, NULL, Transcoder::write_packet, Transcoder::seek);

		/* init muxer, write output file header */
		ret = avformat_write_header(ofmtCtx, NULL);
		if (ret < 0) {
			av_log(NULL, AV_LOG_ERROR, "Error occurred when writing header\n");
			throw TranscoderError(ret);
		}

		stopEncode = false;
		encThread = std::thread(&Transcoder::encode, this);
	}

	void analyseStream(unsigned int index) {
		unsigned int i = index;
		AVStream           *stream = ifmtCtx->streams[i];
		AVCodecContext  *codec_ctx = stream->codec;
		AVMediaType           type = codec_ctx->codec_type;
		AVCodecID               id = codec_ctx->codec_id;
		int                profile = codec_ctx->profile;

		switch (type) {
		case AVMEDIA_TYPE_VIDEO:
			if (videoStreamIndex != NO_STREAM)
				break;
			videoStreamIndex = i;
			transStream[i] = true;
			muxStream[i] = true;
			break;
		case AVMEDIA_TYPE_AUDIO:
			transStream[i] = true;
			muxStream[i] = true;
			transStream[i] = false; // as long as i cannot transcode audio
			//muxStream[i] = false;
			break;
		case AVMEDIA_TYPE_SUBTITLE:
			transStream[i] = false;
			if (!params->hardsubbing) {
				muxStream[i] = true;
				break;
			}
			if (subsStreamIndex != NO_STREAM)
				break;
			subsStreamIndex = i;
			subsStreamNum = subsStreamCount++;
			muxStream[i] = false;
			break;
		default:
			transStream[i] = false;
			muxStream[i] = false;
		}
	}

	void initStreamTranscode(unsigned int index) {
		unsigned int i = index;
		unsigned int o = mappedStream[i];
		AVStream           *istream = ifmtCtx->streams[i];
		AVCodecContext      *decCtx = istream->codec;
		AVMediaType            type = decCtx->codec_type;
		AVCodecID                id = decCtx->codec_id;
		AVStream           *ostream = ofmtCtx->streams[o];
		AVCodecContext      *encCtx = ostream->codec;
		AVCodec *encoder;
		int ret;
	
		//transStream[i] = false;
		//return;
		AVCodec *decoder = avcodec_find_decoder(id);
		ret = avcodec_open2(decCtx, decoder, NULL);
		if (ret < 0) {
			av_log(NULL, AV_LOG_ERROR, "Failed to open decoder for stream #%u: %s\n", i, av_err2str(ret));
			transStream[i] = false;
			muxStream[i] = true;
			return;
		}

		encoder = avcodec_find_encoder_by_name("nvenc");
		if (!encoder) {
			av_log(NULL, AV_LOG_ERROR, "Failed to open nvidia encoder for stream #%u: %s\n", i, av_err2str(ret));
			transStream[i] = false;
			muxStream[i] = true;
			return;
		}
		AVDictionary *codecOptions = NULL;
		encCtx->height = decCtx->height;
		encCtx->width = decCtx->width;
		encCtx->sample_aspect_ratio = decCtx->sample_aspect_ratio;
		if (params->scaling) {
			if (encCtx->width > params->scaleMaxWidth)
				encCtx->width = params->scaleMaxWidth;
			encCtx->height = decCtx->height * encCtx->width / decCtx->width;
		}
		encCtx->pix_fmt = encoder->pix_fmts[0];
		encCtx->time_base = decCtx->time_base;
		ret = av_dict_set(&codecOptions, "profile", "high", 0);
		ret = av_dict_set(&codecOptions, "preset", "llhq", 0);
		ret = av_dict_set(&codecOptions, "twopass", "1", 0);
		encCtx->qmin = 15;
		encCtx->qmax = 18;
		ret = avcodec_open2(encCtx, encoder, &codecOptions);
		if (ret < 0) {
			av_log(NULL, AV_LOG_ERROR, "Cannot open video encoder for stream #%u\n", i);
			transStream[i] = false;
			muxStream[i] = true;
			return;
		}

		ostream->time_base = encCtx->time_base;

		char args[512];
		AVFilter *bufferSrc   = avfilter_get_by_name("buffer");
		AVFilter *bufferSink  = avfilter_get_by_name("buffersink");
		AVFilter *subfilter   = avfilter_get_by_name("subtitles");
		AVFilter *scale = avfilter_get_by_name("scale");
		deleted_unique_ptr<AVFilterInOut>
			outputs(avfilter_inout_alloc(), [&](AVFilterInOut*f){avfilter_inout_free(&f);}),
			inputs(avfilter_inout_alloc(), [&](AVFilterInOut*f){avfilter_inout_free(&f);});
		AVFilterContext *subfilterCtx = NULL;
		AVFilterContext *scaleCtx = NULL;
		filter[o].filterGraph.reset(avfilter_graph_alloc());

		snprintf(args, sizeof(args),
				"video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
				decCtx->width, decCtx->height, decCtx->pix_fmt,
				decCtx->time_base.num, decCtx->time_base.den,
				decCtx->sample_aspect_ratio.num,
				decCtx->sample_aspect_ratio.den);
		ret = avfilter_graph_create_filter(&filter[o].bufferSrcCtx, bufferSrc, "in",
				args, NULL, filter[o].filterGraph.get());
		if (ret < 0) {
			av_log(NULL, AV_LOG_ERROR, "Cannot create buffer source\n");
			transStream[i] = false;
			muxStream[i] = true;
			return;
		}

		if (params->scaling) {
			snprintf(args, sizeof(args),
					"w=%d:h=%d:flags=lanczos",
					encCtx->width, encCtx->height);
			ret = avfilter_graph_create_filter(&scaleCtx,  scale, "scale", args, NULL, filter[o].filterGraph.get());
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Cannot create scale filter\n");
				params->scaling = false;
				encCtx->width = decCtx->width;
				encCtx->height = decCtx->height;
			}
		}

		if (params->hardsubbing && subsStreamIndex != NO_STREAM) {
			snprintf(args, sizeof(args),
					"f=%s:original_size=%dx%d:si=%d",
					params->fullpath.c_str(), decCtx->width, decCtx->height, subsStreamNum);
			ret = avfilter_graph_create_filter(&subfilterCtx,  subfilter, "sub", args, NULL, filter[o].filterGraph.get());
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Cannot create subtitle rendering filter with stream %i: %s\n", subsStreamNum, av_err2str(ret));
				params->hardsubbing = false;
				subsStreamIndex = NO_STREAM;
			}
		}

		ret = avfilter_graph_create_filter(&filter[o].bufferSinkCtx, bufferSink, "out",
				NULL, NULL, filter[o].filterGraph.get());
		if (ret < 0) {
			av_log(NULL, AV_LOG_ERROR, "Cannot create buffer sink\n");
			transStream[i] = false;
			muxStream[i] = true;
			return;
		}

		ret = av_opt_set_bin(filter[o].bufferSinkCtx, "pix_fmts",
				(uint8_t*)&encCtx->pix_fmt, sizeof(encCtx->pix_fmt),
				AV_OPT_SEARCH_CHILDREN);
		if (ret < 0) {
			av_log(NULL, AV_LOG_ERROR, "Cannot set output pixel format\n");
			transStream[i] = false;
			muxStream[i] = true;
			return;
		}

		AVFilterContext *prev = filter[o].bufferSrcCtx;
		AVFilterContext *next = NULL;
		while (next != filter[o].bufferSinkCtx) {
			if (prev == filter[o].bufferSrcCtx && params->hardsubbing)
				next = subfilterCtx;
			else if(prev != scaleCtx && params->scaling)
				next = scaleCtx;
			else
				next = filter[o].bufferSinkCtx;
			ret = avfilter_link(prev, 0, next, 0);
			prev = next;
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Cannot link filter\n");
				transStream[i] = false;
				muxStream[i] = true;
				return;
			}
		}

		/* Endpoints for the filter graph. */
		outputs->name       = av_strdup("in");
		outputs->filter_ctx  = filter[o].bufferSrcCtx;
		outputs->pad_idx    = 0;
		outputs->next       = NULL;

		inputs->name        = av_strdup("out");
		inputs->filter_ctx  = filter[o].bufferSinkCtx;
		inputs->pad_idx     = 0;
		inputs->next        = NULL;
		if ((ret = avfilter_graph_config(filter[o].filterGraph.get(), NULL)) < 0) {
			transStream[i] = false;
			muxStream[i] = true;
			return;
		}
	}

	void encode() {
		int ret;
		unsigned int inStreamIdx, outStreamIdx;
		int gotFrame;
		AVPacket packet;
		deleted_unique_ptr<AVFrame> frame(nullptr, [](AVFrame *frame){av_frame_free(&frame);});
		AVMediaType type;
		int (*decFunc)(AVCodecContext *, AVFrame *, int *, const AVPacket *);
		int (*encFunc)(AVCodecContext *, AVPacket *, const AVFrame *, int *);

		packet.data = NULL;
		packet.size = 0;
		av_init_packet(&packet);

		isEncoding = true;
		while (!stopEncode) {
			if ((ret = av_read_frame(ifmtCtx, &packet)) < 0 )
				break;
			inStreamIdx = packet.stream_index;
			outStreamIdx = mappedStream[inStreamIdx];
			if (outStreamIdx == NO_STREAM)
				continue;

			if (!transStream[inStreamIdx]) {
				av_packet_rescale_ts(&packet,
					ifmtCtx->streams[inStreamIdx]->time_base,
					ofmtCtx->streams[outStreamIdx]->time_base);
				ret = av_interleaved_write_frame(ofmtCtx, &packet);
				if (ret < 0)
					break;
				av_packet_unref(&packet);
				continue;
			}

			type = ifmtCtx->streams[inStreamIdx]->codec->codec_type;
			frame.reset(av_frame_alloc());
			av_packet_rescale_ts(&packet,
				ifmtCtx->streams[inStreamIdx]->time_base,
				ofmtCtx->streams[outStreamIdx]->codec->time_base);
			decFunc = (type == AVMEDIA_TYPE_VIDEO) ? avcodec_decode_video2 :
				avcodec_decode_audio4;
			ret = decFunc(ifmtCtx->streams[inStreamIdx]->codec, frame.get(), &gotFrame, &packet);
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Decoding failed\n");
				break;
			}
			if (!gotFrame) {
				av_packet_unref(&packet);
				continue;
			}

			frame->pts = av_frame_get_best_effort_timestamp(frame.get());
			ret = av_buffersrc_add_frame_flags(filter[outStreamIdx].bufferSrcCtx, frame.get(), 0);
			deleted_unique_ptr<AVFrame> filtFrame(av_frame_alloc(), [](AVFrame*f){av_frame_free(&f);});
			av_buffersink_get_frame(filter[outStreamIdx].bufferSinkCtx, filtFrame.get());
			if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
				av_packet_unref(&packet);
				continue;
			}
			filtFrame->pict_type = AV_PICTURE_TYPE_NONE;

			encFunc = (type == AVMEDIA_TYPE_VIDEO) ? avcodec_encode_video2 :
				avcodec_encode_audio2;
			AVPacket encPkt;
			encPkt.data = NULL;
			encPkt.size = 0;
			av_init_packet(&encPkt);
			ret = encFunc(ofmtCtx->streams[outStreamIdx]->codec, &encPkt, filtFrame.get(), &gotFrame);
			if (ret < 0)
				break;
			if (!gotFrame) {
				av_packet_unref(&packet);
				continue;
			}
			encPkt.stream_index = outStreamIdx;
			av_packet_rescale_ts(&encPkt,
				ofmtCtx->streams[outStreamIdx]->codec->time_base,
				ofmtCtx->streams[outStreamIdx]->time_base);
			ret = av_interleaved_write_frame(ofmtCtx, &encPkt);
			if (ret < 0)
				break;
			av_packet_unref(&packet);
		}
		av_packet_unref(&packet);
		isEncoding = false;

		if (ret < 0) {
			av_log(NULL, AV_LOG_ERROR, "Error while encoding: %s\n", av_err2str(ret)); 
		}
	}

	void deinitEncoding() {
		stopEncode = true;
		encThread.join();
		av_write_trailer(ofmtCtx);
		for (unsigned int i = 0; i < ifmtCtx->nb_streams; i++) {
			avcodec_close(ifmtCtx->streams[i]->codec);
		}
		for (unsigned int i = 0; i < ofmtCtx->nb_streams; i++) {
			avcodec_close(ofmtCtx->streams[i]->codec);
		}
		avformat_close_input(&ifmtCtx);
		avformat_free_context(ofmtCtx);
	}

	std::shared_ptr<VidParams> params;
	AVFormatContext *ifmtCtx;
	AVFormatContext *ofmtCtx;
	std::vector<bool> transStream;
	std::vector<bool> muxStream;
	std::vector<int> mappedStream;
	unsigned int videoStreamIndex;
	unsigned int subsStreamIndex;
	unsigned int subsStreamCount;
	unsigned int subsStreamNum;
	static const unsigned int NO_STREAM = -1;
	std::vector<FilterContext> filter;

	std::unique_ptr<unsigned char[]> smallBuffer;
	static const int smallBufferSize = 100*1024;
	std::unique_ptr<unsigned char[]> bigBuffer;
	static const int bigBufferSize = 100*1024*1024;
	size_t writeBufferPos;
	size_t lastReadPos;

	std::thread encThread;
	bool stopEncode;
	bool isEncoding;

	std::mutex bufferMutex;
	std::condition_variable bufferSignal;
};


typedef IFuseNode <VidParams> IVidNode;
typedef  FuseNode <VidParams>  VidNode;
typedef IFuseGraph<VidParams> IVidGraph;
typedef  FuseGraph<VidParams>  VidGraph;

typedef PathObject<VidParams>  VidPath;

class VidFuse;
class VidRootNode;
class VidOptNode;
class VidBitrateNode;
class VidParamDumpNode;
class VidFilesNode;



class VidFilesNode : public IVidNode, public FuseFDCallback {
	TransparentFuse trans;
public:
	VidFilesNode(IVidGraph *graph, const std::string &baseDir) : IVidNode(graph), trans(baseDir) {
		av_register_all();
		avfilter_register_all();
		av_log_set_level(AV_LOG_INFO);
		av_log(NULL, AV_LOG_INFO, "AV LOG LEVEL: %d\n", av_log_get_level());
	}

	void setBaseDir(const std::string &baseDir) {
		trans.setBaseDir(baseDir);
	}

	std::pair<bool,IFuseNode*> getNextNode(VidPath &path) {
		VidPath end = path.end();
		std::string subpath = path.to(end);
		int res = trans.access(subpath.c_str(), F_OK);
		cout << subpath << ", res: " << res << endl;
		if (res == 0) {
			path = end;
			path.data->filepath = subpath;
			path.data->fullpath = trans.fullpath(subpath.c_str());
			return std::pair<bool,IFuseNode*>(false, this);
		}
		return std::pair<bool,IFuseNode*>(false, nullptr);
	}

	int getattr(VidPath path, struct stat *statbuf) {
		cout << path.data->filepath.c_str() << endl;
		return trans.getattr(path.data->filepath.c_str(), statbuf);
	}

	int readlink(VidPath path, char *link, size_t size) {
		return trans.readlink(path.data->filepath.c_str(), link, size);
	}

	int access(VidPath path, int mask) {
		return trans.access(path.data->filepath.c_str(), mask);
	}

	openres open(VidPath path, struct fuse_file_info *fi) {
		try {
			fi->fh = (fuse_fh_t) (new Transcoder(path.data));
			return openres(0, this);
		} catch (TranscoderError &e) {
			av_log(NULL, AV_LOG_ERROR, "%s\n", e.what());
		}
		return openres(trans.open(path.data->filepath.c_str(), fi), &trans);
	}

	int read(const char *path, char *buf, size_t size, off_t offset, struct fuse_file_info *fi) {
		Transcoder &tr = *(Transcoder*)fi->fh;
		return tr.read(buf, size, offset);
	}

	int release(const char *path, struct fuse_file_info *fi) {
		delete (Transcoder*)fi->fh;
		return 0;
	}

	openres opendir(VidPath path, struct fuse_file_info *fi) {
		return openres(trans.opendir(path.data->filepath.c_str(), fi), &trans);
	}
	int readdir(const char *path, void *buf, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info *fi) {
		return 0;
	}
	int releasedir(const char *path, fuse_file_info *fi) {
		return 0;
	}

	int flush (const char *path, struct fuse_file_info *fi) {
		return 0;
	}

	int fgetattr(const char *path, struct stat *statbuf, struct fuse_file_info *fi) {
		Transcoder &tr = *(Transcoder*)fi->fh;
		int result = trans.getattr(tr.getParams()->filepath.c_str(), statbuf);
		//statbuf->st_size = 100;
		return result;
	}
};


class VidParamDumpNode : public VidNode {
public:
	VidParamDumpNode(IVidGraph *graph) : VidNode(graph, 1000, 1000, S_IFREG | 0444) {}

	int getattr(VidPath path, struct stat *statbuf) {
		statbuf->st_size = 0;
		return this->VidNode::getattr(path, statbuf);
	}

	openres open(VidPath path, struct fuse_file_info *fi) {
		VidPath *np = new VidPath(path);
		fi->fh = (fuse_fh_t) np;
		fi->direct_io = 1;
		fi->nonseekable = 1;
		np->data->dumpsize = 0;
		return openres(0, this);
	}

	int read(const char *path, char *buf, size_t size, off_t offset, struct fuse_file_info *fi) {
		VidPath *p = (VidPath*) fi->fh;
		std::stringstream ss;
		ss << "Bitrate: " << p->data->bitrate << endl;
		cout << "\t\t\tOFFSET: " << offset << endl;
		if ((size_t)offset >= ss.str().size())
			return 0;
		p->data->dumpsize = ss.str().size();
		strncpy(buf, ss.str().c_str() + offset, size);
		return ((ss.str().size() - offset) > size) ? size : ss.str().size() - offset;
	}
	
	int release(const char *path, struct fuse_file_info *fi) {
		delete (VidPath*)fi->fh;
		return 0;
	}

	int fgetattr(const char *path, struct stat *statbuf, struct fuse_file_info *fi) {
		statbuf->st_size = ((VidPath*)fi->fh)->data->dumpsize;
		return this->VidNode::fgetattr(path, statbuf, fi);
	}
};

class VidBitrateNode : public VidNode {
	VidNode *optNode;
public:
	VidBitrateNode(IVidGraph *graph, VidNode *optNode) : VidNode(graph, 1000, 1000), optNode(optNode) {}

	std::pair<bool,IVidNode*> getNextNode(VidPath &path) {
		if (path.isEnd())
			return std::pair<bool,IVidNode*>(false, this);
		// parse bitrate
		unsigned int bitrate = 0;
		for (const char *c = *path; *c; ++c) {
			if (*c < '0' || *c > '9')
				return std::pair<bool,IVidNode*>(false,nullptr);
			bitrate = bitrate * 10 + (*c - '0');
		}
		path.data->bitrate = bitrate;
		++path;
		return std::pair<bool,IVidNode*>(true,optNode);
	}

	int readdir(const char *path, void *buf, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info *fi) {
		struct stat stats, *statbuf = nullptr;
		if (0 == optNode->getattr(PathSplitter::New("")->begin<VidParams>(), &stats))
			statbuf = &stats;
		const char *bitrates[] = {"00240", "00800", "01200", "02000", "04000", "08000", "11650"};
		while (offset < 7) {
			cout << offset << endl;
			if (filler(buf, bitrates[offset], statbuf, offset + 1))
				return 0;
			++offset;
		}
		return 0;
	}
};

class SpeedTestNode : public IVidNode, public FuseFDCallback {
public:
	SpeedTestNode(IVidGraph *graph) : IVidNode(graph) {}

	std::pair<bool,IVidNode*> getNextNode(VidPath &path) {
		if (path.isEnd())
			return std::pair<bool,IVidNode*>(false, this);
		return std::pair<bool,IVidNode*>(false,nullptr);
	}
	int getattr(VidPath path, struct stat *statbuf) {
		statbuf->st_uid = 1000;
		statbuf->st_gid = 1000;
		statbuf->st_mode = S_IFREG | 0555;
		statbuf->st_size = -1;
		return 0;
	}
	int access(VidPath path, int mask) {
		return 0;
	}
	openres open(VidPath path, struct fuse_file_info *fi) {
		fi->nonseekable = 1;
		fi->direct_io = 1;
		return openres(0, this);
	}
	int read(const char *path, char *buf, size_t size, off_t offset, struct fuse_file_info *fi) {
		return size;
	}
	int release(const char *path, struct fuse_file_info *fi) {
		return 0;
	}
};

class VidFuse : public VidGraph {
	VidFilesNode *filesnode;
public:
	VidFuse(const char *rootpath = "/") {
		//this->root = this->registerNewNode(new VidRootNode(this));
		VidNode *vidroot = new VidNode(this, 1000, 1000);
		VidNode *optnode = new VidNode(this, 1000, 1000);
		filesnode = new VidFilesNode(this, rootpath);
		vidroot->registerNewNode("files", filesnode);
		vidroot->registerNewNode("options", optnode);
		optnode->registerNewNode("bitrate", new VidBitrateNode(this, optnode));
		optnode->registerNewNode("dump", new VidParamDumpNode(this));
		optnode->addExistingNode("files", filesnode);
		vidroot->registerNewNode("speedtest", new SpeedTestNode(this));
		this->root = vidroot;
		this->registerNewNode(vidroot);
	}
	
	std::shared_ptr<VidParams> newData() {
		return std::shared_ptr<VidParams>(new VidParams);
	}

	int mount(int argc, char **argv) {
		if (argc > 2) {
			char *rootpath = argv[argc-2];
			argv[argc-2] = argv[argc-1];
			argc--;
			filesnode->setBaseDir(rootpath);
		}
		return VidGraph::mount(argc, argv);
	}
};


int main(int argc, char **argv) {
	VidFuse v;
	v.mount(argc, argv);
}
