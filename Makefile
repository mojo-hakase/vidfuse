vidfuse: vidfuse.cpp
	$(CXX) -std=c++11 -o $@ vidfuse.cpp -Icppfuse -D_FILE_OFFSET_BITS=64 -lfuse -lavutil -lavformat -lavfilter -lavcodec -lpthread -Wall -Wno-unused -g
