ROOT=$(shell pwd)
include Makefile.vars

all: 
	 @$(MAKE) -C yoloOpenVX $<

test: FORCE
	for vid in $(wildcard media/*.mp4); do \
		echo processing $$vid ;\
		/usr/bin/python ./main.py --video $$vid ;\
	done

clean:
	 @$(MAKE) -C yoloOpenVX clean

FORCE:


