ROOT=$(shell dirname $(shell pwd))
include ../Makefile.vars

TARGET_LIB = libyoloOpenVX$(LIB_EXT)

SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:%.cpp=%.o)

.PHONY: all clean

all: INSTALL

INSTALL: ${TARGET_LIB}
	$(CP) ${TARGET_LIB} $(LIB_PATH)

$(TARGET_LIB): $(OBJS)
	$(CC) ${CFLAGS} ${LDFLAGS} -o $@ $^ $(wildcard $(ROOT)/lib/libann*.so)
	@echo done compiling openVX model for yolo network.

%.o : %.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ *pyc $(TARGET_LIB)


