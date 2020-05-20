DEFINES	= $(DEF_SWITCH)

IMAGE_JPEG  = ./dtc_test_images/
EXENUM   = 100
BATCHNUM = 100
THRESH = 0.5

CC		= gcc
CXX		= g++
CFLAGS		= -O3 -Wno-unused-result -Wno-write-strings -Wno-format-extra-args $(DEFINES)
CXXFLAGS	= -O3 -fopenmp -std=c++1z -Wno-unused-result -Wno-write-strings -Wno-format-extra-args -Wno-pointer-arith $(DEFINES) `pkg-config --cflags opencv4`
INCPATH		= -I../lib
LINK		= g++ -fopenmp
LFLAGS		= -Wl,-O1 `pkg-config --libs opencv4`
DEL_FILE	= rm -f

OBJECTS_DIR	= ./

HEADERS		= ./axis.h ./ui.hpp ./dtype.hpp ./detector.hpp ./dlaip_fpga.hpp 
OBJECTS		= dlaip_fpga.o detector.o dlaip.o
TARGET		= dlaip

APP_I8A64  = dlaip_i8_a64

.cpp.o:
	$(CXX)  -c $(CXXFLAGS)  $(INCPATH) -o "$@" "$<"

.PHONY: all
all: dlaip_i8_a64

$(TARGET): $(OBJECTS)
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) -lstdc++ -lstdc++fs

dlaip.o: $(HEADERS)

                                                           
dlaip_i8_a64: DEF_SWITCH = -D__INT8__=1 -DAXI_NBIT=64 -DFIXED_NBIT=8
dlaip_i8_a64: $(TARGET)
	mv $(TARGET) $@

.PHONY: clean
clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)


run:
	./$(APP_I8A64) --verbose 0 --image $(IMAGE_JPEG) -e $(EXENUM)  -b $(BATCHNUM) --thresh $(THRESH)  --nms 0.8 1 2>&1 | tee run.log

valid:
	./$(APP_I8A64) --verbose 0 --image $(IMAGE_JPEG) -b $(BATCHNUM) --thresh 0.01 --valid --nms 0.8 1 2>&1 | tee valid.log

