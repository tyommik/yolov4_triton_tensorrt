CC=g++
LD=ld
CXXFLAGS=-Wall -std=c++11 -g -O

NVCC=nvcc

# space separated compute values ex: computes=70 75. If not present will fetch device's CC
computes=70 86

ifeq ($(computes), )
  computes= $(shell python gpu_cc.py)
  $(info computes: $(computes))
endif

NVCCFLAGS= $(foreach compute, $(computes),-gencode arch=compute_$(compute),code=[sm_$(compute),compute_$(compute)])
$(info NVCCFLAGS: $(NVCCFLAGS))

#NVCCFLAGS=-m64 -gencode arch=compute_$(compute),code=sm_$(compute) \
#               -gencode arch=compute_$(compute),code=compute_$(compute)

# Let nvcc determine cuda compute automatically
NVCCFLAGS=

# These are the directories where I installed TensorRT on my x86_64 PC.
TENSORRT_INCS=-I"/usr/lib/x86_64-linux-gnu/include"
TENSORRT_LIBS=-L"/usr/lib/x86_64-linux-gnu/lib"

# INCS and LIBS
INCS=-I"/usr/local/cuda/include" $(TENSORRT_INCS) -I"/usr/local/include" -I"plugin"
LIBS=-L"/usr/local/cuda/lib64" $(TENSORRT_LIBS) -L"/usr/local/lib" -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group

.PHONY: all clean

all: libyolo_layer.so

clean:
	rm -f *.so *.o

libyolo_layer.so: yolo_layer.o
	$(CC) -shared -o $@ $< $(LIBS)

yolo_layer.o: yolo_layer.cu yolo_layer.h
	$(NVCC) -ccbin $(CC) $(INCS) $(NVCCFLAGS) -Xcompiler -fPIC -c -o $@ $<
