CPP=g++
NVCC = nvcc

CFLAGS=$(OPT) --std=c++11 -g -ggdb -gdwarf-3 -O3
MODULE          := gbdt

.PHONY: all clean

all: $(MODULE)
gbdt: gbdt.cu
	$(NVCC) $^ -o $@ -std=c++11

clean:
	@rm -f $(MODULE) 