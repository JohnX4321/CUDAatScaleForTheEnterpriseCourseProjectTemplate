NVCC := /usr/local/cuda/bin/nvcc
CFLAGS := -O2 --std=c++17
EXE := knn
SRC := knn.cu

all: build

build:
	mkdir -p ../../bin
	$(NVCC) $(SRC) $(CFLAGS) -o ../../bin/$(EXE)

run: build
	../../bin/$(EXE) $(ARGS)

clean:
	rm -f ../../bin/$(EXE)
