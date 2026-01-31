

NVCC := /usr/local/cuda/bin/nvcc
CFLAGS := -O2 --std=c++17
EXE1 := knn
EXE2 := pca_knn
SRC1 := knn.cu
SRC2 := pca_knn.cu

# Use pkg-config to get OpenCV flags (opencv4)
OPENCV_CFLAGS := $(shell pkg-config --cflags --libs opencv4)

all: build

build: build_knn build_pca

.PHONY: build_knn
build_knn:
	mkdir -p ../../bin
	$(NVCC) $(SRC1) $(CFLAGS) $(OPENCV_CFLAGS) -o ../../bin/$(EXE1)

.PHONY: build_pca
build_pca:
	mkdir -p ../../bin
	$(NVCC) $(SRC2) $(CFLAGS) -lcublas -o ../../bin/$(EXE2)

.PHONY: run
run: build
	@echo "Available executables: ../../bin/$(EXE1) and ../../bin/$(EXE2)"

.PHONY: clean
clean:
	rm -f ../../bin/$(EXE1) ../../bin/$(EXE2)
