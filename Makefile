# make            -> serial + openmp (CPU versions)
# make cuda       -> cuda_filter (needs nvcc, e.g. on Google Colab)
# make clean

CC      ?= cc
CFLAGS  ?= -O2 -std=gnu11 -Wall -Wextra
NVCC    ?= nvcc
# sm_75 = Tesla T4 (Colab's free GPU); newer GPUs still run it.
NVFLAGS ?= -O2 -arch=sm_75

ifeq ($(shell uname),Darwin)
  # Apple clang needs Homebrew's OpenMP runtime: brew install libomp
  LIBOMP   := $(shell brew --prefix libomp 2>/dev/null)
  OMPFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  OMPLIBS  := -L$(LIBOMP)/lib -lomp
else
  OMPFLAGS := -fopenmp
  OMPLIBS  := -fopenmp
endif

all: serial openmp

serial: serial.c common.h
	$(CC) $(CFLAGS) -o $@ serial.c -lm

openmp: openmp.c common.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ openmp.c $(OMPLIBS) -lm

cuda: cuda_filter

cuda_filter: cuda.cu common.h
	$(NVCC) $(NVFLAGS) -o $@ cuda.cu

clean:
	rm -f serial openmp cuda_filter

.PHONY: all cuda clean
