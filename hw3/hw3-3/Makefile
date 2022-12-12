NVFLAGS := -std=c++11 -Xptxas="-v" -arch=sm_61 
CFLAGS  := -O3
XCFLAGS := -fopenmp
LDFLAGS := -lm
EXES    := hw3-3

all: $(EXES)

clean:
	rm -f $(EXES)

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(CFLAGS) $(LDFLAGS) -Xcompiler="$(XCFLAGS)" -o $@ $?