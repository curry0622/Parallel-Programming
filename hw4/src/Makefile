CC = mpicc
CXX = mpicxx
CFLAGS = -lm -O3 -march=native -pipe -pthread
CXXFLAGS = $(CFLAGS)
TARGETS = mapreduce

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)

.PHONY: test
test: $(TARGETS)
	python3 ../scripts/test.py -t $(t)
