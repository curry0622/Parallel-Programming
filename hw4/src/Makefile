CXX = mpicxx
CXXFLAGS = -O3 -lm -pthread -std=c++17
TARGETS = mapreduce

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)

.PHONY: test
test: $(TARGETS)
	python3 ../scripts/test.py -t $(t)
