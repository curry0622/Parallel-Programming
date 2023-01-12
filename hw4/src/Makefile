CXX := mpicxx
CXXFLAGS := -std=c++11 -O3 -march=native -pipe -pthread
LIBS := -lm
EXEC := mapreduce
SRC_DIRS := .
SRCS := main.cc $(wildcard $(SRC_DIRS:=/*.cpp))
OBJS := $(SRCS:.cpp=.o)

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LIBS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(EXEC) *.o

test: $(EXEC)
	python3 ../scripts/test.py -t $(t)
