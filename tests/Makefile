CXX=g++
CXXFLAGS=-std=c++14 -I../include -fopenmp
TARGET=mateval.test

all: $(TARGET)

%.test:%.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm *.test
