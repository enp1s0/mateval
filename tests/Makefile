CXX=g++
CXXFLAGS=-std=c++14 -I../include
TARGET=comparison.test cond.test

all: $(TARGET)

%.test:%.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

cond.test:cond.cpp
	$(CXX) $< $(CXXFLAGS) -llapacke -llapack -lblas -lgfortran -o $@


clean:
	rm *.test
