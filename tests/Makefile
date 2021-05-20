CXX=g++
CXXFLAGS=-std=c++14 -I../include -fopenmp
TARGET=comparison.test cond.test
TARGET=mateval.test

all: $(TARGET)

%.test:%.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

cond.test:cond.cpp
	$(CXX) $< $(CXXFLAGS) -llapacke -llapack -lblas -lgfortran -o $@


clean:
	rm *.test
