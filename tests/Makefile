CXX=g++
CXXFLAGS=-std=c++14 -I../include -fopenmp
NVCC=nvcc
NVCCFLAGS=-std=c++14 -Xcompiler="${CXXFLAGS}" -L../build -lmateval_cuda -arch=sm_80 -lcurand -lcusolver
TARGET=comparison.test cond.test cuda.test quad.test norm.test

all: $(TARGET)

%.test:%.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

cond.test:cond.cpp
	$(CXX) $< $(CXXFLAGS) -llapacke -llapack -lblas -lgfortran -o $@

%.test:%.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm *.test
