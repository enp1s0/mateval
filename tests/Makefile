CXX=g++
CXXFLAGS=-std=c++14 -I../include
TARGET=comparison.test

all: $(TARGET)

%.test:%.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm *.test
