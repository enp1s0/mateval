CXX=g++
CXXFLAGS=std=c++14 -I../include
TARGET=mateval.test

all: $(TARGET)

%.test:%.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

clean:
	rm *.test
