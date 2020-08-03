INCFLAGS = -I/usr/local/include/eigen3 -I../src/ -I.
CPPFLAGS = -O3 $(INCFLAGS) -DEIGEN_NDEBUG -fopenmp -Wall -Wno-strict-aliasing
LINKFLAGS = -lz
DEBUGFLAGS = -g -ggdb $(INCFLAGS)

ifeq ($(UNAME), Linux)
	CPP = g++
else
	CPP = g++-10
endif

headers=$(wildcard *.h**)
all: $(patsubst %.cpp, %, $(wildcard *.cpp))

%: %.cpp $(headers)
	$(CPP) $(CPPFLAGS) $< -o $@ $(LINKFLAGS)

clean:
	rm -f $(patsubst %.cpp, %, $(wildcard *.cpp))

