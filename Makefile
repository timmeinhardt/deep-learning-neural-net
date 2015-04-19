run: clean all
	./network

all: main.o misc.o network.o loader.o
	g++ main.o network.o misc.o -o network -lgsl -lgslcblas

main.o: main.cpp
	g++ -Wall -std=c++11 -c main.cpp

network.o: Network.cpp Network.h
	g++ -Wall -std=c++11 -c Network.cpp

misc.o: misc.cpp misc.h
	g++ -Wall -std=c++11 -c misc.cpp

loader.o: loader.cpp loader.h
	g++ -Wall -std=c++11 -c loader.cpp

clean:
	rm -f *o network
