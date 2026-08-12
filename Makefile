CPP=mpicxx
CPPFLAGS= -O3

.PHONY: all clean

all: engine engine.debug

clean:
	rm -rf engine engine.debug

engine: engine.cpp common.cpp
	${CPP} ${CPPFLAGS} $^ -o $@

engine.debug: engine.cpp common.cpp
	${CPP} ${CPPFLAGS} -DDEBUG $^ -o $@