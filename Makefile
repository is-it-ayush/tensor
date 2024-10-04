CC=gcc
CFLAGS=-Wall -Wextra -std=c11 -pedantic -ggdb -O3 -march=native -fsanitize=address,undefined -fno-omit-frame-pointer -fdiagnostics-show-option
LIBS=-lm

all: test

test: test.o libmave.o
	$(CC) $(CFLAGS) -o test test.o libmave.o $(LIBS)

test.o:
	$(CC) $(CFLAGS) -c ./tests/test.c $(LIBS)

libmave.o:
	$(CC) $(CFLAGS) -c ./libmave.c $(LIBS)

clean:
	rm -f test test.o libmave.o
