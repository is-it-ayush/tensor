CC=gcc
CFLAGS=-Wall -Wextra -Werror -std=c99 -pedantic -ggdb
LIBS=

all: main

main: main.o
	$(CC) $(CFLAGS) -o main main.o $(LIBS)

main.o:
	$(CC) $(CFLAGS) -c ./tests/main.c $(LIBS)

clean:
	rm -f main main.o
