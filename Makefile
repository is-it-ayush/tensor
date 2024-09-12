CC=gcc
CFLAGS=-Wall -Wextra -Werror -std=c99 -pedantic -ggdb -O3 -fsanitize=address,undefined -fno-omit-frame-pointer
LIBS=

all: main

main: main.o libmave.o
	$(CC) $(CFLAGS) -o main main.o libmave.o $(LIBS)

main.o:
	$(CC) $(CFLAGS) -c ./tests/main.c $(LIBS)

libmave.o:
	$(CC) $(CFLAGS) -c ./libmave.c $(LIBS)

clean:
	rm -f main main.o libmave.o
