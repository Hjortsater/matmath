CC = clang
CFLAGS = -O3 -fPIC -Wall -Wextra -fopenmp
LDFLAGS = -shared
TARGET = libcmat.so
SRC = hjortmath/cmat.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o