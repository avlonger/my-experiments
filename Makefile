CC=cc
CFLAGS=-Wall -O3 -std=gnu99
LDFLAGS=-lm
BIN_PATH=bin
SRC_PATH=src

all: dirs duval kmp exp borderless border experiment

dirs:
	mkdir -p $(BIN_PATH)

duval:
	$(CC) $(CFLAGS) $(SRC_PATH)/duval.c $(SRC_PATH)/algo/duval.c $(LDFLAGS) -o $(BIN_PATH)/duval

kmp:
	$(CC) $(CFLAGS) $(SRC_PATH)/kmp.c $(SRC_PATH)/algo/kmp.c $(LDFLAGS) -o $(BIN_PATH)/kmp

exp:
	$(CC) $(CFLAGS) $(SRC_PATH)/algo/*.c $(SRC_PATH)/exp.c $(LDFLAGS) -o $(BIN_PATH)/exp

experiment:
	$(CC) $(CFLAGS) $(SRC_PATH)/algo/*.c $(SRC_PATH)/experiment.c $(LDFLAGS) -o $(BIN_PATH)/experiment

borderless:
	$(CC) $(CFLAGS) $(SRC_PATH)/borderless.c $(SRC_PATH)/algo/borderless.c $(LDFLAGS) $(SRC_PATH)/algo/kmp.c -o $(BIN_PATH)/borderless

border:
	$(CC) $(CFLAGS) $(SRC_PATH)/border.c $(SRC_PATH)/algo/kmp.c $(LDFLAGS) -o $(BIN_PATH)/border

clean:
	rm -rf $(BIN_PATH)
