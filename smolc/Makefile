# SmolLM2 C Inference Engine Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native
LDFLAGS = -lm

# Debug build
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -fsanitize=address

TARGET_BASIC = smolc
TARGET_FULL = smolc_full
SRC_BASIC = smolc.c
SRC_FULL = smolc_full.c

.PHONY: all clean debug run test help

all: $(TARGET_BASIC) $(TARGET_FULL)

$(TARGET_BASIC): $(SRC_BASIC) smolc.h
	$(CC) $(CFLAGS) -o $(TARGET_BASIC) $(SRC_BASIC) $(LDFLAGS)

$(TARGET_FULL): $(SRC_FULL) smolc.h
	$(CC) $(CFLAGS) -o $(TARGET_FULL) $(SRC_FULL) $(LDFLAGS)

debug: $(TARGET_BASIC)_debug $(TARGET_FULL)_debug

$(TARGET_BASIC)_debug: $(SRC_BASIC) smolc.h
	$(CC) $(DEBUG_CFLAGS) -o $(TARGET_BASIC)_debug $(SRC_BASIC) $(LDFLAGS)

$(TARGET_FULL)_debug: $(SRC_FULL) smolc.h
	$(CC) $(DEBUG_CFLAGS) -o $(TARGET_FULL)_debug $(SRC_FULL) $(LDFLAGS)

clean:
	rm -f $(TARGET_BASIC) $(TARGET_FULL) $(TARGET_BASIC)_debug $(TARGET_FULL)_debug

run: $(TARGET_BASIC)
	./$(TARGET_BASIC) -m ../models/smollm2-135m-q8.bin -p "Hello, world!" -n 20

test: $(TARGET_BASIC)
	./$(TARGET_BASIC) -m ../models/smollm2-135m-q8.bin -p "The capital of France is" -n 30

help:
	@echo "Targets:"
	@echo "  all     - Build optimized binary"
	@echo "  smolc   - Build Q8-only binary"
	@echo "  smolc_full - Build Q4/Q8 binary"
	@echo "  debug   - Build debug binary with sanitizers"
	@echo "  clean   - Remove built files"
	@echo "  run     - Build and run with sample prompt"
	@echo "  test    - Build and test with verification prompt"
