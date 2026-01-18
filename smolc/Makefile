# SmolLM2 C Inference Engine Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native
LDFLAGS = -lm

# Debug build
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -fsanitize=address

TARGET_BASIC = smolc
TARGET_FULL = smolc_full
TARGET_GGUF = smolc_gguf
SRC_BASIC = smolc.c
SRC_FULL = smolc_full.c
SRC_GGUF = smolc_gguf.c

.PHONY: all clean debug run test help

all: $(TARGET_BASIC) $(TARGET_FULL) $(TARGET_GGUF)

$(TARGET_BASIC): $(SRC_BASIC) smolc.h
	$(CC) $(CFLAGS) -o $(TARGET_BASIC) $(SRC_BASIC) $(LDFLAGS)

$(TARGET_FULL): $(SRC_FULL) smolc.h
	$(CC) $(CFLAGS) -o $(TARGET_FULL) $(SRC_FULL) $(LDFLAGS)

$(TARGET_GGUF): $(SRC_GGUF) smolc_gguf.h
	$(CC) $(CFLAGS) -o $(TARGET_GGUF) $(SRC_GGUF) $(LDFLAGS)

debug: $(TARGET_BASIC)_debug $(TARGET_FULL)_debug $(TARGET_GGUF)_debug

$(TARGET_BASIC)_debug: $(SRC_BASIC) smolc.h
	$(CC) $(DEBUG_CFLAGS) -o $(TARGET_BASIC)_debug $(SRC_BASIC) $(LDFLAGS)

$(TARGET_FULL)_debug: $(SRC_FULL) smolc.h
	$(CC) $(DEBUG_CFLAGS) -o $(TARGET_FULL)_debug $(SRC_FULL) $(LDFLAGS)

$(TARGET_GGUF)_debug: $(SRC_GGUF) smolc_gguf.h
	$(CC) $(DEBUG_CFLAGS) -o $(TARGET_GGUF)_debug $(SRC_GGUF) $(LDFLAGS)

clean:
	rm -f $(TARGET_BASIC) $(TARGET_FULL) $(TARGET_GGUF) $(TARGET_BASIC)_debug $(TARGET_FULL)_debug $(TARGET_GGUF)_debug

run: $(TARGET_BASIC)
	./$(TARGET_BASIC) -m ../models/smollm2-135m-q8.bin -p "Hello, world!" -n 20

test: $(TARGET_BASIC)
	./$(TARGET_BASIC) -m ../models/smollm2-135m-q8.bin -p "The capital of France is" -n 30

help:
	@echo "Targets:"
	@echo "  all        - Build all binaries (smolc, smolc_full, smolc_gguf)"
	@echo "  smolc      - Build Q8-only binary (custom format)"
	@echo "  smolc_full - Build Q4/Q8 binary (custom format)"
	@echo "  smolc_gguf - Build GGUF Q8_0 loader"
	@echo "  debug      - Build debug binaries with sanitizers"
	@echo "  clean      - Remove built files"
	@echo "  run        - Build and run with sample prompt"
	@echo "  test       - Build and test with verification prompt"
