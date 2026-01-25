CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c99
LDFLAGS = -lm
ASM = python3 assembler.py

KERNELS = kernels/matmul_q8.bin kernels/rmsnorm.bin kernels/rope.bin kernels/attention.bin \
          kernels/silu_mul.bin kernels/residual.bin kernels/embed.bin kernels/memcpy.bin \
          kernels/forward.bin

all: test_emulator run_model generate

kernels: $(KERNELS)

kernels/%.bin: kernels/%.s assembler.py
	$(ASM) $< -o $@

test_emulator: test_emulator.c emulator.c emulator.h
	$(CC) $(CFLAGS) -o $@ test_emulator.c emulator.c $(LDFLAGS)

run_model: run_model.c emulator.c emulator.h $(KERNELS)
	$(CC) $(CFLAGS) -o $@ run_model.c emulator.c $(LDFLAGS)

generate: generate.c emulator.c emulator.h $(KERNELS)
	$(CC) $(CFLAGS) -o $@ generate.c emulator.c $(LDFLAGS)

test: test_emulator
	./test_emulator

run: run_model
	./run_model

clean:
	rm -f test_emulator run_model generate *.o kernels/*.bin
