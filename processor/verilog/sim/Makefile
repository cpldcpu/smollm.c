# SMOL-32 Verilog Simulation Makefile

# Tools
IVERILOG = iverilog
VVP = vvp
VERILATOR = verilator
GTKWAVE = gtkwave

# Directories
RTL_DIR = ../rtl
TB_DIR = ../tb
TRACE_DIR = ../traces

# Source files
RTL_SOURCES = \
	$(RTL_DIR)/smol32_defines.vh \
	$(RTL_DIR)/smol32_regfile.v \
	$(RTL_DIR)/smol32_alu.v \
	$(RTL_DIR)/smol32_decode.v \
	$(RTL_DIR)/smol32_control.v \
	$(RTL_DIR)/smol32_core.v \
	$(RTL_DIR)/smol32_top.v

# Include path
INC = -I$(RTL_DIR)

# Default target
.PHONY: all
all: test

# Compile with iverilog
.PHONY: compile
compile: smol32_test

smol32_test: $(RTL_SOURCES) $(TB_DIR)/tb_smol32_top.v
	$(IVERILOG) $(INC) -o $@ $(TB_DIR)/tb_smol32_top.v \
		$(RTL_DIR)/smol32_regfile.v \
		$(RTL_DIR)/smol32_regfile_fp.v \
		$(RTL_DIR)/smol32_regfile_vec.v \
		$(RTL_DIR)/smol32_alu.v \
		$(RTL_DIR)/smol32_fpu.v \
		$(RTL_DIR)/smol32_vecunit.v \
		$(RTL_DIR)/smol32_vecmem.v \
		$(RTL_DIR)/smol32_q8mac.v \
		$(RTL_DIR)/smol32_decode.v \
		$(RTL_DIR)/smol32_control.v \
		$(RTL_DIR)/smol32_core.v \
		$(RTL_DIR)/smol32_top.v

# Run simulation
.PHONY: test
test: smol32_test
	$(VVP) smol32_test

# Kernel test
.PHONY: kernel_test
kernel_test: smol32_kernel_test
	$(VVP) smol32_kernel_test

smol32_kernel_test: $(RTL_SOURCES) $(TB_DIR)/tb_kernel_matmul.v
	$(IVERILOG) $(INC) -o $@ $(TB_DIR)/tb_kernel_matmul.v \
		$(RTL_DIR)/smol32_regfile.v \
		$(RTL_DIR)/smol32_regfile_fp.v \
		$(RTL_DIR)/smol32_regfile_vec.v \
		$(RTL_DIR)/smol32_alu.v \
		$(RTL_DIR)/smol32_fpu.v \
		$(RTL_DIR)/smol32_vecunit.v \
		$(RTL_DIR)/smol32_vecmem.v \
		$(RTL_DIR)/smol32_q8mac.v \
		$(RTL_DIR)/smol32_decode.v \
		$(RTL_DIR)/smol32_control.v \
		$(RTL_DIR)/smol32_core.v \
		$(RTL_DIR)/smol32_top.v

# View waveforms
.PHONY: wave
wave: smol32_test.vcd
	$(GTKWAVE) smol32_test.vcd &

smol32_test.vcd: test

# Verilator compile and run
.PHONY: verilate
verilate: obj_dir/Vsmol32_top
	./obj_dir/Vsmol32_top

obj_dir/Vsmol32_top: $(TB_DIR)/tb_verilator.cpp
	$(VERILATOR) --cc --exe --build --trace -Wall \
		-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-CASEINCOMPLETE \
		-Wno-LATCH -Wno-BLKSEQ -Wno-UNOPTFLAT \
		--public-flat-rw \
		$(INC) \
		--top-module smol32_top \
		$(RTL_DIR)/smol32_regfile.v \
		$(RTL_DIR)/smol32_regfile_fp.v \
		$(RTL_DIR)/smol32_regfile_vec.v \
		$(RTL_DIR)/smol32_alu.v \
		$(RTL_DIR)/smol32_fpu.v \
		$(RTL_DIR)/smol32_vecunit.v \
		$(RTL_DIR)/smol32_vecmem.v \
		$(RTL_DIR)/smol32_q8mac.v \
		$(RTL_DIR)/smol32_decode.v \
		$(RTL_DIR)/smol32_control.v \
		$(RTL_DIR)/smol32_core.v \
		$(RTL_DIR)/smol32_top.v \
		$(TB_DIR)/tb_verilator.cpp \
		-o Vsmol32_top

# Lint check
.PHONY: lint
lint:
	$(VERILATOR) --lint-only -Wall $(INC) \
		$(RTL_DIR)/smol32_regfile.v \
		$(RTL_DIR)/smol32_alu.v \
		$(RTL_DIR)/smol32_decode.v \
		$(RTL_DIR)/smol32_control.v \
		$(RTL_DIR)/smol32_core.v \
		$(RTL_DIR)/smol32_top.v

# Clean
.PHONY: clean
clean:
	rm -f smol32_test smol32_test.vcd smol32_kernel_test tb_kernel_matmul.vcd
	rm -rf obj_dir

.PHONY: help
help:
	@echo "SMOL-32 Verilog Simulation"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Run test (default)"
	@echo "  compile  - Compile with iverilog"
	@echo "  test     - Run simulation"
	@echo "  wave     - View waveforms in GTKWave"
	@echo "  verilate - Compile with Verilator"
	@echo "  lint     - Run Verilator lint check"
	@echo "  clean    - Remove generated files"
