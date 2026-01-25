#!/usr/bin/env python3
"""
SMOL-32 Assembler
Assembles SMOL-32 assembly to binary machine code
"""

import re
import sys
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Register mappings
REGS = {f'R{i}': i for i in range(32)}
REGS.update({'ZERO': 0, 'RA': 1, 'SP': 2, 'FP': 28, 'GP': 29})

FREGS = {f'F{i}': i for i in range(32)}

VREGS = {f'V{i}': i for i in range(8)}

# Opcodes
OP = {
    'LW': 0x00, 'SW': 0x01, 'LF': 0x02, 'SF': 0x03,
    'ALU': 0x04, 'ALUI': 0x05,
    'FPU': 0x08, 'FPUI': 0x09,
    'FSPEC': 0x0C,
    'VARITH': 0x10, 'VSCALAR': 0x11, 'VRED': 0x12,
    'VSPEC': 0x18, 'VLOAD': 0x19, 'VSTORE': 0x1A,
    'Q8SET': 0x1C, 'Q8MAC': 0x1D, 'VQ8': 0x1E,
    'ROPE': 0x20, 'VRMS': 0x21, 'VSOFTMAX': 0x22,
    'BRANCH': 0x30, 'JAL': 0x31, 'JALR': 0x32, 'LOOP': 0x33,
    'SYSTEM': 0x38,
}

# ALU functions
ALU_FUNC = {
    'ADD': 0x00, 'SUB': 0x01, 'MUL': 0x02, 'DIV': 0x03,
    'AND': 0x05, 'OR': 0x06, 'XOR': 0x07,
    'SLL': 0x08, 'SRL': 0x09, 'SRA': 0x0A,
    'SLT': 0x0B, 'SLTU': 0x0C,
    'MV': 0x00,  # MV is ADD with rs2=0
}

# FPU functions
FPU_FUNC = {
    'FADD': 0x00, 'FSUB': 0x01, 'FMUL': 0x02, 'FDIV': 0x03,
    'FMIN': 0x04, 'FMAX': 0x05,
    'FMADD': 0x06, 'FMSUB': 0x07,
    'FCVT.W.S': 0x0A, 'FCVT.S.W': 0x0B,
    'FMOV': 0x0C, 'FABS': 0x0D, 'FNEG': 0x0E,
}

# FP Special functions
FSPEC_FUNC = {
    'FSQRT': 0x00, 'FRSQRT': 0x01, 'FRECIP': 0x02,
    'FEXP': 0x03, 'FLOG': 0x04,
    'FSIN': 0x05, 'FCOS': 0x06,
    'FSILU': 0x07, 'FGELU': 0x08,
    'FTANH': 0x09, 'FSIGMOID': 0x0A,
}

# Vector functions
VARITH_FUNC = {
    'VADD': 0x0, 'VSUB': 0x1, 'VMUL': 0x2, 'VDIV': 0x3,
    'VFMADD': 0x4, 'VMIN': 0x5, 'VMAX': 0x6,
}

VSCALAR_FUNC = {
    'VADDS': 0x0, 'VSUBS': 0x1, 'VMULS': 0x2, 'VDIVS': 0x3,
}

VRED_FUNC = {
    'VREDSUM': 0x0, 'VREDMAX': 0x1, 'VREDMIN': 0x2, 'VREDSQS': 0x3,
}

VSPEC_FUNC = {
    'VSQRT': 0x0, 'VRSQRT': 0x1, 'VEXP': 0x2, 'VSILU': 0x3, 'VGELU': 0x4,
}

# Q8 functions
Q8SET_FUNC = {
    'QSETSCALE': 0x00, 'QSETQBASE': 0x01, 'QSETFBASE': 0x02,
    'FSETBASE': 0x02,  # Alias
}

Q8MAC_FUNC = {
    'ACCZERO': 0x00, 'ACCREAD': 0x01,
    'Q8MAC': 0x02, 'Q8MACINC': 0x03,
}

# Branch conditions
BR_COND = {
    'BEQ': 0x0, 'BNE': 0x1, 'BLT': 0x2, 'BGE': 0x3,
    'BLTU': 0x4, 'BGEU': 0x5, 'BGTZ': 0x6, 'BLEZ': 0x7,
    'BNEZ': 0x1,  # Alias: BNE rs, R0
    'BEQZ': 0x0,  # Alias: BEQ rs, R0
}


@dataclass
class Instruction:
    addr: int
    line: int
    text: str
    encoding: Optional[int] = None
    needs_fixup: bool = False
    fixup_label: str = ""


class Assembler:
    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.instructions: List[Instruction] = []
        self.addr = 0
        self.errors: List[str] = []

    def error(self, line: int, msg: str):
        self.errors.append(f"Line {line}: {msg}")

    def parse_reg(self, s: str) -> int:
        s = s.strip().upper()
        if s in REGS:
            return REGS[s]
        raise ValueError(f"Unknown register: {s}")

    def parse_freg(self, s: str) -> int:
        s = s.strip().upper()
        if s in FREGS:
            return FREGS[s]
        raise ValueError(f"Unknown FP register: {s}")

    def parse_vreg(self, s: str) -> int:
        s = s.strip().upper()
        if s in VREGS:
            return VREGS[s]
        raise ValueError(f"Unknown vector register: {s}")

    def parse_imm(self, s: str, bits: int = 16) -> int:
        s = s.strip()
        if s.startswith('0x') or s.startswith('0X'):
            val = int(s, 16)
        elif s.startswith('0b') or s.startswith('0B'):
            val = int(s, 2)
        else:
            val = int(s)
        max_val = (1 << bits) - 1
        min_val = -(1 << (bits - 1))
        if val < min_val or val > max_val:
            raise ValueError(f"Immediate {val} out of range for {bits} bits")
        return val & ((1 << bits) - 1)

    def parse_mem(self, s: str) -> Tuple[int, int]:
        """Parse memory operand like '16(R3)' -> (16, 3)"""
        m = re.match(r'(-?\d+)\s*\(\s*(\w+)\s*\)', s.strip())
        if not m:
            raise ValueError(f"Invalid memory operand: {s}")
        offset = int(m.group(1))
        reg = self.parse_reg(m.group(2))
        return offset, reg

    def encode_r(self, op: int, rd: int, rs1: int, rs2: int, func: int, ext: int = 0) -> int:
        return ((op & 0x3F) << 26) | ((rd & 0x1F) << 21) | ((rs1 & 0x1F) << 16) | \
               ((rs2 & 0x1F) << 11) | ((func & 0x1F) << 6) | (ext & 0x3F)

    def encode_i(self, op: int, rd: int, rs1: int, imm: int) -> int:
        return ((op & 0x3F) << 26) | ((rd & 0x1F) << 21) | ((rs1 & 0x1F) << 16) | \
               (imm & 0xFFFF)

    def encode_v(self, op: int, vd: int, vs1: int, vs2: int, func: int, ext: int = 0) -> int:
        return ((op & 0x3F) << 26) | ((vd & 0x7) << 23) | ((vs1 & 0x7) << 20) | \
               ((vs2 & 0x7) << 17) | ((func & 0x7) << 14) | (ext & 0x3FFF)

    def assemble_line(self, line: str, line_num: int) -> Optional[int]:
        """Assemble a single line, return encoding or None"""
        # Remove comments
        if '#' in line:
            line = line[:line.index('#')]
        if ';' in line:
            line = line[:line.index(';')]
        line = line.strip()

        if not line:
            return None

        # Check for label
        if ':' in line:
            label, rest = line.split(':', 1)
            self.labels[label.strip()] = self.addr
            line = rest.strip()
            if not line:
                return None

        # Skip assembler directives
        if line.startswith('.'):
            return None

        # Parse mnemonic and operands
        parts = line.split(None, 1)
        mnem = parts[0].upper()
        ops = parts[1].split(',') if len(parts) > 1 else []
        ops = [o.strip() for o in ops]

        try:
            return self.encode_instruction(mnem, ops, line_num)
        except Exception as e:
            self.error(line_num, f"{e}")
            return None

    def encode_instruction(self, mnem: str, ops: List[str], line_num: int) -> int:
        """Encode a single instruction"""

        # ===== Pseudo-instructions =====
        if mnem == 'NOP':
            return self.encode_r(OP['ALU'], 0, 0, 0, ALU_FUNC['ADD'], 0)

        if mnem == 'RET':
            return self.encode_i(OP['JALR'], 0, 1, 0)

        if mnem == 'MV':
            rd = self.parse_reg(ops[0])
            rs = self.parse_reg(ops[1])
            return self.encode_r(OP['ALU'], rd, rs, 0, ALU_FUNC['ADD'], 0)

        if mnem == 'LI':
            rd = self.parse_reg(ops[0])
            imm = self.parse_imm(ops[1])
            return self.encode_i(OP['ALUI'], rd, 0, imm)

        # ===== Load/Store =====
        if mnem == 'LW':
            rd = self.parse_reg(ops[0])
            off, rs1 = self.parse_mem(ops[1])
            return self.encode_i(OP['LW'], rd, rs1, off)

        if mnem == 'SW':
            rs2 = self.parse_reg(ops[0])
            off, rs1 = self.parse_mem(ops[1])
            return self.encode_i(OP['SW'], rs2, rs1, off)

        if mnem == 'LF':
            fd = self.parse_freg(ops[0])
            off, rs1 = self.parse_mem(ops[1])
            return self.encode_i(OP['LF'], fd, rs1, off)

        if mnem == 'SF':
            fs = self.parse_freg(ops[0])
            off, rs1 = self.parse_mem(ops[1])
            return self.encode_i(OP['SF'], fs, rs1, off)

        # ===== Integer ALU R-type =====
        if mnem in ALU_FUNC and mnem not in ('MV',):
            rd = self.parse_reg(ops[0])
            rs1 = self.parse_reg(ops[1])
            rs2 = self.parse_reg(ops[2])
            return self.encode_r(OP['ALU'], rd, rs1, rs2, ALU_FUNC[mnem], 0)

        # ===== Integer ALU I-type =====
        if mnem == 'ADDI':
            rd = self.parse_reg(ops[0])
            rs1 = self.parse_reg(ops[1])
            imm = self.parse_imm(ops[2])
            return self.encode_i(OP['ALUI'], rd, rs1, imm)

        if mnem in ('SLLI', 'SRLI', 'SRAI'):
            rd = self.parse_reg(ops[0])
            rs1 = self.parse_reg(ops[1])
            shamt = self.parse_imm(ops[2], 5)
            func = {'SLLI': 0x08, 'SRLI': 0x09, 'SRAI': 0x0A}[mnem]
            return self.encode_r(OP['ALU'], rd, rs1, 0, func, shamt)

        # ===== FP ALU =====
        if mnem in FPU_FUNC:
            if mnem in ('FMOV', 'FABS', 'FNEG'):
                fd = self.parse_freg(ops[0])
                fs1 = self.parse_freg(ops[1])
                return self.encode_r(OP['FPU'], fd, fs1, 0, FPU_FUNC[mnem], 0)
            elif mnem in ('FCVT.W.S', 'FCVT.S.W'):
                rd = self.parse_reg(ops[0]) if mnem == 'FCVT.W.S' else self.parse_freg(ops[0])
                rs = self.parse_freg(ops[1]) if mnem == 'FCVT.W.S' else self.parse_reg(ops[1])
                return self.encode_r(OP['FPU'], rd, rs, 0, FPU_FUNC[mnem], 0)
            else:
                fd = self.parse_freg(ops[0])
                fs1 = self.parse_freg(ops[1])
                fs2 = self.parse_freg(ops[2])
                return self.encode_r(OP['FPU'], fd, fs1, fs2, FPU_FUNC[mnem], 0)

        # ===== FP Special =====
        if mnem in FSPEC_FUNC:
            fd = self.parse_freg(ops[0])
            fs1 = self.parse_freg(ops[1])
            return self.encode_r(OP['FSPEC'], fd, fs1, 0, FSPEC_FUNC[mnem], 0)

        # ===== Vector Arithmetic =====
        if mnem in VARITH_FUNC:
            vd = self.parse_vreg(ops[0])
            vs1 = self.parse_vreg(ops[1])
            vs2 = self.parse_vreg(ops[2])
            return self.encode_v(OP['VARITH'], vd, vs1, vs2, VARITH_FUNC[mnem], 0)

        # ===== Vector-Scalar (R-type: vd in rd[2:0], vs1 in rs1[2:0], fs in rs2) =====
        if mnem in VSCALAR_FUNC:
            vd = self.parse_vreg(ops[0])
            vs1 = self.parse_vreg(ops[1])
            fs = self.parse_freg(ops[2])
            return self.encode_r(OP['VSCALAR'], vd, vs1, fs, VSCALAR_FUNC[mnem], 0)

        # ===== Vector Reductions (R-type: fd in rd, vs in rs1) =====
        if mnem in VRED_FUNC:
            fd = self.parse_freg(ops[0])
            vs1 = self.parse_vreg(ops[1])
            return self.encode_r(OP['VRED'], fd, vs1, 0, VRED_FUNC[mnem], 0)

        # ===== Vector Special =====
        if mnem in VSPEC_FUNC:
            vd = self.parse_vreg(ops[0])
            vs1 = self.parse_vreg(ops[1])
            return self.encode_v(OP['VSPEC'], vd, vs1, 0, VSPEC_FUNC[mnem], 0)

        # ===== Vector Load/Store (use I-type: vd in rd[2:0], base in rs1, stride in imm) =====
        if mnem == 'LVF':
            vd = self.parse_vreg(ops[0])
            rs1 = self.parse_reg(ops[1])
            stride = int(ops[2]) if len(ops) > 2 else 4
            return self.encode_i(OP['VLOAD'], vd, rs1, stride)

        if mnem == 'SVF':
            vs = self.parse_vreg(ops[0])
            rs1 = self.parse_reg(ops[1])
            stride = int(ops[2]) if len(ops) > 2 else 4
            return self.encode_i(OP['VSTORE'], vs, rs1, stride)

        # ===== Q8 Setup =====
        if mnem == 'QSETSCALE':
            fs = self.parse_freg(ops[0])
            return self.encode_r(OP['Q8SET'], 0, fs, 0, Q8SET_FUNC['QSETSCALE'], 0)

        if mnem in ('QSETBASE', 'QSETQBASE'):
            rs = self.parse_reg(ops[0])
            return self.encode_r(OP['Q8SET'], 0, rs, 0, Q8SET_FUNC['QSETQBASE'], 0)

        if mnem == 'FSETBASE':
            rs = self.parse_reg(ops[0])
            return self.encode_r(OP['Q8SET'], 0, rs, 0, Q8SET_FUNC['QSETFBASE'], 0)

        # ===== Q8 MAC =====
        if mnem == 'ACCZERO':
            return self.encode_r(OP['Q8MAC'], 0, 0, 0, Q8MAC_FUNC['ACCZERO'], 0)

        if mnem == 'ACCREAD':
            fd = self.parse_freg(ops[0])
            return self.encode_r(OP['Q8MAC'], fd, 0, 0, Q8MAC_FUNC['ACCREAD'], 0)

        if mnem == 'Q8MAC':
            n = self.parse_imm(ops[0], 6) if ops else 16
            return self.encode_r(OP['Q8MAC'], 0, 0, 0, Q8MAC_FUNC['Q8MAC'], n)

        if mnem == 'Q8MACINC':
            n = self.parse_imm(ops[0], 6) if ops else 16
            return self.encode_r(OP['Q8MAC'], 0, 0, 0, Q8MAC_FUNC['Q8MACINC'], n)

        # ===== Vector Set Length =====
        if mnem == 'VSETVL':
            rd = self.parse_reg(ops[0])
            rs = self.parse_reg(ops[1])
            return self.encode_r(OP['SYSTEM'], rd, rs, 0, 0x10, 0)

        # ===== Transformer Fused =====
        if mnem == 'ROPE':
            vd1 = self.parse_vreg(ops[0])
            vd2 = self.parse_vreg(ops[1])
            vc = self.parse_vreg(ops[2])
            vs = self.parse_vreg(ops[3])
            return self.encode_v(OP['ROPE'], vd1, vc, vs, vd2, 0)

        # ===== Branches =====
        if mnem in BR_COND:
            if mnem in ('BNEZ', 'BEQZ', 'BGTZ', 'BLEZ'):
                rs1 = self.parse_reg(ops[0])
                rs2 = 0
                target = ops[1]
            else:
                rs1 = self.parse_reg(ops[0])
                rs2 = self.parse_reg(ops[1])
                target = ops[2]

            cond = BR_COND[mnem]

            # Handle label vs immediate
            if target.lstrip('-').isdigit() or target.startswith('0x'):
                offset = int(target)
                # Encode: imm[15:13] = condition, imm[12:0] = signed offset
                imm = ((cond & 0x7) << 13) | (offset & 0x1FFF)
                return self.encode_i(OP['BRANCH'], rs1, rs2, imm)
            else:
                # Label - needs fixup, store condition for later
                imm = (cond & 0x7) << 13  # Offset will be filled in
                encoding = self.encode_i(OP['BRANCH'], rs1, rs2, imm)
                return ('FIXUP', encoding, target)

        # ===== JAL/JALR =====
        if mnem == 'JAL':
            rd = self.parse_reg(ops[0])
            target = ops[1]
            if target.lstrip('-').isdigit() or target.startswith('0x'):
                offset = self.parse_imm(target, 16)
            else:
                offset = 0  # Fixup needed
            return self.encode_i(OP['JAL'], rd, 0, offset)

        if mnem == 'JALR':
            rd = self.parse_reg(ops[0])
            rs1 = self.parse_reg(ops[1])
            off = self.parse_imm(ops[2]) if len(ops) > 2 else 0
            return self.encode_i(OP['JALR'], rd, rs1, off)

        # ===== LOOP =====
        if mnem == 'LOOP':
            rd = self.parse_reg(ops[0])
            target = ops[1]
            if target.lstrip('-').isdigit() or target.startswith('0x'):
                offset = self.parse_imm(target, 16)
                return self.encode_i(OP['LOOP'], rd, 0, offset)
            else:
                # Label - needs fixup (full 16-bit offset)
                encoding = self.encode_i(OP['LOOP'], rd, 0, 0)
                return ('FIXUP_LOOP', encoding, target)

        raise ValueError(f"Unknown instruction: {mnem}")

    def assemble(self, source: str) -> bytes:
        """Assemble source code to binary"""
        lines = source.split('\n')

        # First pass: collect labels and encode instructions
        for i, line in enumerate(lines, 1):
            result = self.assemble_line(line, i)
            if result is not None:
                if isinstance(result, tuple) and result[0] in ('FIXUP', 'FIXUP_LOOP'):
                    # Branch/loop with label that needs fixup
                    fixup_type, enc, label = result
                    inst = Instruction(self.addr, i, line.strip(), enc)
                    inst.needs_fixup = True
                    inst.fixup_label = label
                    inst.fixup_type = fixup_type
                else:
                    inst = Instruction(self.addr, i, line.strip(), result)
                self.instructions.append(inst)
                self.addr += 4

        # Second pass: resolve labels
        for inst in self.instructions:
            if inst.needs_fixup:
                if inst.fixup_label not in self.labels:
                    self.error(inst.line, f"Undefined label: {inst.fixup_label}")
                    continue
                target_addr = self.labels[inst.fixup_label]
                offset = (target_addr - inst.addr) >> 2  # Word offset, PC-relative
                if getattr(inst, 'fixup_type', 'FIXUP') == 'FIXUP_LOOP':
                    # LOOP: full 16-bit signed offset
                    inst.encoding = (inst.encoding & 0xFFFF0000) | (offset & 0xFFFF)
                else:
                    # BRANCH: condition bits in [15:13], offset in [12:0]
                    cond_bits = inst.encoding & 0xE000  # Preserve bits 15:13
                    inst.encoding = (inst.encoding & 0xFFFF0000) | cond_bits | (offset & 0x1FFF)

        if self.errors:
            for e in self.errors:
                print(f"Error: {e}", file=sys.stderr)
            return b''

        # Generate binary
        output = b''
        for inst in self.instructions:
            output += struct.pack('<I', inst.encoding)

        return output

    def disassemble(self, binary: bytes) -> str:
        """Disassemble binary to assembly"""
        lines = []
        for i in range(0, len(binary), 4):
            word = struct.unpack('<I', binary[i:i+4])[0]
            lines.append(f"0x{i:04x}: 0x{word:08x}  {self.decode_instruction(word)}")
        return '\n'.join(lines)

    def decode_instruction(self, word: int) -> str:
        """Decode a single instruction to assembly"""
        op = (word >> 26) & 0x3F
        rd = (word >> 21) & 0x1F
        rs1 = (word >> 16) & 0x1F
        rs2 = (word >> 11) & 0x1F
        func = (word >> 6) & 0x1F
        ext = word & 0x3F
        imm = word & 0xFFFF
        if imm >= 0x8000:
            imm -= 0x10000

        if op == OP['ALU']:
            if func == ALU_FUNC['ADD'] and rs2 == 0 and rd == 0 and rs1 == 0:
                return "NOP"
            if func == ALU_FUNC['ADD'] and rs2 == 0:
                return f"MV R{rd}, R{rs1}"
            shift_names = {0x08: 'SLLI', 0x09: 'SRLI', 0x0A: 'SRAI'}
            if func in shift_names:
                return f"{shift_names[func]} R{rd}, R{rs1}, {ext}"
            for name, f in ALU_FUNC.items():
                if f == func and name != 'MV':
                    return f"{name} R{rd}, R{rs1}, R{rs2}"
            return f"ALU? R{rd}, R{rs1}, R{rs2}, func={func}"

        if op == OP['ALUI']:
            return f"ADDI R{rd}, R{rs1}, {imm}"

        if op == OP['LW']:
            return f"LW R{rd}, {imm}(R{rs1})"

        if op == OP['SW']:
            return f"SW R{rd}, {imm}(R{rs1})"

        if op == OP['LF']:
            return f"LF F{rd}, {imm}(R{rs1})"

        if op == OP['SF']:
            return f"SF F{rd}, {imm}(R{rs1})"

        if op == OP['FPU']:
            for name, f in FPU_FUNC.items():
                if f == func:
                    if name in ('FMOV', 'FABS', 'FNEG'):
                        return f"{name} F{rd}, F{rs1}"
                    return f"{name} F{rd}, F{rs1}, F{rs2}"
            return f"FPU? F{rd}, F{rs1}, F{rs2}"

        if op == OP['FSPEC']:
            for name, f in FSPEC_FUNC.items():
                if f == func:
                    return f"{name} F{rd}, F{rs1}"
            return f"FSPEC? F{rd}, F{rs1}"

        if op == OP['Q8SET']:
            names = {0x00: 'QSETSCALE', 0x01: 'QSETBASE', 0x02: 'FSETBASE'}
            if func in names:
                return f"{names[func]} {'F' if func == 0 else 'R'}{rs1}"
            return f"Q8SET? func={func}"

        if op == OP['Q8MAC']:
            if func == Q8MAC_FUNC['ACCZERO']:
                return "ACCZERO"
            if func == Q8MAC_FUNC['ACCREAD']:
                return f"ACCREAD F{rd}"
            if func == Q8MAC_FUNC['Q8MAC']:
                return f"Q8MAC {ext}"
            if func == Q8MAC_FUNC['Q8MACINC']:
                return f"Q8MACINC {ext}"
            return f"Q8MAC? func={func}"

        if op == OP['BRANCH']:
            cond = (imm >> 13) & 0x7
            offset = imm & 0x1FFF
            if offset >= 0x1000:
                offset -= 0x2000
            cond_names = {0: 'BEQ', 1: 'BNE', 2: 'BLT', 3: 'BGE',
                          4: 'BLTU', 5: 'BGEU', 6: 'BGTZ', 7: 'BLEZ'}
            name = cond_names.get(cond, f'B?{cond}')
            if rs1 == 0 and name == 'BNE':
                return f"BNEZ R{rd}, {offset}"
            if rs1 == 0 and name == 'BEQ':
                return f"BEQZ R{rd}, {offset}"
            if rs1 == 0 and cond == 6:
                return f"BGTZ R{rd}, {offset}"
            return f"{name} R{rd}, R{rs1}, {offset}"

        if op == OP['JAL']:
            return f"JAL R{rd}, {imm}"

        if op == OP['JALR']:
            if rd == 0 and rs1 == 1 and imm == 0:
                return "RET"
            return f"JALR R{rd}, R{rs1}, {imm}"

        if op == OP['LOOP']:
            return f"LOOP R{rd}, {imm}"

        if op == OP['VLOAD']:
            vd = rd & 0x7
            stride = imm if imm >= 0 else imm + 0x10000
            return f"LVF V{vd}, R{rs1}, {stride}"

        if op == OP['VSTORE']:
            vs = rd & 0x7
            stride = imm if imm >= 0 else imm + 0x10000
            return f"SVF V{vs}, R{rs1}, {stride}"

        if op == OP['VARITH']:
            vd = (word >> 23) & 0x7
            vs1 = (word >> 20) & 0x7
            vs2 = (word >> 17) & 0x7
            vfunc = (word >> 14) & 0x7
            names = {0: 'VADD', 1: 'VSUB', 2: 'VMUL', 3: 'VDIV', 4: 'VFMADD'}
            return f"{names.get(vfunc, 'V?')} V{vd}, V{vs1}, V{vs2}"

        if op == OP['VSCALAR']:
            # R-type: vd in rd[2:0], vs1 in rs1[2:0], fs in rs2
            names = {0: 'VADDS', 1: 'VSUBS', 2: 'VMULS', 3: 'VDIVS'}
            return f"{names.get(func, 'VS?')} V{rd & 0x7}, V{rs1 & 0x7}, F{rs2}"

        if op == OP['VRED']:
            # R-type: fd in rd, vs in rs1, func in func field
            names = {0: 'VREDSUM', 1: 'VREDMAX', 2: 'VREDMIN', 3: 'VREDSQS'}
            return f"{names.get(func, 'VRED?')} F{rd}, V{rs1}"

        if op == OP['VSPEC']:
            vd = (word >> 23) & 0x7
            vs1 = (word >> 20) & 0x7
            vfunc = (word >> 14) & 0x7
            names = {0: 'VSQRT', 1: 'VRSQRT', 2: 'VEXP', 3: 'VSILU', 4: 'VGELU'}
            return f"{names.get(vfunc, 'VSPEC?')} V{vd}, V{vs1}"

        if op == OP['SYSTEM']:
            if func == 0x10:
                return f"VSETVL R{rd}, R{rs1}"
            return f"SYSTEM func={func}"

        return f"??? 0x{word:08x}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SMOL-32 Assembler')
    parser.add_argument('input', help='Input assembly file')
    parser.add_argument('-o', '--output', help='Output binary file')
    parser.add_argument('-d', '--disassemble', action='store_true',
                        help='Disassemble instead of assemble')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show assembled instructions')
    args = parser.parse_args()

    asm = Assembler()

    if args.disassemble:
        with open(args.input, 'rb') as f:
            binary = f.read()
        print(asm.disassemble(binary))
    else:
        with open(args.input, 'r') as f:
            source = f.read()

        binary = asm.assemble(source)

        if not binary:
            sys.exit(1)

        if args.verbose:
            print(asm.disassemble(binary))
            print()

        if args.output:
            with open(args.output, 'wb') as f:
                f.write(binary)
            print(f"Wrote {len(binary)} bytes to {args.output}")
        else:
            # Print hex dump
            for i in range(0, len(binary), 4):
                word = struct.unpack('<I', binary[i:i+4])[0]
                print(f"0x{word:08x}")


if __name__ == '__main__':
    main()
