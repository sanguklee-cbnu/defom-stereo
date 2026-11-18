import sys, os, time, json
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NPUv4.npu_utils import *
from NPUv4.data_converter import *
from PCIeAccess import *


AXIL_BASE = 0x0000_0000
DDR_BASE  = 0x8000_0000

REG_CONTROL  = 0x000
REG_BASE_W   = 0x010
REG_BASE_IN  = 0x014
REG_BASE_Q   = 0x018
REG_BASE_B   = 0x01C
REG_BASE_OUT = 0x020
REG_CYC_LO   = 0x024
REG_CYC_HI   = 0x028
REG_CYC_STAT = 0x02C
REG_SIZE_CFG = 0x044
REG_VERSION  = 0x0FC

IN_OFF  = 0x0000_0000
W_OFF   = 0x0100_0000
Q_OFF   = 0x0400_0000
B_OFF   = 0x0410_0000
OUT_OFF = 0x0500_0000

# One-block board test: patch output slot + encoder block0 output slot.
SIZE_CONFIG_JSON = os.environ.get("DINO_SIZE_CONFIG_JSON", "")
if SIZE_CONFIG_JSON and os.path.exists(SIZE_CONFIG_JSON):
    with open(SIZE_CONFIG_JSON, "r") as f:
        size_cfg = json.load(f)
else:
    size_cfg = {
        "active_token_total": 1601,
        "active_mblk_num": 51,
        "active_last_valid_lane": 0,
        "layout_token_total": 1601,
        "layout_mblk_num": 51,
        "size_cfg_word_hex": "0x00033641",
    }

TOKEN_TOTAL = int(size_cfg["active_token_total"])
ACTIVE_M_BLK_NUM = int(size_cfg["active_mblk_num"])
ACTIVE_LAST_VALID_LANE = int(size_cfg["active_last_valid_lane"])
LAYOUT_TOKEN_TOTAL = int(size_cfg.get("layout_token_total", 1601))
M_BLK_NUM = int(size_cfg.get("layout_mblk_num", 51))
SIZE_CFG_WORD = int(str(size_cfg["size_cfg_word_hex"]), 16)
EMBED_DIM   = 384
STRIDE_384  = 384
ENC_BLK_NUM = 1
FINAL_BLK_IDX = 0
STRIDE_PATCH_K = 589
EMBED_NBLK = 12

SIZE_OUT_PATCH_OUT_WORD256 = M_BLK_NUM * STRIDE_384
SIZE_OUT_ACTIVE_WORD256 = ACTIVE_M_BLK_NUM * STRIDE_384
FINAL_OUT_OFF_WORD256 = (1 + FINAL_BLK_IDX) * SIZE_OUT_PATCH_OUT_WORD256
FINAL_OUT_OFF_BYTES   = FINAL_OUT_OFF_WORD256 * 32
FINAL_OUT_BYTES       = SIZE_OUT_ACTIVE_WORD256 * 32

OUT_SLOT_BYTES = SIZE_OUT_PATCH_OUT_WORD256 * 32
OUT_SLOT_COUNT = 1 + ENC_BLK_NUM
OUT_TOTAL_BYTES = OUT_SLOT_BYTES * OUT_SLOT_COUNT
DEAD_WORD = 0xDEADBEEF

# Encoder block0 Q/V scratch locations in the W region.
SIZE_W_PATCH_WGT = EMBED_NBLK * STRIDE_PATCH_K
SIZE_W_Q_WGT = EMBED_NBLK * STRIDE_384
SIZE_W_V_WGT = EMBED_NBLK * STRIDE_384
SIZE_W_K_WGT = EMBED_NBLK * STRIDE_384
SIZE_W_Q_OUT = M_BLK_NUM * STRIDE_384
SIZE_W_V_OUT = EMBED_NBLK * LAYOUT_TOKEN_TOTAL

OFF_W_ENC0 = SIZE_W_PATCH_WGT
OFF_W_Q_WGT = OFF_W_ENC0
OFF_W_V_WGT = OFF_W_Q_WGT + SIZE_W_Q_WGT
OFF_W_K_WGT = OFF_W_V_WGT + SIZE_W_V_WGT
OFF_W_Q_OUT = OFF_W_K_WGT + SIZE_W_K_WGT
OFF_W_V_OUT = OFF_W_Q_OUT + SIZE_W_Q_OUT

Q_OUT_OFF_BYTES = OFF_W_Q_OUT * 32
V_OUT_OFF_BYTES = OFF_W_V_OUT * 32
Q_OUT_BYTES = (ACTIVE_M_BLK_NUM * STRIDE_384) * 32
V_OUT_BYTES = (EMBED_NBLK * TOKEN_TOTAL) * 32

VERIFY_ROOT = os.environ.get(
    "DINO_VERIFY_ROOT",
    ".",
)

VEC_DIR = os.environ.get("DINO_VEC_DIR", os.path.join(VERIFY_ROOT, "vector_nonlinear_560"))
GOLD_DIR = os.environ.get("DINO_GOLD_DIR", os.path.join(VERIFY_ROOT, "golden"))

INPUT_TXT = os.path.join(VEC_DIR, "input256.txt")
WEIGHT_TXT = os.path.join(VEC_DIR, "weight256.txt")
QUANT_TXT = os.path.join(VEC_DIR, "quant256.txt")
PARAM_TXT = os.path.join(VEC_DIR, "parameter256.txt")

GOLD_FINAL_TXT = os.path.join(GOLD_DIR, "blk0", "output256_fc2_post_residual_ln.txt")
GOLD_PATCH_TXT = os.path.join(GOLD_DIR, "output256_final.txt")
GOLD_Q_TXT = os.path.join(GOLD_DIR, "blk0", "output256_q_final.txt")
GOLD_V_TXT = os.path.join(GOLD_DIR, "blk0", "output256_vT_final.txt")


def load_hex32B_to_u32(path, reverse_bytes=True):
    raw = bytearray()
    with open(path, "r") as f:
        for line in f:
            s = line.strip().replace(" ", "")
            if not s:
                continue
            row = bytes.fromhex(s)
            if reverse_bytes:
                row = row[::-1]
            raw.extend(row)

    u8 = np.frombuffer(raw, dtype=np.uint8)
    assert (u8.size % 4) == 0
    return u8.view(np.uint32).copy()


def save_u32_to_hex32B(path, u32, reverse_bytes=True):
    u32 = np.asarray(u32, dtype=np.uint32)
    raw = u32.view(np.uint8)

    with open(path, "w") as f:
        for i in range(0, raw.size, 32):
            row = raw[i:i + 32].tobytes()
            if reverse_bytes:
                row = row[::-1]
            f.write(row.hex() + "\n")


def u8_to_u32_words(u8):
    u8 = np.asarray(u8, dtype=np.uint8)
    assert (u8.size % 4) == 0
    return u8.view(np.uint32).copy()


def reg_write(off, val):
    PCIeRegWrite(AXIL_BASE + off, val)


def reg_read(off):
    return PCIeRegRead(AXIL_BASE + off)


def abs_ddr_addr(host_off):
    return DDR_BASE + host_off


def clear_all_output_slots():
    clear_words = OUT_TOTAL_BYTES // 4
    clear_data = np.full(clear_words, DEAD_WORD, dtype=np.uint32)

    print(
        f"[DDR] clear OUT slots offset=0x{OUT_OFF:08X}, "
        f"bytes=0x{OUT_TOTAL_BYTES:X}, slots={OUT_SLOT_COUNT}"
    )
    Write_PCIe_XDMA(OUT_OFF, clear_data)


def compare_region(name, addr, byte_len, gold_path=None, save_name=None):
    print(f"[DDR] read {name} offset=0x{addr:08X}, bytes=0x{byte_len:X}")

    out0 = u8_to_u32_words(Read_PCIe_XDMA(addr, byte_len))
    time.sleep(0.2)
    out1 = u8_to_u32_words(Read_PCIe_XDMA(addr, byte_len))
    stable = np.array_equal(out0, out1)
    print(f"[CHECK] {name} stable={stable}")

    if save_name is not None:
        save_path = os.path.join(VERIFY_ROOT, save_name)
        save_u32_to_hex32B(save_path, out1, reverse_bytes=True)
        print(f"[SAVE] {save_path}")

    dead_count = int(np.count_nonzero(out1 == DEAD_WORD))
    changed_count = int(out1.size - dead_count)
    print(f"[CHECK] {name} changed_words={changed_count}/{out1.size}")

    if gold_path is None:
        return changed_count > 0

    gold = load_hex32B_to_u32(gold_path)
    if gold.size > out1.size:
        gold = gold[:out1.size]
    print(f"[CHECK] {name} fpga_shape={out1.shape}, gold_shape={gold.shape}")

    if out1.shape != gold.shape:
        print(f"[FAIL] {name} shape mismatch")
        return False

    diff = np.nonzero(out1 != gold)[0]
    if diff.size == 0:
        print(f"[PASS] {name} matches golden")
        return True

    print(f"[FAIL] {name} mismatch_count={diff.size}/{gold.size}")
    for idx in diff[:16]:
        idx = int(idx)
        print(
            f"  idx={idx} "
            f"fpga=0x{int(out1[idx]):08X} "
            f"gold=0x{int(gold[idx]):08X}"
        )
    return False


def scan_output_slots():
    print("\n========== OUT SLOT SCAN ==========")
    for slot in range(OUT_SLOT_COUNT):
        slot_off = OUT_OFF + slot * OUT_SLOT_BYTES
        slot_name = "patch" if slot == 0 else f"blk{slot - 1}"
        compare_region(slot_name, slot_off, OUT_SLOT_BYTES)
    print("===================================\n")


image  = load_hex32B_to_u32(INPUT_TXT)
weight = load_hex32B_to_u32(WEIGHT_TXT)
quant  = load_hex32B_to_u32(QUANT_TXT)
param  = load_hex32B_to_u32(PARAM_TXT)

print("image u32 :", len(image),  "bytes:", len(image)  * 4)
print("weight u32:", len(weight), "bytes:", len(weight) * 4)
print("quant u32 :", len(quant),  "bytes:", len(quant)  * 4)
print("param u32 :", len(param),  "bytes:", len(param)  * 4)
print(
    "[SIZE] "
    f"active_tokens={TOKEN_TOTAL}, active_mblks={ACTIVE_M_BLK_NUM}, "
    f"last_lane={ACTIVE_LAST_VALID_LANE}, layout_tokens={LAYOUT_TOKEN_TOTAL}, "
    f"layout_mblks={M_BLK_NUM}, size_cfg=0x{SIZE_CFG_WORD:08X}"
)

ver = reg_read(REG_VERSION)
print(f"[VIT] VERSION=0x{ver:08X}")

if ver != 0x00010000:
    print("WARNING: VERSION mismatch. Check AXIL_BASE or bitstream.")

print("[DDR] write INPUT")
Write_PCIe_XDMA(IN_OFF, image)

print("[DDR] write WEIGHT")
Write_PCIe_XDMA(W_OFF, weight)

print("[DDR] write QUANT")
Write_PCIe_XDMA(Q_OFF, quant)

print("[DDR] write PARAM")
Write_PCIe_XDMA(B_OFF, param)

clear_all_output_slots()

reg_write(REG_CONTROL, 0x10)

reg_write(REG_BASE_W,   abs_ddr_addr(W_OFF))
reg_write(REG_BASE_IN,  abs_ddr_addr(IN_OFF))
reg_write(REG_BASE_Q,   abs_ddr_addr(Q_OFF))
reg_write(REG_BASE_B,   abs_ddr_addr(B_OFF))
reg_write(REG_BASE_OUT, abs_ddr_addr(OUT_OFF))
reg_write(REG_SIZE_CFG, SIZE_CFG_WORD)

print(f"BASE_W   = 0x{reg_read(REG_BASE_W):08X}")
print(f"BASE_IN  = 0x{reg_read(REG_BASE_IN):08X}")
print(f"BASE_Q   = 0x{reg_read(REG_BASE_Q):08X}")
print(f"BASE_B   = 0x{reg_read(REG_BASE_B):08X}")
print(f"BASE_OUT = 0x{reg_read(REG_BASE_OUT):08X}")
print(f"SIZE_CFG = 0x{reg_read(REG_SIZE_CFG):08X}")

reg_write(REG_CONTROL, 0x1)

t0 = time.time()
timeout_s = 60.0
last_print = time.time()
timed_out = False

while True:
    ctrl = reg_read(REG_CONTROL)

    start_latched = (ctrl >> 0) & 1
    done          = (ctrl >> 1) & 1
    idle          = (ctrl >> 2) & 1
    busy          = (ctrl >> 3) & 1

    cyc_lo = reg_read(REG_CYC_LO)
    cyc_hi = reg_read(REG_CYC_HI)
    cycles = (cyc_hi << 32) | cyc_lo

    now = time.time()

    if now - last_print >= 0.5:
        print(
            f"[POLL] t={now - t0:.1f}s ctrl=0x{ctrl:08X} "
            f"start={start_latched} done={done} idle={idle} busy={busy} "
            f"cycles={cycles}"
        )
        last_print = now

    if done:
        print(f"DONE! ctrl=0x{ctrl:08X}, elapsed={now - t0:.6f}s, cycles={cycles}")
        break

    if now - t0 > timeout_s:
        print(
            f"[TIMEOUT] ctrl=0x{ctrl:08X}, cycles={cycles}, "
            f"start={start_latched}, done={done}, idle={idle}, busy={busy}"
        )
        timed_out = True
        break

scan_output_slots()

compare_region(
    "patch",
    OUT_OFF,
    OUT_SLOT_BYTES,
    gold_path=GOLD_PATCH_TXT,
    save_name="fpga_out_patch.txt",
)

compare_region(
    "blk0 Q_OUT scratch",
    W_OFF + Q_OUT_OFF_BYTES,
    Q_OUT_BYTES,
    gold_path=GOLD_Q_TXT,
    save_name="fpga_out_blk0_q_out.txt",
)

compare_region(
    "blk0 V_OUT scratch",
    W_OFF + V_OUT_OFF_BYTES,
    V_OUT_BYTES,
    gold_path=GOLD_V_TXT,
    save_name="fpga_out_blk0_vT_out.txt",
)

result_addr = OUT_OFF + FINAL_OUT_OFF_BYTES
ok = compare_region(
    "blk0 final",
    result_addr,
    FINAL_OUT_BYTES,
    gold_path=GOLD_FINAL_TXT,
    save_name="fpga_out_blk0_fc2_post_residual_ln.txt",
)

if timed_out:
    if ok:
        print("[DIAG] blk0 output matches golden even though done did not assert.")
        raise SystemExit(0)
    print("[DIAG] blk0 output does not match golden after timeout.")
    raise SystemExit(2)

if not ok:
    raise SystemExit(1)

print("PASS")
