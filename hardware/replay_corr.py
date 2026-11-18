"""Replay one DEFOM delta-update pixel against a recorded XSim output.

The integer operations follow the rtl0929 delta lookup in native_corr.py.
The small fixture contains the two correlation rows for output pixel (0, 41).
"""
from pathlib import Path


FIXTURES = Path(__file__).resolve().parent / "fixtures"
WIDTHS = (160, 80)
QUANTIZATION = ((18232, 28), (31389, 29))
PIXEL_X = 41
DISPARITY_Q8 = 235


def read_hex_words(name, count, hex_digits):
    words = (FIXTURES / name).read_text(encoding="ascii").splitlines()
    if len(words) != count or any(len(word) != hex_digits for word in words):
        raise ValueError(f"Unexpected word count or width in {name}")
    return [int(word, 16) for word in words]


def signed(value, bits):
    sign = 1 << (bits - 1)
    return (value + sign) % (2 * sign) - sign


def unpack_lanes(words, bits):
    """The least significant lane is the first element in the memory row."""
    return [signed(word >> offset, bits)
            for word in words for offset in range(0, 128, bits)]


def lookup(row, coordinate_q8, multiplier, shift):
    left, fraction = divmod(coordinate_q8, 256)
    a = row[left] if 0 <= left < len(row) else 0
    b = row[left + 1] if 0 <= left + 1 < len(row) else 0
    weighted = signed(a * (256 - fraction) + b * fraction, 32)
    sample = ((weighted >> 7) + 1) >> 1
    product = signed(sample * multiplier, 48)
    requantized = signed((product >> (shift - 1)) + 1, 32) >> 1
    return max(-128, min(127, requantized))


def replay():
    coordinate = signed(PIXEL_X * 256 - DISPARITY_Q8, 32)
    level1_coordinate = (signed(coordinate + 1, 32) >> 1
                         if coordinate >= 0 else coordinate >> 1)
    channels = []
    for level, center in enumerate((coordinate, level1_coordinate)):
        words = read_hex_words(f"row{level}.hex", WIDTHS[level] // 4, 32)
        row = unpack_lanes(words, 32)
        multiplier, shift = QUANTIZATION[level]
        for offset in range(-4, 5):
            channels.append(lookup(row, center + 256 * offset, multiplier, shift))
    return channels + [0] * (64 - len(channels))


def check():
    output = replay()
    expected = [signed(byte, 8) for byte in read_hex_words("expected.hex", 64, 2)]
    frozen = [signed(byte, 8) for byte in read_hex_words("frozen.hex", 64, 2)]
    captured = unpack_lanes(read_hex_words("xsim_output.hex", 4, 32), 8)
    for channel, (actual, reference, rtl) in enumerate(zip(output, expected, captured)):
        if actual != reference or actual != rtl:
            raise ValueError(f"Channel {channel}: computed={actual}, "
                             f"reference={reference}, XSim={rtl}")
    differences = [(index, actual, old)
                   for index, (actual, old) in enumerate(zip(output, frozen))
                   if actual != old]
    if differences != [(9, 21, 20)]:
        raise ValueError(f"Unexpected frozen-model differences: {differences}")
    print("PASS: 18 INT8 correlation channels and 46 padding bytes match "
          "the reference and the recorded four XSim beats.")
    print("Channel 9: RTL arithmetic = 21, frozen model = 20.")


if __name__ == "__main__":
    check()
