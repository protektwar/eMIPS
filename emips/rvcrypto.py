"""RISC-V scalar cryptography (LF5; the unprivileged manual 20250508 ch. 31:
31.4's Sail for every instruction): Zkne / Zknd (AES), Zknh (SHA-256 /
SHA-512), Zksed (SM4), Zksh (SM3).  Zbkb / Zbkc / Zbkx are emips.simrv's
(LF1); the entropy source (Zkr's seed CSR) is emips.rvcsr's.

- **AES**: the S-box is FIPS-197's (the multiplicative inverse in
  GF(2^8) mod x^8+x^4+x^3+x+1, then the affine map); a 32-bit column holds
  its bytes from bit 0 (byte 0 = row 0); the RV64 instructions see the
  128-bit state as rs1 (columns 0-1) and rs2 (columns 2-3).
- **SM4**: the S-box is GB/T 32907-2016's (not in the RISC-V manual: copied
  from Spike's sm4_common.h, docs keeps it; a test compares).
"""

from __future__ import annotations

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF


def gfmul(a: int, b: int) -> int:
    """a x b in GF(2^8) mod x^8 + x^4 + x^3 + x + 1."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a = (a << 1) ^ (0x11B if a & 0x80 else 0)
        b >>= 1
    return out


def _sbox() -> tuple[list[int], list[int]]:
    fwd, inv = [0] * 256, [0] * 256
    for x in range(256):
        y = next((c for c in range(1, 256) if gfmul(x, c) == 1), 0)    # (0 -> 0)
        s = y
        for k in range(1, 5):
            s ^= ((y << k) | (y >> (8 - k))) & 0xFF
        s ^= 0x63
        fwd[x], inv[s] = s, x
    return fwd, inv


AES_SBOX, AES_SBOX_INV = _sbox()
#: aes_decode_rcon (round 0-9; 10 = 0: the last key schedule step)
RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x00)
#: GB/T 32907-2016's S-box (Spike's sm4_common.h)
SM4_SBOX = (
    0xD6, 0x90, 0xE9, 0xFE, 0xCC, 0xE1, 0x3D, 0xB7, 0x16, 0xB6, 0x14, 0xC2,
    0x28, 0xFB, 0x2C, 0x05, 0x2B, 0x67, 0x9A, 0x76, 0x2A, 0xBE, 0x04, 0xC3,
    0xAA, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99, 0x9C, 0x42, 0x50, 0xF4,
    0x91, 0xEF, 0x98, 0x7A, 0x33, 0x54, 0x0B, 0x43, 0xED, 0xCF, 0xAC, 0x62,
    0xE4, 0xB3, 0x1C, 0xA9, 0xC9, 0x08, 0xE8, 0x95, 0x80, 0xDF, 0x94, 0xFA,
    0x75, 0x8F, 0x3F, 0xA6, 0x47, 0x07, 0xA7, 0xFC, 0xF3, 0x73, 0x17, 0xBA,
    0x83, 0x59, 0x3C, 0x19, 0xE6, 0x85, 0x4F, 0xA8, 0x68, 0x6B, 0x81, 0xB2,
    0x71, 0x64, 0xDA, 0x8B, 0xF8, 0xEB, 0x0F, 0x4B, 0x70, 0x56, 0x9D, 0x35,
    0x1E, 0x24, 0x0E, 0x5E, 0x63, 0x58, 0xD1, 0xA2, 0x25, 0x22, 0x7C, 0x3B,
    0x01, 0x21, 0x78, 0x87, 0xD4, 0x00, 0x46, 0x57, 0x9F, 0xD3, 0x27, 0x52,
    0x4C, 0x36, 0x02, 0xE7, 0xA0, 0xC4, 0xC8, 0x9E, 0xEA, 0xBF, 0x8A, 0xD2,
    0x40, 0xC7, 0x38, 0xB5, 0xA3, 0xF7, 0xF2, 0xCE, 0xF9, 0x61, 0x15, 0xA1,
    0xE0, 0xAE, 0x5D, 0xA4, 0x9B, 0x34, 0x1A, 0x55, 0xAD, 0x93, 0x32, 0x30,
    0xF5, 0x8C, 0xB1, 0xE3, 0x1D, 0xF6, 0xE2, 0x2E, 0x82, 0x66, 0xCA, 0x60,
    0xC0, 0x29, 0x23, 0xAB, 0x0D, 0x53, 0x4E, 0x6F, 0xD5, 0xDB, 0x37, 0x45,
    0xDE, 0xFD, 0x8E, 0x2F, 0x03, 0xFF, 0x6A, 0x72, 0x6D, 0x6C, 0x5B, 0x51,
    0x8D, 0x1B, 0xAF, 0x92, 0xBB, 0xDD, 0xBC, 0x7F, 0x11, 0xD9, 0x5C, 0x41,
    0x1F, 0x10, 0x5A, 0xD8, 0x0A, 0xC1, 0x31, 0x88, 0xA5, 0xCD, 0x7B, 0xBD,
    0x2D, 0x74, 0xD0, 0x12, 0xB8, 0xE5, 0xB4, 0xB0, 0x89, 0x69, 0x97, 0x4A,
    0x0C, 0x96, 0x77, 0x7E, 0x65, 0xB9, 0xF1, 0x09, 0xC5, 0x6E, 0xC6, 0x84,
    0x18, 0xF0, 0x7D, 0xEC, 0x3A, 0xDC, 0x4D, 0x20, 0x79, 0xEE, 0x5F, 0x3E,
    0xD7, 0xCB, 0x39, 0x48)


def rol32(v: int, n: int) -> int:
    v &= M32
    n %= 32
    return ((v << n) | (v >> (32 - n))) & M32


def ror32(v: int, n: int) -> int:
    return rol32(v, 32 - n % 32)


def ror64(v: int, n: int) -> int:
    v &= M64
    return ((v >> n) | (v << (64 - n))) & M64


def _bytes32(x: int) -> list[int]:
    return [(x >> (8 * k)) & 0xFF for k in range(4)]


def _word(bs: list[int]) -> int:
    return sum(b << (8 * k) for k, b in enumerate(bs))


def mixcolumn_fwd(x: int) -> int:
    s0, s1, s2, s3 = _bytes32(x)
    return _word([gfmul(s0, 2) ^ gfmul(s1, 3) ^ s2 ^ s3,
                  s0 ^ gfmul(s1, 2) ^ gfmul(s2, 3) ^ s3,
                  s0 ^ s1 ^ gfmul(s2, 2) ^ gfmul(s3, 3),
                  gfmul(s0, 3) ^ s1 ^ s2 ^ gfmul(s3, 2)])


def mixcolumn_inv(x: int) -> int:
    s = _bytes32(x)
    m = (0xE, 0xB, 0xD, 0x9)
    return _word([gfmul(s[0], m[(0 - r) % 4]) ^ gfmul(s[1], m[(1 - r) % 4])
                  ^ gfmul(s[2], m[(2 - r) % 4]) ^ gfmul(s[3], m[(3 - r) % 4]) for r in range(4)])


def mixcolumn_byte_fwd(so: int) -> int:
    """aes_mixcolumn_byte_fwd: the column of so in row 0 (3s, s, s, 2s from
    the top)."""
    return gfmul(so, 3) << 24 | so << 16 | so << 8 | gfmul(so, 2)


def mixcolumn_byte_inv(so: int) -> int:
    return gfmul(so, 0xB) << 24 | gfmul(so, 0xD) << 16 | gfmul(so, 0x9) << 8 | gfmul(so, 0xE)


def subword(x: int, box) -> int:
    return _word([box[b] for b in _bytes32(x)])


def shiftrows64(rs2: int, rs1: int, inverse: bool) -> int:
    """aes_rv64_shiftrows_fwd / _inv: the low 64 bits (columns 0-1) of the
    state rs2:rs1 after ShiftRows (InvShiftRows)."""
    state = [(rs1 >> (8 * k)) & 0xFF for k in range(8)] + \
        [(rs2 >> (8 * k)) & 0xFF for k in range(8)]
    out = 0
    for c in range(2):
        for r in range(4):
            src = (c - r) % 4 if inverse else (c + r) % 4
            out |= state[4 * src + r] << (8 * (4 * c + r))
    return out


def _sx32(v: int) -> int:
    v &= M32
    return v - (1 << 32) if v >> 31 else v


def _executors() -> dict:
    from .simrv import ILLEGAL, RVTrap, _reg, _set
    ex: dict = {}

    def op(fn):
        def run(m, i, pc):
            f = i.f
            _set(m, f["rd"], fn(m, _reg(m, f["rs1"]), _reg(m, f.get("rs2", 0)), f))
        return run

    def aes32(box, mix):
        def fn(m, a, b, f):
            shamt = 8 * f["bs"]
            si = (b >> shamt) & 0xFF
            so = box[si]
            v = mix(so) if mix else so
            return _sx32(a ^ rol32(v, shamt))
        return fn
    ex["aes32esi"] = op(aes32(AES_SBOX, None))
    ex["aes32esmi"] = op(aes32(AES_SBOX, mixcolumn_byte_fwd))
    ex["aes32dsi"] = op(aes32(AES_SBOX_INV, None))
    ex["aes32dsmi"] = op(aes32(AES_SBOX_INV, mixcolumn_byte_inv))

    def aes64(inverse, mix):
        def fn(m, a, b, f):
            sr = shiftrows64(b, a, inverse)
            box = AES_SBOX_INV if inverse else AES_SBOX
            sb = subword(sr & M32, box) | subword(sr >> 32, box) << 32
            if mix:
                col = mixcolumn_inv if inverse else mixcolumn_fwd
                sb = col(sb & M32) | col(sb >> 32) << 32
            return sb
        return fn
    ex["aes64es"] = op(aes64(False, False))
    ex["aes64esm"] = op(aes64(False, True))
    ex["aes64ds"] = op(aes64(True, False))
    ex["aes64dsm"] = op(aes64(True, True))
    ex["aes64im"] = op(lambda m, a, b, f: mixcolumn_inv(a & M32) | mixcolumn_inv(a >> 32) << 32)

    def ks1i(m, i, pc):
        rnum = i.f["rnum"]
        if rnum > 10:                           # (31.4.10: 0xB-0xF reserved)
            raise RVTrap(ILLEGAL, i.raw)
        tmp1 = (_reg(m, i.f["rs1"]) >> 32) & M32
        tmp2 = tmp1 if rnum == 0xA else ror32(tmp1, 8)
        tmp3 = subword(tmp2, AES_SBOX) ^ RCON[rnum]
        _set(m, i.f["rd"], tmp3 | tmp3 << 32)
    ex["aes64ks1i"] = ks1i

    def ks2(m, a, b, f):
        w0 = (a >> 32) ^ (b & M32)
        w1 = w0 ^ (b >> 32)
        return (w1 & M32) << 32 | (w0 & M32)
    ex["aes64ks2"] = op(ks2)

    for name, rots, shift in (("sha256sig0", (7, 18), 3), ("sha256sig1", (17, 19), 10),
                              ("sha256sum0", (2, 13, 22), None),
                              ("sha256sum1", (6, 11, 25), None)):
        def sha256(m, a, b, f, rots=rots, shift=shift):
            x = a & M32
            v = 0
            for r in rots:
                v ^= ror32(x, r)
            if shift is not None:
                v ^= x >> shift
            return _sx32(v)
        ex[name] = op(sha256)
    for name, rots, shift in (("sha512sig0", (1, 8), 7), ("sha512sig1", (19, 61), 6),
                              ("sha512sum0", (28, 34, 39), None),
                              ("sha512sum1", (14, 18, 41), None)):
        def sha512(m, a, b, f, rots=rots, shift=shift):
            v = 0
            for r in rots:
                v ^= ror64(a, r)
            if shift is not None:
                v ^= a >> shift
            return v
        ex[name] = op(sha512)

    def r32(terms):                             # RV32's SHA-512 halves (31.4.31-31.4.36)
        def fn(m, a, b, f):
            v = 0
            for src, sh in terms:
                x = (a if src == 1 else b) & M32
                v ^= (x << sh if sh > 0 else x >> -sh) & M32
            return _sx32(v)
        return fn
    ex["sha512sig0h"] = op(r32(((1, -1), (1, -7), (1, -8), (2, 31), (2, 24))))
    ex["sha512sig0l"] = op(r32(((1, -1), (1, -7), (1, -8), (2, 31), (2, 25), (2, 24))))
    ex["sha512sig1h"] = op(r32(((1, 3), (1, -6), (1, -19), (2, -29), (2, 13))))
    ex["sha512sig1l"] = op(r32(((1, 3), (1, -6), (1, -19), (2, -29), (2, 26), (2, 13))))
    ex["sha512sum0r"] = op(r32(((1, 25), (1, 30), (1, -28), (2, -7), (2, -2), (2, 4))))
    ex["sha512sum1r"] = op(r32(((1, 23), (1, -14), (1, -18), (2, -9), (2, 18), (2, 14))))

    ex["sm3p0"] = op(lambda m, a, b, f: _sx32(a ^ rol32(a, 9) ^ rol32(a, 17)))
    ex["sm3p1"] = op(lambda m, a, b, f: _sx32(a ^ rol32(a, 15) ^ rol32(a, 23)))

    def sm4(ks):
        def fn(m, a, b, f):
            shamt = 8 * f["bs"]
            x = SM4_SBOX[(b >> shamt) & 0xFF]
            if ks:
                y = x ^ ((x & 0x07) << 29) ^ ((x & 0xFE) << 7) ^ ((x & 0x01) << 23) \
                    ^ ((x & 0xF8) << 13)
            else:
                y = x ^ (x << 8) ^ (x << 2) ^ (x << 18) ^ ((x & 0x3F) << 26) \
                    ^ ((x & 0xC0) << 10)
            return _sx32(rol32(y & M32, shamt) ^ a)
        return fn
    ex["sm4ed"] = op(sm4(False))
    ex["sm4ks"] = op(sm4(True))
    return ex
