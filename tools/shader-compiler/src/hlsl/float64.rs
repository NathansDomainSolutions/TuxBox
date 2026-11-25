//! Exactly specified double-precision forms on register pairs: rounding to
//! an integral value, conversions between f64 and 32-bit integers or f32,
//! and the IEEE-754 binary64 add, multiply, fused multiply-add and ldexp.
//! Like the f64 comparisons they use bit operations on `uint2` (low, high)
//! pairs, so no host FP64 support is required. Each result is exact; the
//! only mode dependence is whether f64 denormals are flushed (MODE bit 6
//! for inputs; bit 7 for arithmetic outputs).
//!
//! The arithmetic helpers are soft float on 32-bit words, rounding to
//! nearest even. Speculative, not checked on hardware: MODE.FP_ROUND is
//! ignored; output flushing tests the rounded result and keeps its sign; a
//! NaN operand is returned quieted (the first NaN operand, in source order,
//! payload kept), and an invalid operation returns +qNaN 0x7ff8000000000000.
use super::{r, Emitter, Enc, Insn, Opnd};

pub(super) const HELPERS: &str = r#"
uint2 tb_f64_round(uint2 x, uint mode, bool keepDenormals) {
    // mode: 0 toward zero, 1 floor, 2 ceil, 3 nearest even.
    uint e = (x.y >> 20) & 0x7ffu, sign = x.y & 0x80000000u;
    if (e == 0x7ffu) return ((x.y & 0xfffffu) | x.x) != 0u ? uint2(x.x, x.y | 0x80000u) : x;
    if (e >= 1075u) return x;
    if (e == 0u && !keepDenormals) x = uint2(0u, sign);
    bool nonzero = ((x.y & 0x7fffffffu) | x.x) != 0u;
    if (e < 1023u) {
        bool one = mode == 1u ? (sign != 0u && nonzero) : mode == 2u ? (sign == 0u && nonzero)
            : mode == 3u && e == 1022u && ((x.y & 0xfffffu) | x.x) != 0u;
        return uint2(0u, sign | (one ? 0x3ff00000u : 0u));
    }
    uint f = 1075u - e;
    uint2 mask = f >= 32u ? uint2(0xffffffffu, (1u << (f - 32u)) - 1u) : uint2((1u << f) - 1u, 0u);
    uint2 unit = f >= 32u ? uint2(0u, 1u << (f - 32u)) : uint2(1u << f, 0u);
    uint2 frac = x & mask, t = x & ~mask;
    bool hasFrac = (frac.x | frac.y) != 0u, up = false;
    if (mode == 1u) up = sign != 0u && hasFrac;
    if (mode == 2u) up = sign == 0u && hasFrac;
    if (mode == 3u) {
        uint2 half = f > 32u ? uint2(0u, 1u << (f - 33u)) : uint2(1u << (f - 1u), 0u);
        bool above = frac.y > half.y || (frac.y == half.y && frac.x > half.x);
        bool tie = frac.x == half.x && frac.y == half.y;
        up = above || (tie && ((t.x & unit.x) | (t.y & unit.y)) != 0u);
    }
    if (up) {
        uint lo = t.x + unit.x;
        t = uint2(lo, t.y + unit.y + (lo < t.x ? 1u : 0u));
    }
    return t;
}
uint tb_f64_to_i32(uint2 x, bool signedResult) {
    // Truncate; saturate out-of-range values and return 0 for NaN.
    uint e = (x.y >> 20) & 0x7ffu;
    bool neg = (x.y >> 31) != 0u;
    if (e == 0x7ffu && ((x.y & 0xfffffu) | x.x) != 0u) return 0u;
    if (e < 1023u) return 0u;
    if (signedResult && e >= 1054u) return neg ? 0x80000000u : 0x7fffffffu;
    if (!signedResult && neg) return 0u;
    if (!signedResult && e >= 1055u) return 0xffffffffu;
    uint s = 52u - (e - 1023u);
    uint hi = (x.y & 0xfffffu) | 0x100000u;
    uint v = s >= 32u ? hi >> (s - 32u) : (hi << (32u - s)) | (x.x >> s);
    return signedResult && neg ? 0u - v : v;
}
uint2 tb_i32_to_f64(uint v, bool signedInput) {
    uint sign = 0u;
    if (signedInput && (v >> 31) != 0u) { sign = 0x80000000u; v = 0u - v; }
    if (v == 0u) return uint2(0u, 0u);
    uint p = firstbithigh(v), rest = v ^ (1u << p), sh = 52u - p;
    uint2 m = sh >= 32u ? uint2(0u, rest << (sh - 32u)) : uint2(rest << sh, rest >> (32u - sh));
    return uint2(m.x, sign | ((1023u + p) << 20) | m.y);
}
uint2 tb_f32_to_f64(uint b, bool keepDenormals) {
    // Exact widening; NaN is quieted and keeps its payload's top bits.
    uint sign = b & 0x80000000u, e = (b >> 23) & 0xffu, m = b & 0x7fffffu, E;
    if (e == 0xffu) return uint2(m << 29, sign | 0x7ff00000u | (m != 0u ? 0x80000u : 0u) | (m >> 3));
    if (e == 0u) {
        if (m == 0u || !keepDenormals) return uint2(0u, sign);
        uint p = firstbithigh(m);
        m = (m << (23u - p)) & 0x7fffffu;
        E = p + 874u;
    } else {
        E = e + 896u;
    }
    return uint2(m << 29, sign | (E << 20) | (m >> 3));
}
uint tb_f64_to_f32(uint2 x, bool keepDenormals) {
    // Round to nearest even. An f64 denormal is below half the smallest f32
    // denormal, so it gives zero whatever the input denormal mode.
    uint sign = x.y & 0x80000000u, e = (x.y >> 20) & 0x7ffu, mh = x.y & 0xfffffu;
    if (e == 0x7ffu) return ((mh | x.x) != 0u) ? (sign | 0x7fc00000u | (mh << 3) | (x.x >> 29)) : (sign | 0x7f800000u);
    if (e == 0u) return sign;
    uint m24 = ((mh | 0x100000u) << 3) | (x.x >> 29), rem = x.x & 0x1fffffffu, bits;
    int eb = (int)e - 896;
    bool up;
    if (eb >= 255) return sign | 0x7f800000u;
    if (eb > 0) {
        bits = ((uint)eb << 23) | (m24 & 0x7fffffu);
        up = rem > 0x10000000u || (rem == 0x10000000u && (bits & 1u) != 0u);
    } else {
        uint k = (uint)(1 - eb);
        if (k > 24u) return sign;
        bits = k < 24u ? m24 >> k : 0u;
        uint dropped = m24 & ((1u << k) - 1u), half = 1u << (k - 1u);
        up = dropped > half || (dropped == half && (rem != 0u || (bits & 1u) != 0u));
    }
    bits += up ? 1u : 0u;
    if (!keepDenormals && (bits & 0x7f800000u) == 0u) bits = 0u;
    return sign | bits;
}
"#;

/// Soft-float arithmetic on register pairs (VOP3 only).
pub(super) const ARITH_HELPERS: &str = r#"
uint2 tb_u64_add(uint2 a, uint2 b) {
    uint lo = a.x + b.x;
    return uint2(lo, a.y + b.y + (lo < a.x ? 1u : 0u));
}
uint2 tb_u64_sub(uint2 a, uint2 b) {
    return uint2(a.x - b.x, a.y - b.y - (a.x < b.x ? 1u : 0u));
}
bool tb_u64_less(uint2 a, uint2 b) {
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}
uint2 tb_u64_shl(uint2 a, uint n) {
    // 0 <= n < 64.
    if (n == 0u) return a;
    if (n >= 32u) return uint2(0u, a.x << (n - 32u));
    return uint2(a.x << n, (a.y << n) | (a.x >> (32u - n)));
}
uint2 tb_u64_shr_jam(uint2 a, uint n) {
    // Shift right; any nonzero bit shifted out sets bit 0.
    if (n == 0u) return a;
    if (n >= 64u) return uint2((a.x | a.y) != 0u ? 1u : 0u, 0u);
    uint2 s;
    bool lost;
    if (n >= 32u) {
        uint k = n - 32u;
        s = uint2(a.y >> k, 0u);
        lost = a.x != 0u || (k != 0u && (a.y << (32u - k)) != 0u);
    } else {
        s = uint2((a.x >> n) | (a.y << (32u - n)), a.y >> n);
        lost = (a.x << (32u - n)) != 0u;
    }
    return uint2(s.x | (lost ? 1u : 0u), s.y);
}
uint tb_u64_clz(uint2 a) {
    return a.y != 0u ? 31u - firstbithigh(a.y) : a.x != 0u ? 63u - firstbithigh(a.x) : 64u;
}
uint2 tb_u32_mul_wide(uint a, uint b) {
    uint a0 = a & 0xffffu, a1 = a >> 16, b0 = b & 0xffffu, b1 = b >> 16;
    uint p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint mid = (p00 >> 16) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return uint2((mid << 16) | (p00 & 0xffffu), p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16));
}
uint4 tb_u64_mul(uint2 a, uint2 b) {
    // 128-bit product; .x is the least significant word.
    uint2 p00 = tb_u32_mul_wide(a.x, b.x), p01 = tb_u32_mul_wide(a.x, b.y);
    uint2 p10 = tb_u32_mul_wide(a.y, b.x), p11 = tb_u32_mul_wide(a.y, b.y);
    uint2 c1 = tb_u64_add(tb_u64_add(uint2(p00.y, 0u), uint2(p01.x, 0u)), uint2(p10.x, 0u));
    uint2 c2 = tb_u64_add(tb_u64_add(uint2(p01.y, 0u), uint2(p10.y, 0u)), tb_u64_add(uint2(p11.x, 0u), uint2(c1.y, 0u)));
    return uint4(p00.x, c1.x, c2.x, p11.y + c2.y);
}
uint4 tb_u128_add(uint4 a, uint4 b) {
    uint2 lo = tb_u64_add(a.xy, b.xy);
    return uint4(lo, tb_u64_add(tb_u64_add(a.zw, b.zw), uint2(tb_u64_less(lo, a.xy) ? 1u : 0u, 0u)));
}
uint4 tb_u128_sub(uint4 a, uint4 b) {
    uint2 lo = tb_u64_sub(a.xy, b.xy);
    return uint4(lo, tb_u64_sub(tb_u64_sub(a.zw, b.zw), uint2(tb_u64_less(a.xy, b.xy) ? 1u : 0u, 0u)));
}
bool tb_u128_less(uint4 a, uint4 b) {
    return tb_u64_less(a.zw, b.zw) || (a.z == b.z && a.w == b.w && tb_u64_less(a.xy, b.xy));
}
uint4 tb_u128_shl(uint4 a, uint n) {
    // 0 <= n < 128.
    if (n >= 64u) { a = uint4(0u, 0u, a.xy); n -= 64u; }
    if (n >= 32u) { a = uint4(0u, a.xyz); n -= 32u; }
    if (n == 0u) return a;
    return uint4(a.x << n, (a.y << n) | (a.x >> (32u - n)), (a.z << n) | (a.y >> (32u - n)), (a.w << n) | (a.z >> (32u - n)));
}
uint4 tb_u128_shr_jam(uint4 a, uint n) {
    if (n >= 128u) return uint4((a.x | a.y | a.z | a.w) != 0u ? 1u : 0u, 0u, 0u, 0u);
    bool lost = false;
    if (n >= 64u) { lost = (a.x | a.y) != 0u; a = uint4(a.zw, 0u, 0u); n -= 64u; }
    if (n >= 32u) { lost = lost || a.x != 0u; a = uint4(a.yzw, 0u); n -= 32u; }
    if (n != 0u) {
        lost = lost || (a.x << (32u - n)) != 0u;
        a = uint4((a.x >> n) | (a.y << (32u - n)), (a.y >> n) | (a.z << (32u - n)), (a.z >> n) | (a.w << (32u - n)), a.w >> n);
    }
    return uint4(a.x | (lost ? 1u : 0u), a.yzw);
}
uint tb_u128_clz(uint4 a) {
    return a.w != 0u ? 31u - firstbithigh(a.w) : a.z != 0u ? 63u - firstbithigh(a.z)
        : a.y != 0u ? 95u - firstbithigh(a.y) : a.x != 0u ? 127u - firstbithigh(a.x) : 128u;
}
bool tb_f64_nan(uint2 x) {
    uint m = x.y & 0x7fffffffu;
    return m > 0x7ff00000u || (m == 0x7ff00000u && x.x != 0u);
}
uint2 tb_f64_sig(uint2 x, bool keepDenormals) {
    // Significand with its implicit bit; flushed subnormals give zero.
    uint e = (x.y >> 20) & 0x7ffu;
    if (e == 0u) return keepDenormals ? uint2(x.x, x.y & 0xfffffu) : uint2(0u, 0u);
    return uint2(x.x, (x.y & 0xfffffu) | 0x100000u);
}
int tb_f64_exp(uint2 x) {
    // Biased exponent; subnormals scale like exponent 1.
    return max(int((x.y >> 20) & 0x7ffu), 1);
}
uint2 tb_f64_normalize(uint2 sig, inout int e) {
    // Move the leading one of a nonzero significand to bit 52.
    uint shift = tb_u64_clz(sig) - 11u;
    e -= int(shift);
    return tb_u64_shl(sig, shift);
}
uint2 tb_f64_pack(uint sign, int e, uint2 sig, bool keepDenormals) {
    // sig has its leading one at bit 62 (or is a subnormal with e == 0) and
    // its rounding bits below bit 10; e is one less than the biased exponent
    // of bit 62. Rounds to nearest even: MODE.FP_ROUND is ignored.
    // A rounding carry at e == 0x7fd produces the infinity encoding itself.
    if (e > 0x7fd) return uint2(0u, sign | 0x7ff00000u);
    if (e < 0) {
        sig = tb_u64_shr_jam(sig, uint(-e));
        e = 0;
    }
    uint roundBits = sig.x & 0x3ffu;
    sig = tb_u64_add(sig, uint2(0x200u, 0u));
    sig = uint2((sig.x >> 10) | (sig.y << 22), sig.y >> 10);
    if (roundBits == 0x200u) sig.x &= ~1u;
    if ((sig.x | sig.y) == 0u) e = 0;
    uint hi = sign | ((uint(e) << 20) + sig.y);
    // Speculative: output flushing tests the rounded result.
    if (!keepDenormals && (hi & 0x7ff00000u) == 0u) return uint2(0u, sign);
    return uint2(sig.x, hi);
}
uint2 tb_f64_add(uint2 a, uint2 b, bool keepIn, bool keepOut) {
    if (tb_f64_nan(a) || tb_f64_nan(b)) return tb_f64_nan(a) ? uint2(a.x, a.y | 0x80000u) : uint2(b.x, b.y | 0x80000u);
    uint signA = a.y & 0x80000000u, signB = b.y & 0x80000000u;
    bool infA = (a.y & 0x7ff00000u) == 0x7ff00000u, infB = (b.y & 0x7ff00000u) == 0x7ff00000u;
    if (infA && infB && signA != signB) return uint2(0u, 0x7ff80000u);
    if (infA) return a;
    if (infB) return b;
    uint2 sigA = tb_u64_shl(tb_f64_sig(a, keepIn), 10u), sigB = tb_u64_shl(tb_f64_sig(b, keepIn), 10u);
    int expA = tb_f64_exp(a), expB = tb_f64_exp(b);
    if ((sigA.x | sigA.y | sigB.x | sigB.y) == 0u) return uint2(0u, signA & signB);
    if (expA < expB || (expA == expB && tb_u64_less(sigA, sigB))) {
        uint2 ts = sigA; sigA = sigB; sigB = ts;
        int te = expA; expA = expB; expB = te;
        uint tn = signA; signA = signB; signB = tn;
    }
    sigB = tb_u64_shr_jam(sigB, uint(expA - expB));
    int e = expA - 1;
    uint2 sig;
    if (signA == signB) {
        sig = tb_u64_add(sigA, sigB);
        if (sig.y >= 0x80000000u) { sig = tb_u64_shr_jam(sig, 1u); e += 1; }
    } else {
        sig = tb_u64_sub(sigA, sigB);
        if ((sig.x | sig.y) == 0u) return uint2(0u, 0u);
        uint shift = tb_u64_clz(sig) - 1u;
        sig = tb_u64_shl(sig, shift);
        e -= int(shift);
    }
    return tb_f64_pack(signA, e, sig, keepOut);
}
uint2 tb_f64_mul(uint2 a, uint2 b, bool keepIn, bool keepOut) {
    if (tb_f64_nan(a) || tb_f64_nan(b)) return tb_f64_nan(a) ? uint2(a.x, a.y | 0x80000u) : uint2(b.x, b.y | 0x80000u);
    uint sign = (a.y ^ b.y) & 0x80000000u;
    bool infA = (a.y & 0x7ff00000u) == 0x7ff00000u, infB = (b.y & 0x7ff00000u) == 0x7ff00000u;
    uint2 sigA = tb_f64_sig(a, keepIn), sigB = tb_f64_sig(b, keepIn);
    bool zero = (sigA.x | sigA.y) == 0u || (sigB.x | sigB.y) == 0u;
    if (infA || infB) return zero ? uint2(0u, 0x7ff80000u) : uint2(0u, sign | 0x7ff00000u);
    if (zero) return uint2(0u, sign);
    int expA = tb_f64_exp(a), expB = tb_f64_exp(b);
    sigA = tb_f64_normalize(sigA, expA);
    sigB = tb_f64_normalize(sigB, expB);
    uint4 p = tb_u64_mul(tb_u64_shl(sigA, 10u), tb_u64_shl(sigB, 11u));
    uint2 sig = uint2(p.z | ((p.x | p.y) != 0u ? 1u : 0u), p.w);
    int e = expA + expB - 0x3ff;
    if (sig.y < 0x40000000u) { e -= 1; sig = tb_u64_shl(sig, 1u); }
    return tb_f64_pack(sign, e, sig, keepOut);
}
uint2 tb_f64_fma(uint2 a, uint2 b, uint2 c, bool keepIn, bool keepOut) {
    if (tb_f64_nan(a) || tb_f64_nan(b) || tb_f64_nan(c))
        return tb_f64_nan(a) ? uint2(a.x, a.y | 0x80000u) : tb_f64_nan(b) ? uint2(b.x, b.y | 0x80000u) : uint2(c.x, c.y | 0x80000u);
    uint signP = (a.y ^ b.y) & 0x80000000u, signC = c.y & 0x80000000u;
    bool infA = (a.y & 0x7ff00000u) == 0x7ff00000u, infB = (b.y & 0x7ff00000u) == 0x7ff00000u;
    bool infC = (c.y & 0x7ff00000u) == 0x7ff00000u;
    uint2 sigA = tb_f64_sig(a, keepIn), sigB = tb_f64_sig(b, keepIn), sigC = tb_f64_sig(c, keepIn);
    bool zeroP = (sigA.x | sigA.y) == 0u || (sigB.x | sigB.y) == 0u, zeroC = (sigC.x | sigC.y) == 0u;
    if (infA || infB) return zeroP || (infC && signC != signP) ? uint2(0u, 0x7ff80000u) : uint2(0u, signP | 0x7ff00000u);
    if (infC) return c;
    if (zeroP) return zeroC ? uint2(0u, signP & signC) : tb_f64_pack(signC, tb_f64_exp(c) - 1, tb_u64_shl(sigC, 10u), keepOut);
    int expA = tb_f64_exp(a), expB = tb_f64_exp(b);
    sigA = tb_f64_normalize(sigA, expA);
    sigB = tb_f64_normalize(sigB, expB);
    // The exact product, leading one at bit 125 or 126, is p * 2^scale.
    uint4 p = tb_u128_shl(tb_u64_mul(sigA, sigB), 21u);
    int scale = expA + expB - 2171;
    uint sign = signP;
    if (!zeroC) {
        // The addend, leading one at bit 126. Whenever alignment drops bits
        // the other operand is at least 2^20 times larger, so the sticky
        // bit stays far below the rounding position.
        int expC = tb_f64_exp(c);
        sigC = tb_f64_normalize(sigC, expC);
        uint4 q = uint4(0u, 0u, tb_u64_shl(sigC, 10u));
        int scaleC = expC - 1149;
        if (scaleC > scale) { p = tb_u128_shr_jam(p, uint(scaleC - scale)); scale = scaleC; }
        else q = tb_u128_shr_jam(q, uint(scale - scaleC));
        if (signC == signP) {
            p = tb_u128_add(p, q);
        } else if (tb_u128_less(p, q)) {
            p = tb_u128_sub(q, p);
            sign = signC;
        } else {
            p = tb_u128_sub(p, q);
            if ((p.x | p.y | p.z | p.w) == 0u) return uint2(0u, 0u);
        }
    }
    if (p.w >= 0x80000000u) { p = tb_u128_shr_jam(p, 1u); scale += 1; }
    else { uint shift = tb_u128_clz(p) - 1u; p = tb_u128_shl(p, shift); scale -= int(shift); }
    uint2 sig = uint2(p.z | ((p.x | p.y) != 0u ? 1u : 0u), p.w);
    return tb_f64_pack(sign, scale + 1148, sig, keepOut);
}
uint2 tb_f64_ldexp(uint2 a, int n, bool keepIn, bool keepOut) {
    if (tb_f64_nan(a)) return uint2(a.x, a.y | 0x80000u);
    if ((a.y & 0x7ff00000u) == 0x7ff00000u) return a;
    uint sign = a.y & 0x80000000u;
    uint2 sig = tb_f64_sig(a, keepIn);
    if ((sig.x | sig.y) == 0u) return uint2(0u, sign);
    int e = tb_f64_exp(a);
    sig = tb_f64_normalize(sig, e);
    return tb_f64_pack(sign, e - 1 + clamp(n, -2200, 2200), tb_u64_shl(sig, 10u), keepOut);
}
"#;

/// Forms whose first source is an f64 register pair.
fn pair_source(m: &str) -> bool {
    matches!(m, "v_trunc_f64" | "v_floor_f64" | "v_ceil_f64" | "v_rndne_f64" | "v_cvt_i32_f64" | "v_cvt_u32_f64" | "v_cvt_f32_f64") || arith(m)
}

/// Forms whose result is an f64 register pair.
pub(super) fn pair_result(m: &str) -> bool {
    matches!(m, "v_trunc_f64" | "v_floor_f64" | "v_ceil_f64" | "v_rndne_f64" | "v_cvt_f64_i32" | "v_cvt_f64_u32" | "v_cvt_f64_f32") || arith(m)
}

/// Soft-float arithmetic forms, which use `ARITH_HELPERS`.
pub(super) fn arith(m: &str) -> bool {
    matches!(m, "v_add_f64" | "v_mul_f64" | "v_fma_f64" | "v_ldexp_f64")
}

/// Number of leading f64 register-pair sources.
fn pair_sources(m: &str) -> usize {
    match m {
        "v_fma_f64" => 3,
        "v_add_f64" | "v_mul_f64" => 2,
        _ if pair_source(m) => 1,
        _ => 0,
    }
}

pub(super) fn handles(m: &str) -> bool {
    pair_source(m) || pair_result(m)
}

/// High register of each register-pair source, for the live-in analysis.
pub(super) fn pair_source_highs(ins: &Insn) -> Vec<Opnd> {
    ins.src.iter().take(pair_sources(&ins.mnemonic)).filter_map(|o| match *o {
        Opnd::Vgpr(n) => n.checked_add(1).map(Opnd::Vgpr),
        Opnd::Sgpr(n @ 0..=104) => Some(Opnd::Sgpr(n + 1)),
        _ => None,
    }).collect()
}

impl Emitter<'_> {
    pub(super) fn lower_f64(&mut self, ins: &Insn) -> Result<bool, String> {
        let m = ins.mnemonic.as_str();
        if !handles(m) {
            return Ok(false);
        }
        if arith(m) {
            return self.lower_f64_arith(ins).map(|()| true);
        }
        // Ordinary VGPR operands only; the VOP3 form without abs, neg,
        // clamp or omod. Verified in compute shaders only.
        let plain = match ins.enc {
            Enc::Vop1 => ins.aux.sdwa.is_none() && ins.raw.len() == 1,
            Enc::Vop3 => ins.raw.len() == 2 && ins.raw[0] & 0xff00 == 0 && ins.raw[1] >> 27 == 0,
            _ => false,
        };
        let (Some(Opnd::Vgpr(dst)), Some(Opnd::Vgpr(src))) = (ins.dst.first(), ins.src.first()) else {
            return Err("unsupported f64 operand".into());
        };
        if !plain || !ins.is_complete() {
            return Err("unsupported f64 encoding or modifiers".into());
        }
        if self.ex.stage != super::Stage::Compute {
            return Err("unverified f64 shader stage".into());
        }
        let (dst, src) = (*dst, *src);
        if (pair_result(m) && dst == u8::MAX) || (pair_source(m) && src == u8::MAX) {
            return Err("VGPR pair overflows".into());
        }
        let source = if pair_source(m) {
            let high = Opnd::Vgpr(src + 1);
            self.touch(&high);
            format!("uint2({}, {})", r(&Opnd::Vgpr(src))?, r(&high)?)
        } else {
            r(&Opnd::Vgpr(src))?
        };
        let keep = if self.regs.contains("tb_fp_mode") { "(tb_fp_mode & 0x40u) != 0u" } else { "true" };
        // The f32 side follows MODE's f32 denormal field: bit 4 keeps input denormals and
        // bit 5 keeps results. Its default (MODE 0xc0) flushes both.
        let tracked = self.regs.contains("tb_fp_mode");
        let keep32 = |bit: u32| if tracked { format!("(tb_fp_mode & {bit:#x}u) != 0u") } else { "false".into() };
        let value = match m {
            "v_trunc_f64" => format!("tb_f64_round({source}, 0u, {keep})"),
            "v_floor_f64" => format!("tb_f64_round({source}, 1u, {keep})"),
            "v_ceil_f64" => format!("tb_f64_round({source}, 2u, {keep})"),
            "v_rndne_f64" => format!("tb_f64_round({source}, 3u, {keep})"),
            "v_cvt_f64_i32" => format!("tb_i32_to_f64({source}, true)"),
            "v_cvt_f64_u32" => format!("tb_i32_to_f64({source}, false)"),
            "v_cvt_f64_f32" => format!("tb_f32_to_f64({source}, {})", keep32(0x10)),
            "v_cvt_f32_f64" => format!("tb_f64_to_f32({source}, {})", keep32(0x20)),
            "v_cvt_i32_f64" => format!("tb_f64_to_i32({source}, true)"),
            _ => format!("tb_f64_to_i32({source}, false)"),
        };
        if !pair_result(m) {
            self.wv(&Opnd::Vgpr(dst), &value)?;
            return Ok(true);
        }
        let t = self.tmp;
        self.tmp += 1;
        self.line(&format!("uint2 t{t}f64 = {value};"));
        self.wv(&Opnd::Vgpr(dst), &format!("t{t}f64.x"))?;
        let high = Opnd::Vgpr(dst + 1);
        self.touch(&high);
        self.wv(&high, &format!("t{t}f64.y"))?;
        Ok(true)
    }

    /// V_ADD_F64, V_MUL_F64, V_FMA_F64 and V_LDEXP_F64 (VOP3). Source abs
    /// and neg act on the sign bit of each pair; clamp, omod and op_sel are
    /// refused. Register pairs, inline constants and literals as sources.
    fn lower_f64_arith(&mut self, ins: &Insn) -> Result<(), String> {
        let m = ins.mnemonic.as_str();
        let pairs = pair_sources(m);
        if ins.enc != Enc::Vop3 || !ins.is_complete() {
            return Err("unsupported f64 arithmetic encoding".into());
        }
        if ins.aux.imm != 0 || ins.aux.imm2 != 0 {
            return Err("unsupported f64 clamp or output modifier".into());
        }
        // Abs and neg only on the f64 sources (not on ldexp's exponent).
        let modifiable = (1u8 << pairs) - 1;
        if ins.raw[0] & 0x7800 != 0 || ins.aux.mask & !(modifiable | modifiable << 3) != 0 {
            return Err("unsupported f64 op_sel or source modifier".into());
        }
        if self.ex.stage != super::Stage::Compute {
            return Err("unverified f64 shader stage".into());
        }
        let Some(&Opnd::Vgpr(dst)) = ins.dst.first() else {
            return Err("unsupported f64 operand".into());
        };
        if dst == u8::MAX {
            return Err("VGPR pair overflows".into());
        }
        let mut args = Vec::new();
        for (i, operand) in ins.src.iter().take(pairs).enumerate() {
            let (low, mut high) = self.compare_pair(operand, "f64")?;
            if ins.aux.mask & (1 << i) != 0 {
                high = format!("({high} & 0x7fffffffu)");
            }
            if ins.aux.mask & (8 << i) != 0 {
                high = format!("({high} ^ 0x80000000u)");
            }
            args.push(format!("uint2({low}, {high})"));
        }
        if m == "v_ldexp_f64" {
            let exponent = &ins.src[1];
            if !matches!(exponent, Opnd::Vgpr(_) | Opnd::Sgpr(_) | Opnd::ConstInt(_) | Opnd::Literal(_)) {
                return Err("unsupported ldexp exponent operand".into());
            }
            args.push(format!("asint({})", r(exponent)?));
        }
        let mode = self.regs.contains("tb_fp_mode");
        let keep_in = if mode { "(tb_fp_mode & 0x40u) != 0u" } else { "true" };
        let keep_out = if mode { "(tb_fp_mode & 0x80u) != 0u" } else { "true" };
        let op = &m[2..m.len() - 4];
        let t = self.tmp;
        self.tmp += 1;
        self.line(&format!("uint2 t{t}f64 = tb_f64_{op}({}, {keep_in}, {keep_out});", args.join(", ")));
        self.wv(&Opnd::Vgpr(dst), &format!("t{t}f64.x"))?;
        let high = Opnd::Vgpr(dst + 1);
        self.touch(&high);
        self.wv(&high, &format!("t{t}f64.y"))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use crate::{hlsl, ShaderExtract, Stage};

    const V4: [u32; 2] = [0x7e08_02ff, 0x1357_2468];
    const V5: [u32; 2] = [0x7e0a_02ff, 0x2468_1357];
    const ENDPGM: u32 = 0xbf81_0000;

    /// Translate `v_mov_b32 v4/v5, literal`, then `body`, then `s_endpgm`.
    pub(in crate::hlsl) fn translate(stage: Stage, body: &[u32]) -> Result<String, String> {
        let words: Vec<u32> = V4.iter().chain(&V5).chain(body).chain(&[ENDPGM]).copied().collect();
        let isa: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let ex = ShaderExtract { flat_inputs: Vec::new(), flat_outputs: Vec::new(),
            stage, shader_model: (6, 0), required_features: None, bitcode_stripped: true,
            fspx_config: None, ps09_pixel_inputs: None, fssm: Vec::new(), entry: Some("main".into()),
            isa_data: isa.clone(), isa, isa_offset_in_fssm: 0, psv_threads: None, isa_preamble: [0, 1, 1, 1],
            has_input_sig: false, has_output_sig: false, has_psv0: false, has_root_sig: false,
            input_sig: Vec::new(), output_sig: Vec::new(), resources: Vec::new(), gs_info: None,
            mesh_info: None, primitive_sig: Vec::new(), root_sig: None,
        };
        hlsl::translate(&ex).map(|t| t.hlsl)
    }

    #[test]
    fn pair_forms_write_both_destination_words() {
        // LLVM gfx1030: v_floor_f64_e32 v[8:9], v[4:5]
        let hlsl = translate(Stage::Compute, &[0x7e10_3504]).unwrap();
        assert!(hlsl.contains("tb_f64_round(uint2(v4, v5), 1u, true)"), "{hlsl}");
        assert!(hlsl.contains("v8 = t") && hlsl.contains("v9 = t"), "{hlsl}");
        // v_rndne_f64_e64 v[8:9], v[4:5]
        assert!(translate(Stage::Compute, &[0xd599_0008, 0x0201_0104]).unwrap().contains(", 3u, true)"));
        // v_cvt_i32_f64_e32 v8, v[4:5]
        assert!(translate(Stage::Compute, &[0x7e10_0704]).unwrap().contains("tb_f64_to_i32(uint2(v4, v5), true)"));
        // v_cvt_f64_u32_e32 v[8:9], v4
        assert!(translate(Stage::Compute, &[0x7e10_2d04]).unwrap().contains("tb_i32_to_f64(v4, false)"));
    }

    #[test]
    fn f32_conversions_round_and_follow_the_f32_denormal_mode() {
        // LLVM gfx1030: v_cvt_f64_f32_e32 v[8:9], v4
        let hlsl = translate(Stage::Compute, &[0x7e10_2104]).unwrap();
        assert!(hlsl.contains("= tb_f32_to_f64(v4, false);"), "{hlsl}");
        assert!(hlsl.contains("v8 = t") && hlsl.contains("v9 = t"), "{hlsl}");
        // v_cvt_f32_f64_e32 v8, v[4:5]
        assert!(translate(Stage::Compute, &[0x7e10_1f04]).unwrap().contains("v8 = tb_f64_to_f32(uint2(v4, v5), false)"));
        // s_denorm_mode 0 makes the mode dynamic.
        let hlsl = translate(Stage::Compute, &[0xbfa5_0000, 0x7e10_2104, 0x7e14_1f04]).unwrap();
        assert!(hlsl.contains("tb_f32_to_f64(v4, (tb_fp_mode & 0x10u) != 0u)"), "{hlsl}");
        assert!(hlsl.contains("v10 = tb_f64_to_f32(uint2(v4, v5), (tb_fp_mode & 0x20u) != 0u)"), "{hlsl}");
        // Modifiers stay refused: v_cvt_f64_f32_e64 v[8:9], -v4 and v_cvt_f32_f64_e64 ... clamp
        assert!(translate(Stage::Compute, &[0xd590_0008, 0x2000_0104]).is_err());
        assert!(translate(Stage::Compute, &[0xd58f_8008, 0x0000_0104]).is_err());
    }

    #[test]
    fn denormal_mode_controls_input_flushing() {
        // s_denorm_mode 0, then v_floor_f64_e32 v[8:9], v[4:5]
        let hlsl = translate(Stage::Compute, &[0xbfa5_0000, 0x7e10_3504]).unwrap();
        assert!(hlsl.contains("(tb_fp_mode & 0x40u) != 0u"), "{hlsl}");
    }

    #[test]
    fn arithmetic_forms_call_soft_float_helpers() {
        // LLVM gfx1030: v_add_f64 v[8:9], v[4:5], v[6:7]
        let hlsl = translate(Stage::Compute, &[0xd564_0008, 0x0002_0d04]).unwrap();
        assert!(hlsl.contains("tb_f64_add(uint2(v4, v5), uint2(v6, v7), true, true);"), "{hlsl}");
        assert!(hlsl.contains("uint2 tb_f64_add(uint2 a, uint2 b, bool keepIn, bool keepOut)"));
        assert!(hlsl.contains("v8 = t") && hlsl.contains("v9 = t"), "{hlsl}");
        assert!(!hlsl.contains("double"));
        // Both halves of the unwritten second pair are read on entry.
        assert!(hlsl.lines().next().unwrap().contains(" v6 v7"), "{hlsl}");
        // LLVM gfx1030: v_mul_f64 v[8:9], v[4:5], v[6:7]
        let hlsl = translate(Stage::Compute, &[0xd565_0008, 0x0002_0d04]).unwrap();
        assert!(hlsl.contains("tb_f64_mul(uint2(v4, v5), uint2(v6, v7), true, true);"), "{hlsl}");
        // LLVM gfx1030: v_fma_f64 v[8:9], v[4:5], v[6:7], v[4:5]
        let hlsl = translate(Stage::Compute, &[0xd54c_0008, 0x0412_0d04]).unwrap();
        assert!(hlsl.contains("tb_f64_fma(uint2(v4, v5), uint2(v6, v7), uint2(v4, v5), true, true);"), "{hlsl}");
        // LLVM gfx1030: v_ldexp_f64 v[8:9], v[4:5], v6
        let hlsl = translate(Stage::Compute, &[0xd568_0008, 0x0002_0d04]).unwrap();
        assert!(hlsl.contains("tb_f64_ldexp(uint2(v4, v5), asint(v6), true, true);"), "{hlsl}");
        // The rounding and conversion helpers alone do not pull in soft float.
        assert!(!translate(Stage::Compute, &[0x7e10_3504]).unwrap().contains("tb_f64_add"));
    }

    #[test]
    fn arithmetic_source_modifiers_flip_or_clear_the_sign_bit() {
        // LLVM gfx1030: v_add_f64 v[8:9], v[4:5], -v[6:7]
        let hlsl = translate(Stage::Compute, &[0xd564_0008, 0x4002_0d04]).unwrap();
        assert!(hlsl.contains("tb_f64_add(uint2(v4, v5), uint2(v6, (v7 ^ 0x80000000u)), true, true);"), "{hlsl}");
        // LLVM gfx1030: v_add_f64 v[8:9], |v[4:5]|, -|v[6:7]|
        let hlsl = translate(Stage::Compute, &[0xd564_0308, 0x4002_0d04]).unwrap();
        assert!(hlsl.contains("tb_f64_add(uint2(v4, (v5 & 0x7fffffffu)), uint2(v6, ((v7 & 0x7fffffffu) ^ 0x80000000u)), true, true);"), "{hlsl}");
        // LLVM gfx1030: v_fma_f64 v[8:9], -v[4:5], v[6:7], -v[2:3]
        let hlsl = translate(Stage::Compute, &[0xd54c_0008, 0xa40a_0d04]).unwrap();
        assert!(hlsl.contains("tb_f64_fma(uint2(v4, (v5 ^ 0x80000000u)), uint2(v6, v7), uint2(v2, (v3 ^ 0x80000000u)), true, true);"), "{hlsl}");
        // LLVM gfx1030: v_ldexp_f64 v[8:9], -v[4:5], s6
        let hlsl = translate(Stage::Compute, &[0xd568_0008, 0x2000_0d04]).unwrap();
        assert!(hlsl.contains("tb_f64_ldexp(uint2(v4, (v5 ^ 0x80000000u)), asint(s6), true, true);"), "{hlsl}");
    }

    #[test]
    fn arithmetic_accepts_scalar_pairs_and_constants() {
        // LLVM gfx1030: v_add_f64 v[8:9], s[4:5], v[6:7]
        let hlsl = translate(Stage::Compute, &[0xd564_0008, 0x0002_0c04]).unwrap();
        assert!(hlsl.contains("tb_f64_add(uint2(s4, s5), uint2(v6, v7), true, true);"), "{hlsl}");
        // LLVM gfx1030: v_mul_f64 v[8:9], 2.0, v[4:5]
        let hlsl = translate(Stage::Compute, &[0xd565_0008, 0x0002_08f4]).unwrap();
        assert!(hlsl.contains("tb_f64_mul(uint2(0x0u, 0x40000000u), uint2(v4, v5), true, true);"), "{hlsl}");
        // LLVM gfx1030: v_mul_f64 v[8:9], 0x40080000, v[4:5] (the literal is the high word)
        let hlsl = translate(Stage::Compute, &[0xd565_0008, 0x0002_08ff, 0x4008_0000]).unwrap();
        assert!(hlsl.contains("tb_f64_mul(uint2(0x0u, 0x40080000u), uint2(v4, v5), true, true);"), "{hlsl}");
        // LLVM gfx1030: v_add_f64 v[8:9], -1, v[4:5] (integer constants are 64-bit integers)
        let hlsl = translate(Stage::Compute, &[0xd564_0008, 0x0002_08c1]).unwrap();
        assert!(hlsl.contains("tb_f64_add(uint2(0xffffffffu, 0xffffffffu), uint2(v4, v5), true, true);"), "{hlsl}");
        // LLVM gfx1030: v_fma_f64 v[8:9], v[4:5], v[6:7], 1.0
        let hlsl = translate(Stage::Compute, &[0xd54c_0008, 0x03ca_0d04]).unwrap();
        assert!(hlsl.contains("uint2(0x0u, 0x3ff00000u), true, true);"), "{hlsl}");
    }

    #[test]
    fn arithmetic_follows_the_f64_denormal_mode() {
        // s_denorm_mode 0, then v_add_f64 v[8:9], v[4:5], v[6:7]
        let hlsl = translate(Stage::Compute, &[0xbfa5_0000, 0xd564_0008, 0x0002_0d04]).unwrap();
        assert!(hlsl.contains("uint2(v6, v7), (tb_fp_mode & 0x40u) != 0u, (tb_fp_mode & 0x80u) != 0u);"), "{hlsl}");
    }

    #[test]
    fn arithmetic_output_modifiers_and_unmodelled_forms_are_refused() {
        // LLVM gfx1030: v_add_f64 v[8:9], v[4:5], v[6:7] clamp
        assert!(translate(Stage::Compute, &[0xd564_8008, 0x0002_0d04]).is_err());
        // LLVM gfx1030: v_mul_f64 v[8:9], v[4:5], v[6:7] mul:2
        assert!(translate(Stage::Compute, &[0xd565_0008, 0x0802_0d04]).is_err());
        // v_add_f64 v[8:9], v[4:5], v[6:7] with op_sel bit 11 set by hand
        assert!(translate(Stage::Compute, &[0xd564_0808, 0x0002_0d04]).is_err());
        // v_ldexp_f64 v[8:9], v[4:5], -v6 (llvm-mc rejects it; neg bit set by hand)
        assert!(translate(Stage::Compute, &[0xd568_0008, 0x4002_0d04]).is_err());
        // LLVM gfx1030: v_ldexp_f64 v[8:9], v[4:5], 2.0 (an inline float as the integer exponent)
        assert!(translate(Stage::Compute, &[0xd568_0008, 0x0001_e904]).is_err());
        // LLVM gfx1030: v_min_f64 v[8:9], v[4:5], v[6:7] and v_max_f64 (not modelled)
        assert!(translate(Stage::Compute, &[0xd566_0008, 0x0002_0d04]).is_err());
        assert!(translate(Stage::Compute, &[0xd567_0008, 0x0002_0d04]).is_err());
        assert!(translate(Stage::Pixel, &[0xd564_0008, 0x0002_0d04]).is_err());
    }

    #[test]
    fn modifiers_scalar_sources_and_other_stages_are_refused() {
        // v_floor_f64_e64 v[8:9], -v[4:5]
        assert!(translate(Stage::Compute, &[0xd59a_0008, 0x2201_0104]).is_err());
        // v_floor_f64_e32 v[8:9], s[4:5]
        assert!(translate(Stage::Compute, &[0x7e10_3404]).is_err());
        assert!(translate(Stage::Pixel, &[0x7e10_3504]).is_err());
    }
}
