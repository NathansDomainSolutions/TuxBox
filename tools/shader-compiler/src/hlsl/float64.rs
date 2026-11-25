//! Exactly specified double-precision forms on register pairs: rounding to
//! an integral value, conversions between f64 and 32-bit integers or f32,
//! the IEEE-754 binary64 add, multiply, fused multiply-add and ldexp, and
//! fract and frexp as LLVM 18 constant-folds the amdgcn intrinsics (fract
//! is x - floor(x) rounded, at most 0x3fefffffffffffff, NaN for infinities;
//! frexp normalizes denormals and gives exponent 0 for inf/NaN).
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
//! The same NaN rule and the flushing modes apply to fract and frexp. VOP3
//! abs and neg act on the sign bit of a floating-point source, a NaN's too.
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
uint2 tb_f64_fract(uint2 x, bool keepIn, bool keepOut) {
    // x - floor(x), rounded to nearest even, but at most 0x3fefffffffffffff
    // (the largest f64 below 1.0). Infinities give +qNaN (inf - inf).
    uint2 f = tb_f64_round(x, 1u, keepIn);
    uint2 d = tb_f64_add(x, uint2(f.x, f.y ^ 0x80000000u), keepIn, keepOut);
    return !tb_f64_nan(d) && tb_u64_less(uint2(0xffffffffu, 0x3fefffffu), d) ? uint2(0xffffffffu, 0x3fefffffu) : d;
}
uint2 tb_f64_frexp_mant(uint2 x, bool keepIn) {
    // Significand scaled into [0.5, 1) with x's sign; zeros and infinities
    // are returned as they are and a NaN is quieted.
    if (tb_f64_nan(x)) return uint2(x.x, x.y | 0x80000u);
    if ((x.y & 0x7ff00000u) == 0x7ff00000u) return x;
    uint sign = x.y & 0x80000000u;
    uint2 sig = tb_f64_sig(x, keepIn);
    if ((sig.x | sig.y) == 0u) return uint2(0u, sign);
    int e = tb_f64_exp(x);
    sig = tb_f64_normalize(sig, e);
    return uint2(sig.x, sign | 0x3fe00000u | (sig.y & 0xfffffu));
}
uint tb_f64_frexp_exp(uint2 x, bool keepIn) {
    // The matching exponent; 0 for zeros, infinities and NaNs.
    if ((x.y & 0x7ff00000u) == 0x7ff00000u) return 0u;
    uint2 sig = tb_f64_sig(x, keepIn);
    if ((sig.x | sig.y) == 0u) return 0u;
    int e = tb_f64_exp(x);
    sig = tb_f64_normalize(sig, e);
    return asuint(e - 1022);
}
"#;

/// Forms whose first source is an f64 register pair.
fn pair_source(m: &str) -> bool {
    matches!(m, "v_trunc_f64" | "v_floor_f64" | "v_ceil_f64" | "v_rndne_f64" | "v_cvt_i32_f64" | "v_cvt_u32_f64" | "v_cvt_f32_f64")
        || extras(m)
        || arith(m)
}

/// Forms whose result is an f64 register pair.
pub(super) fn pair_result(m: &str) -> bool {
    matches!(m, "v_trunc_f64" | "v_floor_f64" | "v_ceil_f64" | "v_rndne_f64" | "v_cvt_f64_i32" | "v_cvt_f64_u32" | "v_cvt_f64_f32")
        || matches!(m, "v_fract_f64" | "v_frexp_mant_f64")
        || arith(m)
}

/// Soft-float arithmetic forms, which use `ARITH_HELPERS`.
pub(super) fn arith(m: &str) -> bool {
    matches!(m, "v_add_f64" | "v_mul_f64" | "v_fma_f64" | "v_ldexp_f64")
}

/// Single-source forms whose helpers live in `ARITH_HELPERS`.
fn extras(m: &str) -> bool {
    matches!(m, "v_fract_f64" | "v_frexp_mant_f64" | "v_frexp_exp_i32_f64")
}

/// Forms that need `ARITH_HELPERS`.
pub(super) fn soft(m: &str) -> bool {
    arith(m) || extras(m)
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

/// Apply VOP3 source abs and neg (`mask` bits 0 and 3) to the word holding the sign bit.
fn sign_modifiers(word: String, mask: u8) -> String {
    let word = if mask & 1 != 0 { format!("({word} & 0x7fffffffu)") } else { word };
    if mask & 8 != 0 { format!("({word} ^ 0x80000000u)") } else { word }
}

impl Emitter<'_> {
    pub(super) fn lower_f64(&mut self, ins: &Insn) -> Result<bool, String> {
        let m = ins.mnemonic.as_str();
        if matches!(m, "v_min_f64" | "v_max_f64") {
            // LLVM 18 maps these to fminnum_ieee/fmaxnum_ieee in IEEE mode (a
            // signalling NaN gives a quiet NaN) and to fminnum/fmaxnum otherwise
            // (it gives the other operand). MODE.IEEE comes from PGM_RSRC1, which
            // is not tracked (LLVM defaults it off for graphics shaders), and
            // neither mapping pins the result for -0 against +0.
            return Err("f64 min/max not modelled: NaN and signed-zero results depend on MODE.IEEE, which is not tracked".into());
        }
        if !handles(m) {
            return Ok(false);
        }
        if arith(m) {
            return self.lower_f64_arith(ins).map(|()| true);
        }
        // VOP1, or VOP3 without clamp, omod or op_sel. Abs and neg act on the
        // sign bit of a floating-point source (not the integer converts').
        let float_source = m != "v_cvt_f64_i32" && m != "v_cvt_f64_u32";
        match ins.enc {
            Enc::Vop1 if ins.aux.sdwa.is_none() => {}
            Enc::Vop3 if ins.raw.len() >= 2 => {
                if ins.aux.imm != 0 || ins.aux.imm2 != 0 || ins.raw[0] & 0x7800 != 0 {
                    return Err("unsupported f64 clamp, output modifier or op_sel".into());
                }
                if ins.aux.mask & !(if float_source { 0x09 } else { 0 }) != 0 {
                    return Err("unsupported f64 source modifier".into());
                }
            }
            _ => return Err("unsupported f64 encoding or modifiers".into()),
        }
        if !ins.is_complete() {
            return Err("unsupported f64 encoding or modifiers".into());
        }
        if self.ex.stage != super::Stage::Compute {
            return Err("unverified f64 shader stage".into());
        }
        let (Some(&Opnd::Vgpr(dst)), Some(src)) = (ins.dst.first(), ins.src.first()) else {
            return Err("unsupported f64 operand".into());
        };
        if pair_result(m) && dst == u8::MAX {
            return Err("VGPR pair overflows".into());
        }
        let mask = if ins.enc == Enc::Vop3 { ins.aux.mask } else { 0 };
        // Register pairs, inline constants (f64 values, or 64-bit integers)
        // and literals (the high word) as f64 sources; any 32-bit operand
        // r() models as the source of the 32-bit converts.
        let source = if pair_source(m) {
            let (low, high) = self.compare_pair(src, "f64")?;
            format!("uint2({low}, {})", sign_modifiers(high, mask))
        } else {
            sign_modifiers(r(src)?, mask)
        };
        let mode = self.regs.contains("tb_fp_mode");
        let keep = if mode { "(tb_fp_mode & 0x40u) != 0u" } else { "true" };
        let keep_out = if mode { "(tb_fp_mode & 0x80u) != 0u" } else { "true" };
        // The f32 side follows MODE's f32 denormal field: bit 4 keeps input denormals and
        // bit 5 keeps results. Its default (MODE 0xc0) flushes both.
        let keep32 = |bit: u32| if mode { format!("(tb_fp_mode & {bit:#x}u) != 0u") } else { "false".into() };
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
            "v_cvt_u32_f64" => format!("tb_f64_to_i32({source}, false)"),
            "v_fract_f64" => format!("tb_f64_fract({source}, {keep}, {keep_out})"),
            "v_frexp_mant_f64" => format!("tb_f64_frexp_mant({source}, {keep})"),
            "v_frexp_exp_i32_f64" => format!("tb_f64_frexp_exp({source}, {keep})"),
            _ => return Err("unsupported f64 operation".into()),
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
#[path = "../tests/hlsl/float64.rs"]
pub(super) mod tests;
