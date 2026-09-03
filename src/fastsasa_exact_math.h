/*
 * Bit-reproducible double-precision helpers shared by the CPU reference and
 * the CUDA FP64 kernels.
 *
 * Two things make a GPU FP64 result differ from the host in the last bit even
 * when the algorithm is identical: the CUDA compiler contracts a*b+c into a
 * fused multiply-add by default, and the CUDA libm acos/atan2 round
 * differently from the host libm in a sizable fraction of inputs (measured:
 * ~14% for acos, ~27% for atan2). Both are avoided here. Every arithmetic
 * operation goes through the FASTSASA_X_* macros, which map to the
 * non-contractible __dmul_rn/__dadd_rn/... intrinsics in device code and to
 * plain IEEE operations on the host (the host build disables contraction with
 * -ffp-contract=off), and acos/atan2 are implemented once, in this header,
 * from the FreeBSD msun sources as distributed by musl, so the host and the
 * device execute the same sequence of correctly rounded operations.
 *
 * The acos/atan/atan2 implementations are derived from FreeBSD msun
 * (e_acos.c, s_atan.c, e_atan2.c) via musl:
 *
 *   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *   Developed at SunPro/SunSoft, a Sun Microsystems, Inc. business.
 *   Permission to use, copy, modify, and distribute this software is freely
 *   granted, provided that this notice is preserved.
 */
#ifndef FASTSASA_EXACT_MATH_H
#define FASTSASA_EXACT_MATH_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__)
#define FASTSASA_X_FN static __host__ __device__ __forceinline__
#else
#define FASTSASA_X_FN static inline
#endif

#if defined(__CUDA_ARCH__)
#define FASTSASA_X_MUL(a, b) __dmul_rn((a), (b))
#define FASTSASA_X_ADD(a, b) __dadd_rn((a), (b))
#define FASTSASA_X_SUB(a, b) __dsub_rn((a), (b))
#define FASTSASA_X_DIV(a, b) __ddiv_rn((a), (b))
#else
#define FASTSASA_X_MUL(a, b) ((a) * (b))
#define FASTSASA_X_ADD(a, b) ((a) + (b))
#define FASTSASA_X_SUB(a, b) ((a) - (b))
#define FASTSASA_X_DIV(a, b) ((a) / (b))
#endif

FASTSASA_X_FN uint32_t
fastsasa_x_high_word(double value)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return (uint32_t)(bits >> 32);
}

FASTSASA_X_FN uint32_t
fastsasa_x_low_word(double value)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return (uint32_t)bits;
}

FASTSASA_X_FN double
fastsasa_x_clear_low_word(double value)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    bits &= 0xffffffff00000000ull;
    memcpy(&value, &bits, sizeof(bits));
    return value;
}

/* asin/acos rational approximation R(z) from e_asin.c. */
FASTSASA_X_FN double
fastsasa_x_asin_r(double z)
{
    const double pS0 = 1.66666666666666657415e-01;
    const double pS1 = -3.25565818622400915405e-01;
    const double pS2 = 2.01212532134862925881e-01;
    const double pS3 = -4.00555345006794114027e-02;
    const double pS4 = 7.91534994289814532176e-04;
    const double pS5 = 3.47933107596021167570e-05;
    const double qS1 = -2.40339491173441421878e+00;
    const double qS2 = 2.02094576023350569471e+00;
    const double qS3 = -6.88283971605453293030e-01;
    const double qS4 = 7.70381505559019352791e-02;
    const double p = FASTSASA_X_MUL(
        z, FASTSASA_X_ADD(pS0, FASTSASA_X_MUL(
               z, FASTSASA_X_ADD(pS1, FASTSASA_X_MUL(
                      z, FASTSASA_X_ADD(pS2, FASTSASA_X_MUL(
                             z, FASTSASA_X_ADD(pS3, FASTSASA_X_MUL(
                                    z, FASTSASA_X_ADD(pS4, FASTSASA_X_MUL(z, pS5)))))))))));
    const double q = FASTSASA_X_ADD(
        1.0, FASTSASA_X_MUL(
                 z, FASTSASA_X_ADD(qS1, FASTSASA_X_MUL(
                        z, FASTSASA_X_ADD(qS2, FASTSASA_X_MUL(
                               z, FASTSASA_X_ADD(qS3, FASTSASA_X_MUL(z, qS4))))))));

    return FASTSASA_X_DIV(p, q);
}

FASTSASA_X_FN double
fastsasa_exact_acos(double x)
{
    const double pio2_hi = 1.57079632679489655800e+00;
    const double pio2_lo = 6.12323399573676603587e-17;
    const uint32_t hx = fastsasa_x_high_word(x);
    const uint32_t ix = hx & 0x7fffffffu;
    double z, w, s, c, df;

    if (ix >= 0x3ff00000u) {
        const uint32_t lx = fastsasa_x_low_word(x);

        if (((ix - 0x3ff00000u) | lx) == 0u) {
            return (hx >> 31) ? 2.0 * pio2_hi : 0.0;
        }
        return (x - x) / (x - x); /* NaN for |x| > 1 */
    }
    if (ix < 0x3fe00000u) {
        if (ix <= 0x3c600000u) return pio2_hi;
        return FASTSASA_X_SUB(
            pio2_hi,
            FASTSASA_X_SUB(x, FASTSASA_X_SUB(pio2_lo,
                                         FASTSASA_X_MUL(x, fastsasa_x_asin_r(FASTSASA_X_MUL(x, x))))));
    }
    if (hx >> 31) {
        z = FASTSASA_X_MUL(FASTSASA_X_ADD(1.0, x), 0.5);
        s = sqrt(z);
        w = FASTSASA_X_SUB(FASTSASA_X_MUL(fastsasa_x_asin_r(z), s), pio2_lo);
        return FASTSASA_X_MUL(2.0, FASTSASA_X_SUB(pio2_hi, FASTSASA_X_ADD(s, w)));
    }
    z = FASTSASA_X_MUL(FASTSASA_X_SUB(1.0, x), 0.5);
    s = sqrt(z);
    df = fastsasa_x_clear_low_word(s);
    c = FASTSASA_X_DIV(FASTSASA_X_SUB(z, FASTSASA_X_MUL(df, df)), FASTSASA_X_ADD(s, df));
    w = FASTSASA_X_ADD(FASTSASA_X_MUL(fastsasa_x_asin_r(z), s), c);
    return FASTSASA_X_MUL(2.0, FASTSASA_X_ADD(df, w));
}

FASTSASA_X_FN double
fastsasa_exact_atan(double x)
{
    const double atanhi[4] = {
        4.63647609000806093515e-01,
        7.85398163397448278999e-01,
        9.82793723247329054082e-01,
        1.57079632679489655800e+00,
    };
    const double atanlo[4] = {
        2.26987774529616870924e-17,
        3.06161699786838301793e-17,
        1.39033110312309984516e-17,
        6.12323399573676603587e-17,
    };
    const double aT0 = 3.33333333333329318027e-01;
    const double aT1 = -1.99999999998764832476e-01;
    const double aT2 = 1.42857142725034663711e-01;
    const double aT3 = -1.11111104054623557880e-01;
    const double aT4 = 9.09088713343650656196e-02;
    const double aT5 = -7.69187620504482999495e-02;
    const double aT6 = 6.66107313738753120669e-02;
    const double aT7 = -5.83357013379057348645e-02;
    const double aT8 = 4.97687799461593236017e-02;
    const double aT9 = -3.65315727442169155270e-02;
    const double aT10 = 1.62858201153657823623e-02;
    uint32_t ix = fastsasa_x_high_word(x);
    const uint32_t sign = ix >> 31;
    double z, w, s1, s2;
    int id;

    ix &= 0x7fffffffu;
    if (ix >= 0x44100000u) { /* |x| >= 2^66 */
        if (x != x) return x;
        z = atanhi[3];
        return sign ? -z : z;
    }
    if (ix < 0x3fdc0000u) { /* |x| < 0.4375 */
        if (ix < 0x3e400000u) return x; /* |x| < 2^-27 */
        id = -1;
    } else {
        x = fabs(x);
        if (ix < 0x3ff30000u) { /* |x| < 1.1875 */
            if (ix < 0x3fe60000u) { /* 7/16 <= |x| < 11/16 */
                id = 0;
                x = FASTSASA_X_DIV(FASTSASA_X_SUB(FASTSASA_X_MUL(2.0, x), 1.0), FASTSASA_X_ADD(2.0, x));
            } else { /* 11/16 <= |x| < 19/16 */
                id = 1;
                x = FASTSASA_X_DIV(FASTSASA_X_SUB(x, 1.0), FASTSASA_X_ADD(x, 1.0));
            }
        } else {
            if (ix < 0x40038000u) { /* |x| < 2.4375 */
                id = 2;
                x = FASTSASA_X_DIV(FASTSASA_X_SUB(x, 1.5), FASTSASA_X_ADD(1.0, FASTSASA_X_MUL(1.5, x)));
            } else { /* 2.4375 <= |x| < 2^66 */
                id = 3;
                x = FASTSASA_X_DIV(-1.0, x);
            }
        }
    }
    z = FASTSASA_X_MUL(x, x);
    w = FASTSASA_X_MUL(z, z);
    s1 = FASTSASA_X_MUL(
        z, FASTSASA_X_ADD(aT0, FASTSASA_X_MUL(
               w, FASTSASA_X_ADD(aT2, FASTSASA_X_MUL(
                      w, FASTSASA_X_ADD(aT4, FASTSASA_X_MUL(
                             w, FASTSASA_X_ADD(aT6, FASTSASA_X_MUL(
                                    w, FASTSASA_X_ADD(aT8, FASTSASA_X_MUL(w, aT10)))))))))));
    s2 = FASTSASA_X_MUL(
        w, FASTSASA_X_ADD(aT1, FASTSASA_X_MUL(
               w, FASTSASA_X_ADD(aT3, FASTSASA_X_MUL(
                      w, FASTSASA_X_ADD(aT5, FASTSASA_X_MUL(
                             w, FASTSASA_X_ADD(aT7, FASTSASA_X_MUL(w, aT9)))))))));
    if (id < 0) return FASTSASA_X_SUB(x, FASTSASA_X_MUL(x, FASTSASA_X_ADD(s1, s2)));
    z = FASTSASA_X_SUB(atanhi[id],
                     FASTSASA_X_SUB(FASTSASA_X_SUB(FASTSASA_X_MUL(x, FASTSASA_X_ADD(s1, s2)), atanlo[id]), x));
    return sign ? -z : z;
}

FASTSASA_X_FN double
fastsasa_exact_atan2(double y, double x)
{
    const double pi = 3.1415926535897931160E+00;
    const double pi_lo = 1.2246467991473531772E-16;
    uint32_t ix = fastsasa_x_high_word(x);
    const uint32_t lx = fastsasa_x_low_word(x);
    uint32_t iy = fastsasa_x_high_word(y);
    const uint32_t ly = fastsasa_x_low_word(y);
    uint32_t m;
    double z;

    if (x != x || y != y) return x + y;
    if (((ix - 0x3ff00000u) | lx) == 0u) return fastsasa_exact_atan(y); /* x = 1.0 */
    m = ((iy >> 31) & 1u) | ((ix >> 30) & 2u); /* 2*sign(x)+sign(y) */
    ix &= 0x7fffffffu;
    iy &= 0x7fffffffu;

    if ((iy | ly) == 0u) { /* y = 0 */
        switch (m) {
        case 0:
        case 1:
            return y;
        case 2:
            return pi;
        default:
            return -pi;
        }
    }
    if ((ix | lx) == 0u) return (m & 1u) ? -pi / 2 : pi / 2; /* x = 0 */
    if (ix == 0x7ff00000u) { /* x is INF */
        if (iy == 0x7ff00000u) {
            switch (m) {
            case 0:
                return pi / 4;
            case 1:
                return -pi / 4;
            case 2:
                return 3 * pi / 4;
            default:
                return -3 * pi / 4;
            }
        }
        switch (m) {
        case 0:
            return 0.0;
        case 1:
            return -0.0;
        case 2:
            return pi;
        default:
            return -pi;
        }
    }
    if (ix + (64u << 20) < iy || iy == 0x7ff00000u) { /* |y/x| > 2^64 */
        return (m & 1u) ? -pi / 2 : pi / 2;
    }
    if ((m & 2u) && iy + (64u << 20) < ix) { /* |y/x| < 2^-64, x < 0 */
        z = 0.0;
    } else {
        z = fastsasa_exact_atan(fabs(FASTSASA_X_DIV(y, x)));
    }
    switch (m) {
    case 0:
        return z;
    case 1:
        return -z;
    case 2:
        return FASTSASA_X_SUB(pi, FASTSASA_X_SUB(z, pi_lo));
    default:
        return FASTSASA_X_SUB(FASTSASA_X_SUB(z, pi_lo), pi);
    }
}

/* Kahan step shared by every host-side accumulation so totals, residue and
 * selection sums are the same double on every backend. */
FASTSASA_X_FN void
fastsasa_exact_kahan_add(double value, double *sum, double *compensation)
{
    const double corrected = value - *compensation;
    const double next = *sum + corrected;

    *compensation = (next - *sum) - corrected;
    *sum = next;
}

#endif /* FASTSASA_EXACT_MATH_H */
