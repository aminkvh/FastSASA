// FP64 math shared by the Vulkan shaders that need IEEE-exact double
// division, square root, acos and atan2 (GLSL has no double
// transcendental functions). Requires GL_ARB_gpu_shader_fp64.
#ifndef FASTSASA_FP64_MATH_GLSL
#define FASTSASA_FP64_MATH_GLSL

// acos/atan/atan2 derived from FreeBSD msun (e_acos.c, s_atan.c, e_atan2.c)
// via musl, the same operation sequence as src/fastsasa_exact_math.h used by
// the CPU reference and the CUDA kernels; `precise` blocks FMA contraction so
// every rounding matches.
//
//   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//   Developed at SunPro/SunSoft, a Sun Microsystems, Inc. business.
//   Permission to use, copy, modify, and distribute this software is freely
//   granted, provided that this notice is preserved.
//
// musl is Copyright (c) 2005-2020 Rich Felker, et al., MIT license
// (licenses/musl-MIT.txt).
double exact_nan()
{
    return packDouble2x32(uvec2(0u, 0x7ff80000u));
}

// Vulkan does not require FP64 division and square root to be correctly
// rounded, and some drivers (Mesa RADV) approximate them. These versions
// refine the driver's estimate to the correctly rounded IEEE result the CPU
// reference gets, so the arithmetic matches on every device. Residuals are
// computed exactly with Veltkamp/Dekker products rather than fma, because
// SPIR-V lets a driver split an fma that is not decorated NoContraction and
// glslang does not decorate fma; `precise` does protect these plain
// multiplies and adds. Inputs here are finite, normal and away from
// overflow; zero is handled for the square root.
void two_prod(double a, double b, out double p, out double e)
{
    const double split = 134217729.0; // 2^27 + 1
    precise double ca = split * a;
    precise double ah = ca - (ca - a);
    precise double al = a - ah;
    precise double cb = split * b;
    precise double bh = cb - (cb - b);
    precise double bl = b - bh;
    precise double product = a * b;
    precise double error = ((ah * bh - product) + ah * bl + al * bh) + al * bl;
    p = product;
    e = error;
}

double div_rn(double a, double b)
{
    precise double y = 1.0 / b;
    double p, pe;
    for (int i = 0; i < 2; ++i) {
        two_prod(b, y, p, pe);
        precise double e = (1.0 - p) - pe;
        precise double refined = y + y * e;
        y = refined;
    }
    precise double q = a * y;
    two_prod(b, q, p, pe);
    precise double r = (a - p) - pe;
    precise double result = q + r * y;
    return result;
}

double sqrt_rn(double a)
{
    if (a == 0.0) return a;
    precise double y = inversesqrt(a);
    precise double g = a * y;
    precise double h = 0.5 * y;
    double p, pe;
    for (int i = 0; i < 2; ++i) {
        two_prod(g, h, p, pe);
        precise double r = (0.5 - p) - pe;
        precise double g_next = g + g * r;
        precise double h_next = h + h * r;
        g = g_next;
        h = h_next;
    }
    two_prod(g, g, p, pe);
    precise double d = (a - p) - pe;
    precise double result = g + d * h;
    return result;
}
#define RN_DIV(a, b) div_rn((a), (b))
#define RN_SQRT(x) sqrt_rn(x)

double asin_r(double z)
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
    precise double p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    precise double q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    precise double r = RN_DIV(p, q);
    return r;
}

double exact_acos(double x)
{
    const double pio2_hi = 1.57079632679489655800e+00;
    const double pio2_lo = 6.12323399573676603587e-17;
    const uvec2 words = unpackDouble2x32(x);
    const uint hx = words.y;
    const uint ix = hx & 0x7fffffffu;

    if (ix >= 0x3ff00000u) {
        if (((ix - 0x3ff00000u) | words.x) == 0u) return (hx >> 31) != 0u ? 2.0 * pio2_hi : 0.0;
        return exact_nan();
    }
    if (ix < 0x3fe00000u) {
        if (ix <= 0x3c600000u) return pio2_hi;
        precise double xx = x * x;
        precise double t = x * asin_r(xx);
        precise double u = pio2_lo - t;
        precise double v = x - u;
        precise double r = pio2_hi - v;
        return r;
    }
    if ((hx >> 31) != 0u) {
        precise double z = (1.0 + x) * 0.5;
        precise double s = RN_SQRT(z);
        precise double w = asin_r(z) * s - pio2_lo;
        precise double sw = s + w;
        precise double r = 2.0 * (pio2_hi - sw);
        return r;
    }
    precise double z = (1.0 - x) * 0.5;
    precise double s = RN_SQRT(z);
    const double df = packDouble2x32(uvec2(0u, unpackDouble2x32(s).y));
    precise double c = RN_DIV(z - df * df, s + df);
    precise double w = asin_r(z) * s + c;
    precise double r = 2.0 * (df + w);
    return r;
}

double exact_atan(double input_value)
{
    const double atanhi[4] = double[4](
        4.63647609000806093515e-01, 7.85398163397448278999e-01,
        9.82793723247329054082e-01, 1.57079632679489655800e+00);
    const double atanlo[4] = double[4](
        2.26987774529616870924e-17, 3.06161699786838301793e-17,
        1.39033110312309984516e-17, 6.12323399573676603587e-17);
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
    double x = input_value;
    uint ix = unpackDouble2x32(x).y;
    const uint sign = ix >> 31;
    int id;

    ix &= 0x7fffffffu;
    if (ix >= 0x44100000u) {
        if (isnan(x)) return x;
        return sign != 0u ? -atanhi[3] : atanhi[3];
    }
    if (ix < 0x3fdc0000u) {
        if (ix < 0x3e400000u) return x;
        id = -1;
    } else {
        x = abs(x);
        if (ix < 0x3ff30000u) {
            if (ix < 0x3fe60000u) {
                id = 0;
                precise double t = RN_DIV(2.0 * x - 1.0, 2.0 + x);
                x = t;
            } else {
                id = 1;
                precise double t = RN_DIV(x - 1.0, x + 1.0);
                x = t;
            }
        } else {
            if (ix < 0x40038000u) {
                id = 2;
                precise double t = RN_DIV(x - 1.5, 1.0 + 1.5 * x);
                x = t;
            } else {
                id = 3;
                precise double t = RN_DIV(-1.0, x);
                x = t;
            }
        }
    }
    precise double z = x * x;
    precise double w = z * z;
    precise double s1 = z * (aT0 + w * (aT2 + w * (aT4 + w * (aT6 + w * (aT8 + w * aT10)))));
    precise double s2 = w * (aT1 + w * (aT3 + w * (aT5 + w * (aT7 + w * aT9))));
    precise double s = s1 + s2;
    if (id < 0) {
        precise double r = x - x * s;
        return r;
    }
    precise double xs = x * s;
    precise double r = atanhi[id] - ((xs - atanlo[id]) - x);
    return sign != 0u ? -r : r;
}

double exact_atan2(double y, double x)
{
    const double pi = 3.1415926535897931160e+00;
    const double pi_lo = 1.2246467991473531772e-16;
    if (isnan(x) || isnan(y)) return x + y;
    const uvec2 wx = unpackDouble2x32(x);
    const uvec2 wy = unpackDouble2x32(y);
    uint ix = wx.y;
    const uint lx = wx.x;
    uint iy = wy.y;
    const uint ly = wy.x;
    if (((ix - 0x3ff00000u) | lx) == 0u) return exact_atan(y);
    const uint m = ((iy >> 31) & 1u) | ((ix >> 30) & 2u);
    ix &= 0x7fffffffu;
    iy &= 0x7fffffffu;
    if ((iy | ly) == 0u) {
        if (m <= 1u) return y;
        return m == 2u ? pi : -pi;
    }
    if ((ix | lx) == 0u) return (m & 1u) != 0u ? -pi / 2.0 : pi / 2.0;
    if (ix == 0x7ff00000u) {
        if (iy == 0x7ff00000u) {
            if (m == 0u) return pi / 4.0;
            if (m == 1u) return -pi / 4.0;
            if (m == 2u) return 3.0 * pi / 4.0;
            return -3.0 * pi / 4.0;
        }
        if (m == 0u) return 0.0;
        if (m == 1u) return -0.0;
        if (m == 2u) return pi;
        return -pi;
    }
    if (ix + (64u << 20) < iy || iy == 0x7ff00000u) return (m & 1u) != 0u ? -pi / 2.0 : pi / 2.0;
    double z;
    if ((m & 2u) != 0u && iy + (64u << 20) < ix) {
        z = 0.0;
    } else {
        precise double q = RN_DIV(y, x);
        z = exact_atan(abs(q));
    }
    if (m == 0u) return z;
    if (m == 1u) return -z;
    if (m == 2u) {
        precise double r = pi - (z - pi_lo);
        return r;
    }
    precise double r = (z - pi_lo) - pi;
    return r;
}
#endif
