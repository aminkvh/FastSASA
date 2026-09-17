/* Builds against an installed FastSASA (find_package(FastSASA CONFIG)) and
 * runs a real calculation through it, so the installed headers, library,
 * soname, and ABI version are all exercised, not just present.
 *
 * Fixture: atom A (radius 1.0) at the origin, atom B (radius 0.5 + 1e-9) at
 * x = 1.5, one test point at +x. In FP64 A's point is buried (total = pi);
 * in FP32 B's radius rounds to 0.5 and the point is exposed (total = 5 pi). */
#include "fastsasa.h"
#include "fastsasa_cpu.h"
#include "fastsasa_trajectory.h"

#include <math.h>
#include <stdio.h>

static int
run(int precision, double expected)
{
    const double x[2] = {0.0, 1.5};
    const double y[2] = {0.0, 0.0};
    const double z[2] = {0.0, 0.0};
    const double expanded_radii[2] = {1.0, 0.5 + 1.0e-9};
    const double test_point[3] = {1.0, 0.0, 0.0};
    double sasa[2] = {0.0, 0.0};
    const int status = fastsasa_cpu_shrake_rupley_precision(
        2, 1, x, y, z, expanded_radii, test_point, 1, precision, sasa);
    const double total = sasa[0] + sasa[1];

    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "precision %d: status %d\n", precision, status);
        return 1;
    }
    if (fabs(total - expected) > 1.0e-6) {
        fprintf(stderr, "precision %d: total %.9f, expected %.9f\n", precision, total, expected);
        return 1;
    }
    return 0;
}

int
main(void)
{
    if (fastsasa_abi_version() != FASTSASA_ABI_VERSION) {
        fprintf(stderr, "ABI version mismatch: library %u, header %u\n",
                fastsasa_abi_version(), FASTSASA_ABI_VERSION);
        return 1;
    }
    if (fastsasa_sizeof_parameters() != sizeof(fastsasa_parameters)) {
        fprintf(stderr, "fastsasa_parameters layout mismatch\n");
        return 1;
    }
    if (run(FASTSASA_PRECISION_FP64, M_PI) || run(FASTSASA_PRECISION_FP32, 5.0 * M_PI)) return 1;
    printf("fastsasa_consumer,status,pass\n");
    return 0;
}
