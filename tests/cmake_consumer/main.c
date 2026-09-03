#include "fastsasa.h"

int
main(void)
{
    return fastsasa_abi_version() == FASTSASA_ABI_VERSION ? 0 : 1;
}
