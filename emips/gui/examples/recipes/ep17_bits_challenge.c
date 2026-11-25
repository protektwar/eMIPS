/* MIPS by Example in eMIPS, episode 17, the challenge solved (the C version
 * of ep17_bits_challenge.asm). */
#include "emips.h"

static const unsigned pairs[4] = {123, 45, 1000, 1000};
static const unsigned nums[4] = {64, 96, 1, 0};

unsigned umul(unsigned a, unsigned b)     /* shift and add */
{
    unsigned p = 0;
    while (b) {
        if (b & 1)
            p += a;
        a <<= 1;
        b >>= 1;
    }
    return p;
}

int pow2(unsigned n)                      /* exactly one 1 bit */
{
    return n != 0 && (n & (n - 1)) == 0;
}

int main(void)
{
    for (int i = 0; i < 4; i += 2) {
        print_int(pairs[i]);
        print_string(" * ");
        print_int(pairs[i + 1]);
        print_string(" = ");
        print_int(umul(pairs[i], pairs[i + 1]));
        print_char('\n');
    }
    for (int i = 0; i < 4; i++) {
        print_int(nums[i]);
        print_string(pow2(nums[i]) ? " is a power of 2\n" : " is not a power of 2\n");
    }
    return 0;
}
