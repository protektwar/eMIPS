/* MIPS by Example in eMIPS, episode 17 — Division without div; counting 1
 * bits three ways (the C version of ep17_bits.asm). */
#include "emips.h"

static const unsigned pairs[4] = {1000, 7, 2026, 10};
static const int nums[2] = {2026, -1};
static const unsigned char bits4[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

/* long division in binary: one bit of n a step */
unsigned udiv(unsigned n, unsigned d, unsigned *rem)
{
    unsigned q = 0, r = 0;
    for (int i = 0; i < 32; i++) {
        r = r << 1 | n >> 31;             /* bring down the next bit */
        n <<= 1;
        q <<= 1;
        if (r >= d) {                     /* d fits: subtract, a 1 */
            r -= d;
            q |= 1;
        }
    }
    *rem = r;
    return q;
}

int ones_loop(unsigned n)                 /* one bit a step */
{
    int c = 0;
    while (n) {
        c += n & 1;
        n >>= 1;
    }
    return c;
}

int ones_clear(unsigned n)                /* one step per 1 */
{
    int c = 0;
    while (n) {
        n &= n - 1;                       /* the lowest 1 cleared */
        c++;
    }
    return c;
}

int ones_table(unsigned n)                /* four bits a step */
{
    int c = 0;
    for (int i = 0; i < 8; i++) {
        c += bits4[n & 15];
        n >>= 4;
    }
    return c;
}

int main(void)
{
    for (int i = 0; i < 4; i += 2) {
        unsigned r, q = udiv(pairs[i], pairs[i + 1], &r);
        print_int(pairs[i]);
        print_string(" / ");
        print_int(pairs[i + 1]);
        print_string(" = ");
        print_int(q);
        print_string(" remainder ");
        print_int(r);
        print_char('\n');
    }
    for (int i = 0; i < 2; i++) {
        print_string("ones in ");
        print_int(nums[i]);
        print_string(":");
        print_char(' ');
        print_int(ones_loop(nums[i]));
        print_char(' ');
        print_int(ones_clear(nums[i]));
        print_char(' ');
        print_int(ones_table(nums[i]));
        print_char('\n');
    }
    return 0;
}
