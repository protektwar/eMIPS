/* MIPS by Example in eMIPS, episode 15, the challenge solved (the C version
 * of ep15_decide_challenge.asm). */
#include "emips.h"

static int nums[3] = {42, -5, 17};
static const int years[4] = {1900, 2000, 2024, 2026};

void sort3(int *v)                        /* three compare-and-swap steps */
{
    int t;
    if (v[0] > v[1]) { t = v[0]; v[0] = v[1]; v[1] = t; }
    if (v[1] > v[2]) { t = v[1]; v[1] = v[2]; v[2] = t; }
    if (v[0] > v[1]) { t = v[0]; v[0] = v[1]; v[1] = t; }
}

int leap(int y)
{
    return y % 400 == 0 || (y % 100 != 0 && (y & 3) == 0);
}

int main(void)
{
    sort3(nums);
    print_string("sorted:");
    for (int i = 0; i < 3; i++) {
        print_char(' ');
        print_int(nums[i]);
    }
    print_char('\n');
    for (int i = 0; i < 4; i++) {
        print_int(years[i]);
        print_string(leap(years[i]) ? " is a leap year\n" : " is not a leap year\n");
    }
    return 0;
}
