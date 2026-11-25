/* MIPS by Example in eMIPS, episode 1, the challenge solved (the C version
 * of ep01_frames_challenge.asm: at -O 0 GCC keeps every argument in the
 * caller's home area). */
#include "emips.h"

int max2(int a, int b)
{
    return a >= b ? a : b;
}

int max3(int a, int b, int c)
{
    return max2(max2(a, b), c);
}

int max3_home(int a, int b, int c)
{
    return max2(max2(a, b), c);
}

static void print_result(const char *text, int n)
{
    print_string(text);
    print_int(n);
    print_char('\n');
}

int main(void)
{
    print_result("max3(3, 9, 5) = ", max3(3, 9, 5));
    print_result("max3_home(3, 9, 5) = ", max3_home(3, 9, 5));
    print_result("max3(-2, -7, -4) = ", max3(-2, -7, -4));
    print_result("max3_home(-2, -7, -4) = ", max3_home(-2, -7, -4));
    return 0;
}
