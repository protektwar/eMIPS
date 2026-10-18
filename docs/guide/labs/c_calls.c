/* Lab: the MIPS o32 calling convention (python3 -m emips cc c_calls.c -O 0).
 * mix() takes six arguments: where does each one arrive? */
#include "emips.h"

int mix(int a, int b, int c, int d, int e, int f)
{
    return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f;
}

int main(void)
{
    int r = mix(1, 2, 3, 4, 5, 6);
    print_int(r);
    print_char('\n');
    return 0;
}
