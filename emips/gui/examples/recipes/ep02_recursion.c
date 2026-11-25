/* MIPS by Example in eMIPS, episode 2 — Arguments and recursion (the C
 * version of ep02_recursion.asm: GCC passes e at 16($sp) too). */
#include "emips.h"

int square(int x)
{
    return x * x;
}

int sum_of_squares(int a, int b, int c, int d, int e)
{
    return square(a) + square(b) + square(c) + square(d) + square(e);
}

int fact(int n)
{
    if (n <= 1)
        return 1;
    return n * fact(n - 1);
}

static void print_result(const char *text, int n)
{
    print_string(text);
    print_int(n);
    print_char('\n');
}

int main(void)
{
    print_result("sum of squares = ", sum_of_squares(1, 2, 3, 4, 5));
    print_result("fact(5) = ", fact(5));
    print_result("fact(10) = ", fact(10));
    return 0;
}
