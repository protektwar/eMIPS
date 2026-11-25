/* MIPS by Example in eMIPS, episode 2, the challenge solved (the C version
 * of ep02_recursion_challenge.asm). */
#include "emips.h"

int fib(int n)
{
    if (n <= 1)
        return n;
    return fib(n - 1) + fib(n - 2);
}

int sum6(int a, int b, int c, int d, int e, int f)
{
    return a + b + c + d + e + f;
}

int main(void)
{
    print_string("fib:");
    for (int i = 0; i <= 10; i++) {
        print_char(' ');
        print_int(fib(i));
    }
    print_char('\n');
    print_string("sum6(1, 2, 3, 4, 5, 6) = ");
    print_int(sum6(1, 2, 3, 4, 5, 6));
    print_char('\n');
    return 0;
}
