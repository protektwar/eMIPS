/* MIPS by Example in eMIPS, episode 15 — First decisions: the largest of
 * three, even and odd (the C version of ep15_decide.asm). */
#include "emips.h"

int max3(int a, int b, int c)
{
    int best = a;
    if (b > best)
        best = b;
    if (c > best)
        best = c;
    return best;
}

int odd(int n)
{
    return n & 1;                         /* bit 0: right for negatives too */
}

static void parity(int n)
{
    print_int(n);
    print_string(" is ");
    print_string(odd(n) ? "odd\n" : "even\n");
}

int main(void)
{
    int a, b, c;

    print_string("Three numbers:\n");
    a = read_int();
    b = read_int();
    c = read_int();
    print_string("largest: ");
    print_int(max3(a, b, c));
    print_char('\n');
    parity(a);
    parity(b);
    parity(c);
    return 0;
}
