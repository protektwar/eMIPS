/* MIPS by Example in eMIPS, episode 16 — Arithmetic: the four operations,
 * shifts, area and perimeter (the C version of ep16_arith.asm). */
#include "emips.h"

static void print_eq(int a, const char *op, int b, int result)
{
    print_int(a);
    print_string(op);
    print_int(b);
    print_string(" = ");
    print_int(result);
    print_char('\n');
}

int main(void)
{
    int a, b;

    print_string("Two numbers:\n");
    a = read_int();
    b = read_int();
    print_eq(a, " + ", b, a + b);
    print_eq(a, " - ", b, a - b);
    print_eq(a, " * ", b, a * b);
    print_eq(a, " / ", b, a / b);         /* rounds toward 0 */
    print_eq(a, " % ", b, a % b);
    print_eq(a, " << ", 2, a << 2);       /* times 4 */
    print_eq(-a, " >> ", 1, -a >> 1);     /* half, rounded down: GCC's sra */
    print_string("area ");
    print_int(a * b);
    print_string(", perimeter ");
    print_int(2 * (a + b));
    print_char('\n');
    return 0;
}
