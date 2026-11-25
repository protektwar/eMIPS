/* MIPS by Example in eMIPS, episode 20 — Floating point: a circle's area
 * with doubles (the C version of ep20_circle.asm). */
#include "emips.h"

#define PI 3.141592653589793

double area(double r)
{
    return r * r * PI;
}

double circumference(double r)
{
    return 2 * r * PI;
}

int main(void)
{
    double r, a;

    print_string("Radius:\n");
    r = read_double();
    a = area(r);
    print_string("area ");
    print_double(a);
    print_string(", rounded ");
    print_int((int)(a + 0.5));            /* (int) cuts: + 0.5 rounds a positive */
    print_string("\ncircumference ");
    print_double(circumference(r));
    print_char('\n');
    return 0;
}
