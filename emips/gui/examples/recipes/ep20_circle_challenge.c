/* MIPS by Example in eMIPS, episode 20, the challenge solved (the C version
 * of ep20_circle_challenge.asm). */
#include "emips.h"

static const double temps[3] = {37.0, -40.0, 100.0};
static const int quads[6] = {1, -3, 2, 1, 2, 5};

double fahrenheit(double c)
{
    return c * 9.0 / 5.0 + 32.0;           /* the fraction stays */
}

static double sqrt_d(double x)            /* sqrt.d: no C library */
{
    double r;
    __asm__ ("sqrt.d %0, %1" : "=f"(r) : "f"(x));
    return r;
}

void roots(int a, int b, int c)
{
    double da = a, db = b, dc = c, disc;

    print_string("roots of ");
    print_int(a);
    print_char(' ');
    print_int(b);
    print_char(' ');
    print_int(c);
    print_string(":");
    disc = db * db - 4 * (da * dc);
    if (disc < 0) {
        print_string(" none\n");
        return;
    }
    disc = sqrt_d(disc);
    print_char(' ');
    print_double((-db + disc) / (2 * da));
    print_char(' ');
    print_double((-db - disc) / (2 * da));
    print_char('\n');
}

int main(void)
{
    for (int i = 0; i < 3; i++) {
        print_double(temps[i]);
        print_string(" C = ");
        print_double(fahrenheit(temps[i]));
        print_string(" F\n");
    }
    for (int i = 0; i < 6; i += 3)
        roots(quads[i], quads[i + 1], quads[i + 2]);
    return 0;
}
