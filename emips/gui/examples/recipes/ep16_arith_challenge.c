/* MIPS by Example in eMIPS, episode 16, the challenge solved (the C version
 * of ep16_arith_challenge.asm). */
#include "emips.h"

static const unsigned seconds[2] = {3725, 86399};
static const int temps[3] = {100, -40, 37};

static void two2(unsigned n)              /* ':' and two digits */
{
    print_char(':');
    print_char('0' + n / 10);
    print_char('0' + n % 10);
}

void hms(unsigned s)
{
    print_int(s / 3600);
    s %= 3600;
    two2(s / 60);
    two2(s % 60);
    print_char('\n');
}

int fahrenheit(int c)
{
    return c * 9 / 5 + 32;                /* multiply first, then divide */
}

int main(void)
{
    for (int i = 0; i < 2; i++) {
        print_int(seconds[i]);
        print_string(" s = ");
        hms(seconds[i]);
    }
    for (int i = 0; i < 3; i++) {
        print_int(temps[i]);
        print_string(" C = ");
        print_int(fahrenheit(temps[i]));
        print_string(" F\n");
    }
    return 0;
}
