/* MIPS by Example in eMIPS, episode 30, the challenge solved (the C version
 * of ep30_savings_challenge.asm). */
#include "emips.h"

static unsigned fees;                     /* every fee paid, in cents */

unsigned fee(unsigned b)
{
    return b < 200000 ? b / 100 : 0;
}

unsigned month(unsigned b)
{
    unsigned f;
    b += 10000;
    f = fee(b);
    fees += f;                            /* a total in memory */
    return b - f;
}

unsigned year(unsigned b)
{
    for (int m = 0; m < 12; m++)
        b = month(b);
    return b + b * 5 / 100;
}

static void print_money(unsigned c)
{
    print_char('$');
    print_int(c / 100);
    print_char('.');
    print_char('0' + c % 100 / 10);
    print_char('0' + c % 10);
}

int main(void)
{
    unsigned b = 100000;
    int years = 0;

    do {                                  /* a year at a time, until 10000.00 */
        b = year(b);
        years++;
    } while (b < 1000000);
    print_string("10000.00 reached in year ");
    print_int(years);
    print_string(": ");
    print_money(b);
    print_char('\n');
    print_string("fees paid: ");
    print_money(fees);
    print_char('\n');
    return 0;
}
