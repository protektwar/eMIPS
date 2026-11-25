/* MIPS by Example in eMIPS, episode 30 — A simulation with nested calls: a
 * savings account, in cents (the C version of ep30_savings.asm). */
#include "emips.h"

unsigned fee(unsigned b)                  /* 1 % while under 2000.00 */
{
    return b < 200000 ? b / 100 : 0;
}

unsigned month(unsigned b)
{
    b += 10000;                           /* the deposit: 100.00 */
    return b - fee(b);
}

unsigned year(unsigned b)
{
    for (int m = 0; m < 12; m++)
        b = month(b);
    return b + b * 5 / 100;               /* 5 % interest */
}

static void print_money(unsigned c)
{
    print_char('$');
    print_int(c / 100);
    print_char('.');
    print_char('0' + c % 100 / 10);
    print_char('0' + c % 10);
}

void simulate(unsigned b, int years)
{
    for (int y = 1; y <= years; y++) {
        b = year(b);
        print_string("year ");
        print_int(y);
        print_string(": ");
        print_money(b);
        print_char('\n');
    }
}

int main(void)
{
    simulate(100000, 3);                  /* 1000.00, 3 years */
    return 0;
}
