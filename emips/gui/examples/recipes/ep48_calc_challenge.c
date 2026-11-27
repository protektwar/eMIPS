/* MIPS by Example in eMIPS, episode 48 — the challenge, solved: % and a
 * division by zero (the C version of ep48_calc_challenge.asm): the divisor
 * is tested first, because RISC-V's div would quietly give -1. */
#include "emips.h"

static const char *inputs[] = {"17%5+1", "2*(10%4)", "7/(3-3)", "100-9%0",
                               "(1+2)*(3+4)%5", 0};
static const char *pos;                 /* the next character to read */
static int err;                         /* 1: a division by zero happened */

static int expr(void);

/* factor = number | ( expr ) | - factor */
static int factor(void)
{
    if (*pos == '(') {
        pos++;
        int v = expr();
        pos++;                          /* the ) */
        return v;
    }
    if (*pos == '-') {
        pos++;
        return -factor();
    }
    int v = 0;
    while (*pos >= '0' && *pos <= '9')
        v = v * 10 + (*pos++ - '0');
    return v;
}

/* term = factor { (* | / | %) factor } */
static int term(void)
{
    int v = factor();
    while (*pos == '*' || *pos == '/' || *pos == '%') {
        char op = *pos++;
        int r = factor();
        if (op == '*')
            v = v * r;
        else if (r == 0)
            err = 1;                    /* RISC-V would not complain */
        else
            v = op == '/' ? v / r : v % r;
    }
    return v;
}

/* expr = term { (+ | -) term } */
static int expr(void)
{
    int v = term();
    while (*pos == '+' || *pos == '-') {
        char op = *pos++;
        int r = term();
        v = op == '+' ? v + r : v - r;
    }
    return v;
}

int main(void)
{
    for (const char **e = inputs; *e; e++) {
        pos = *e;
        err = 0;
        print_string(*e);
        print_string(" = ");
        int v = expr();
        if (err)
            print_string("error: division by zero");
        else
            print_int(v);
        print_char('\n');
    }
    return 0;
}
// eMIPS: --arch rv32
