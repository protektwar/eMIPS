/* MIPS by Example in eMIPS, episode 48 — A whole program in RISC-V: a
 * calculator (the C version of ep48_calc.asm), for RV32: recursive
 * descent, one function a rule of the grammar. */
#include "emips.h"

static const char *inputs[] = {"2+3*4", "(2+3)*4", "100/7-2*3", "-5+20",
                               "(1+2)*(3+4)-(5-6)", 0};
static const char *pos;                 /* the next character to read */

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

/* term = factor { (* | /) factor } */
static int term(void)
{
    int v = factor();
    while (*pos == '*' || *pos == '/') {
        char op = *pos++;
        int r = factor();
        v = op == '*' ? v * r : v / r;
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
        print_string(*e);
        print_string(" = ");
        print_int(expr());
        print_char('\n');
    }
    return 0;
}
// eMIPS: --arch rv32
