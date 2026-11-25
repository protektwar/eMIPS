/* MIPS by Example in eMIPS, episode 27 — A calculator: jump tables and
 * function pointers (the C version of ep27_calc.asm: a switch, which GCC
 * turns into a jump table). */
#include "emips.h"

static const int exprs[6][3] = {
    {12, '+', 30}, {100, '-', 58}, {6, '*', 7}, {84, '/', 2}, {47, '%', 5}, {3, '^', 4},
};

int calc(int a, int op, int b, int *ok)
{
    *ok = 1;
    switch (op) {
    case '+': return a + b;
    case '-': return a - b;
    case '*': return a * b;
    case '/': return a / b;
    case '%': return a % b;
    }
    *ok = 0;                              /* the default */
    return 0;
}

int main(void)
{
    for (int i = 0; i < 6; i++) {
        int ok, r = calc(exprs[i][0], exprs[i][1], exprs[i][2], &ok);
        print_int(exprs[i][0]);
        print_char(' ');
        print_char(exprs[i][1]);
        print_char(' ');
        print_int(exprs[i][2]);
        if (ok) {
            print_string(" = ");
            print_int(r);
            print_char('\n');
        } else
            print_string(": unknown operator\n");
    }
    return 0;
}
