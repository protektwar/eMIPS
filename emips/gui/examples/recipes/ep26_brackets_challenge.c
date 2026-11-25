/* MIPS by Example in eMIPS, episode 26, the challenge solved (the C version
 * of ep26_brackets_challenge.asm). */
#include "emips.h"

static const char *const exprs[3] = {"3 4 + 2 *", "5 1 2 + 4 * + 3 -", "9 2 - 3 *"};

int postfix(const char *s)                /* single digits, + - * */
{
    int stack[32], top = 0;
    for (; *s; s++) {
        int c = *s;
        if (c == ' ')
            continue;
        if (c >= '0' && c <= '9')
            stack[top++] = c - '0';       /* push the number */
        else {
            int r = stack[--top], l = stack[--top];   /* the right one: pushed last */
            stack[top++] = c == '+' ? l + r : c == '-' ? l - r : l * r;
        }
    }
    return stack[top - 1];
}

int main(void)
{
    for (int i = 0; i < 3; i++) {
        print_string(exprs[i]);
        print_string(" = ");
        print_int(postfix(exprs[i]));
        print_char('\n');
    }
    return 0;
}
