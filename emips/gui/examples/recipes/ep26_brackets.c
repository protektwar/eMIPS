/* MIPS by Example in eMIPS, episode 26 — Balanced brackets with the stack
 * (the C version of ep26_brackets.asm: an array as the stack, where the
 * assembly uses the real one). */
#include "emips.h"

static const char *const tests[4] = {"(a[b]{c})", "(a[b)c]", "((x)", "x)"};

int balanced(const char *s)               /* -1, or the problem's index */
{
    char stack[64];
    int top = 0;                          /* empty */
    for (int i = 0; s[i]; i++) {
        char c = s[i];
        if (c == '(' || c == '[' || c == '{')
            stack[top++] = c;             /* push */
        else if (c == ')' || c == ']' || c == '}') {
            char want = c == ')' ? '(' : c - 2;   /* ']' - 2 = '[' */
            if (top == 0 || stack[--top] != want) /* pop, compare */
                return i;
        }
    }
    if (top != 0)                         /* some left open */
        return (int)strlen(s);
    return -1;
}

int main(void)
{
    for (int i = 0; i < 4; i++) {
        int k = balanced(tests[i]);
        print_string(tests[i]);
        if (k < 0)
            print_string(" balanced\n");
        else {
            print_string(" error at ");
            print_int(k);
            print_char('\n');
        }
    }
    return 0;
}
