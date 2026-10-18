/* Lab: recursion and the stack (python3 -m emips cc c_recursion.c -O 0).
 * fact(n) calls itself: watch the call stack (bt) and $sp. */
#include "emips.h"

int fact(int n)
{
    if (n <= 1)
        return 1;
    return n * fact(n - 1);
}

int main(void)
{
    print_int(fact(6));
    print_char('\n');
    return 0;
}
