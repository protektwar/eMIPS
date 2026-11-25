/* MIPS by Example in eMIPS, episode 28 — Recursion: from n to 1 and back,
 * all binary patterns (the C version of ep28_recursion.asm). */
#include "emips.h"

static char buf[4];                       /* n digits and the 0 after them */

static void print_num(int n)
{
    print_char(' ');
    print_int(n);
}

void updown(int n)                        /* n ... 1 ... n */
{
    print_num(n);                         /* on the way down */
    if (n == 1)
        return;
    updown(n - 1);
    print_num(n);                         /* and on the way back up */
}

void patterns(int i, int n)               /* every mix of 0 and 1 */
{
    if (i == n) {
        print_char(' ');
        print_string(buf);
        return;
    }
    buf[i] = '0';
    patterns(i + 1, n);
    buf[i] = '1';
    patterns(i + 1, n);
}

int main(void)
{
    print_string("updown(3):");
    updown(3);
    print_char('\n');
    print_string("patterns(3):");
    buf[3] = 0;
    patterns(0, 3);
    print_char('\n');
    return 0;
}
