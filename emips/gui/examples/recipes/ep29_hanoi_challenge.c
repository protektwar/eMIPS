/* MIPS by Example in eMIPS, episode 29, the challenge solved (the C version
 * of ep29_hanoi_challenge.asm). */
#include "emips.h"

static int counts[4];                     /* moves of disks 1-4 */

void count(int n)
{
    if (n == 0)
        return;
    count(n - 1);
    counts[n - 1]++;                      /* disk n moves once here */
    count(n - 1);
}

int total(int n)                          /* T(n) = 2 T(n - 1) + 1 */
{
    return n == 0 ? 0 : 2 * total(n - 1) + 1;
}

int main(void)
{
    count(4);
    print_string("4 disks, moves per disk:");
    for (int i = 0; i < 4; i++) {
        print_char(' ');
        print_int(counts[i]);
    }
    print_char('\n');
    print_string("T(n):");
    for (int n = 1; n <= 8; n++) {
        print_char(' ');
        print_int(total(n));
    }
    print_char('\n');
    print_string("2^n - 1:");
    for (int n = 1; n <= 8; n++) {
        print_char(' ');
        print_int((1 << n) - 1);
    }
    print_char('\n');
    return 0;
}
