/* MIPS by Example in eMIPS, episode 28, the challenge solved (the C version
 * of ep28_recursion_challenge.asm). */
#include "emips.h"

static char word[4] = "abc";

int power(int x, unsigned n)              /* by squaring */
{
    int h;
    if (n == 0)
        return 1;
    h = power(x, n / 2);
    h *= h;
    if (n & 1)                            /* odd: once more x */
        h *= x;
    return h;
}

static void swap(int i, int j)
{
    char t = word[i];
    word[i] = word[j];
    word[j] = t;
}

void permute(int i, int n)                /* every order of word[i .. n - 1] */
{
    if (i == n) {
        print_char(' ');
        print_string(word);
        return;
    }
    for (int j = i; j < n; j++) {
        swap(i, j);                       /* word[j] first */
        permute(i + 1, n);
        swap(i, j);                       /* back */
    }
}

int main(void)
{
    print_string("3^13 = ");
    print_int(power(3, 13));
    print_char('\n');
    print_string("permutations:");
    permute(0, 3);
    print_char('\n');
    return 0;
}
