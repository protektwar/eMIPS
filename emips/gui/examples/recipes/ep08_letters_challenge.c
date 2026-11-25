/* MIPS by Example in eMIPS, episode 8, the challenge solved (the C version
 * of ep08_letters_challenge.asm). */
#include "emips.h"

static const char text[] = "MIPS by Example in eMIPS";
static int counts[26];

static void count(const char *s, int *c)
{
    for (; *s != '\0'; s++) {
        unsigned i = (unsigned)((*s | 0x20) - 'a');
        if (i < 26)
            c[i]++;
    }
}

static int most(const int *c)             /* the largest count */
{
    int n = c[0];
    for (int i = 1; i < 26; i++)
        if (c[i] > n)
            n = c[i];
    return n;
}

void bars(const int *c)
{
    for (int i = 0; i < 26; i++)
        if (c[i] > 0) {
            print_char('a' + i);
            print_char(' ');
            for (int k = 0; k < c[i]; k++)
                print_char('*');
            print_char('\n');
        }
}

void ties(const int *c, int n)
{
    for (int i = 0; i < 26; i++)
        if (c[i] == n) {
            print_char(' ');
            print_char('a' + i);
        }
}

int main(void)
{
    int n;

    count(text, counts);
    bars(counts);
    n = most(counts);
    print_string("most frequent, ");
    print_int(n);
    print_string(" times:");
    ties(counts, n);
    print_char('\n');
    return 0;
}
