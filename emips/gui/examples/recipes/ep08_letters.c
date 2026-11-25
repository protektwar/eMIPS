/* MIPS by Example in eMIPS, episode 8 — Counting letters: the most frequent
 * character (the C version of ep08_letters.asm). */
#include "emips.h"

static const char text[] = "The quick brown fox jumps over the lazy dog";
static int counts[26];

void count(const char *s, int *c)
{
    for (; *s != '\0'; s++) {
        unsigned i = (unsigned)((*s | 0x20) - 'a');   /* capitals as small letters */
        if (i < 26)
            c[i]++;
    }
}

/* the largest count: its letter (returned), the count in *n */
char most(const int *c, int *n)
{
    char best = 'a';
    *n = c[0];
    for (int i = 1; i < 26; i++)
        if (c[i] > *n) {
            best = 'a' + i;
            *n = c[i];
        }
    return best;
}

int main(void)
{
    int n;
    char m;

    count(text, counts);
    for (int i = 0; i < 26; i++)
        if (counts[i] >= 2) {
            print_char('a' + i);
            print_char(' ');
            print_int(counts[i]);
            print_char('\n');
        }
    m = most(counts, &n);
    print_string("most frequent: ");
    print_char(m);
    print_char(',');
    print_char(' ');
    print_int(n);
    print_string(" times\n");
    return 0;
}
