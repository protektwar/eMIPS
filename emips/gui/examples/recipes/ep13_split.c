/* MIPS by Example in eMIPS, episode 13 — Splitting a line into words (the C
 * version of ep13_split.asm). */
#include "emips.h"

static char buf[64];
static char *words[16];

/* cut the line into words in place; their addresses into w (at most max)
 * -> the number of words */
int split(char *s, char **w, int max)
{
    int n = 0;

    for (;;) {
        while (*s == ' ')
            s++;
        if (*s == '\0' || *s == '\n' || n == max)
            return n;
        w[n++] = s;
        while (*s != '\0' && *s != '\n' && *s != ' ')
            s++;
        if (*s == '\0')
            return n;
        if (*s == '\n') {
            *s = '\0';
            return n;
        }
        *s++ = '\0';                     /* the space after it: its end */
    }
}

int main(void)
{
    int n;

    print_string("A line? ");
    read_string(buf, 64);
    n = split(buf, words, 16);
    print_int(n);
    print_string(" words:\n");
    for (int i = 0; i < n; i++) {
        print_int(i + 1);
        print_string(": ");
        print_string(words[i]);
        print_char('\n');
    }
    return 0;
}
