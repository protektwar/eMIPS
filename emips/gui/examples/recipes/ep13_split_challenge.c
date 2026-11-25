/* MIPS by Example in eMIPS, episode 13, the challenge solved (the C version
 * of ep13_split_challenge.asm). */
#include "emips.h"

static char buf[64], line[64];
static char *words[16];

static int split(char *s, char **w, int max)
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
        *s++ = '\0';
    }
}

void join(char *out, char **w, int n)    /* one space between the words */
{
    for (int i = 0; i < n; i++) {
        if (i > 0)
            *out++ = ' ';
        for (const char *p = w[i]; *p != '\0'; p++)
            *out++ = *p;
    }
    *out = '\0';
}

char *longest(char **w, int n, int *len)
{
    char *best = 0;
    *len = -1;
    for (int i = 0; i < n; i++) {
        int k = 0;
        while (w[i][k] != '\0')
            k++;
        if (k > *len) {
            best = w[i];
            *len = k;
        }
    }
    return best;
}

int main(void)
{
    int n, len;
    char *w;

    print_string("A line? ");
    read_string(buf, 64);
    n = split(buf, words, 16);
    join(line, words, n);
    print_string("joined: \"");
    print_string(line);
    print_string("\"\nlongest: ");
    w = longest(words, n, &len);
    print_string(w);
    print_string(" (");
    print_int(len);
    print_string(")\n");
    return 0;
}
