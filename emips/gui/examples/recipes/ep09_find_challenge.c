/* MIPS by Example in eMIPS, episode 9, the challenge solved (the C version
 * of ep09_find_challenge.asm). */
#include "emips.h"

static const char text[] = "she sells sea shells by the sea shore";

static int find(const char *text, const char *needle)
{
    for (const char *start = text;; start++) {
        const char *t = start, *n = needle;
        while (*n != '\0' && *t == *n) {
            t++;
            n++;
        }
        if (*n == '\0')
            return start - text;
        if (*start == '\0')
            return -1;
    }
}

void all(const char *needle)              /* every place, searched again past each */
{
    const char *from = text;
    int p;

    print_string(needle);
    print_string(":");
    while ((p = find(from, needle)) >= 0) {
        from += p;
        print_char(' ');
        print_int(from - text);
        from++;
    }
    print_char('\n');
}

int last(const char *t, const char *needle)
{
    const char *from = t;
    int p, latest = -1;

    while ((p = find(from, needle)) >= 0) {
        from += p;
        latest = from - t;
        from++;
    }
    return latest;
}

int main(void)
{
    all("sea");
    all("ll");
    print_string("last s: ");
    print_int(last(text, "s"));
    print_char('\n');
    return 0;
}
