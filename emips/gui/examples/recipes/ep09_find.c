/* MIPS by Example in eMIPS, episode 9 — Finding a substring (the C version
 * of ep09_find.asm). */
#include "emips.h"

static const char text[] = "she sells sea shells by the sea shore";
static const char *needles[] = {"sea", "shell", "shore", "sure", 0};

/* where needle first starts in text, or -1 */
int find(const char *text, const char *needle)
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

int main(void)
{
    for (const char **p = needles; *p != 0; p++) {
        int place = find(text, *p);
        print_string(*p);
        print_string(": ");
        print_int(place);
        print_char('\n');
    }
    return 0;
}
