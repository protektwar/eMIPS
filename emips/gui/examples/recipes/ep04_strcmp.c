/* MIPS by Example in eMIPS, episode 4 — Comparing two strings (the C
 * version of ep04_strcmp.asm). */
#include "emips.h"

int my_strcmp(const char *a, const char *b)
{
    while (*a != '\0' && *a == *b) {
        a++;
        b++;
    }
    return (unsigned char)*a - (unsigned char)*b;
}

static const char *pairs[] = {"apple", "apply", "pear", "peach",
                              "MIPS", "MIPS", "Zebra", "apple", 0};

int main(void)
{
    for (const char **p = pairs; p[0] != 0; p += 2) {
        int r = my_strcmp(p[0], p[1]);
        print_string(p[0]);
        print_string(r < 0 ? " < " : r > 0 ? " > " : " = ");
        print_string(p[1]);
        print_char('\n');
    }
    return 0;
}
