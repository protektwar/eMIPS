/* MIPS by Example in eMIPS, episode 7 — Reverse in place and palindromes
 * (the C version of ep07_reverse.asm). */
#include "emips.h"

static char text[] = "stressed";
static const char *words[] = {"racecar", "level", "MIPS", 0};

static char *last(char *s)                 /* the last character */
{
    while (s[1] != '\0')
        s++;
    return s;
}

void reverse(char *s)
{
    char *left = s, *right = last(s);
    while (left < right) {
        char c = *left;
        *left++ = *right;
        *right-- = c;
    }
}

int palindrome(const char *s)
{
    const char *left = s, *right = last((char *)s);
    while (left < right)
        if (*left++ != *right--)
            return 0;
    return 1;
}

int main(void)
{
    reverse(text);
    print_string(text);
    print_char('\n');
    for (const char **w = words; *w != 0; w++) {
        print_string(*w);
        print_string(palindrome(*w) ? ": a palindrome\n" : ": not a palindrome\n");
    }
    return 0;
}
