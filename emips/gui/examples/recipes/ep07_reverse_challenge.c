/* MIPS by Example in eMIPS, episode 7, the challenge solved (the C version
 * of ep07_reverse_challenge.asm). */
#include "emips.h"

static char sentence[] = "MIPS by Example";
static const char *phrases[] = {"Never odd or even", "No lemon, no melon",
                                "Not a palindrome", 0};

static void reverse_range(char *left, char *right)
{
    while (left < right) {
        char c = *left;
        *left++ = *right;
        *right-- = c;
    }
}

void reverse_words(char *s)
{
    for (;;) {
        while (*s == ' ')
            s++;
        if (*s == '\0')
            return;
        char *first = s;
        while (s[1] != '\0' && s[1] != ' ')
            s++;
        reverse_range(first, s);
        s++;
    }
}

static int letter(int c)                   /* a small letter, or 0 */
{
    c |= 0x20;
    return (unsigned)(c - 'a') < 26 ? c : 0;
}

int palindrome_loose(const char *s)
{
    const char *left = s, *right = s;
    while (right[1] != '\0')
        right++;
    while (left < right) {
        int a = letter((unsigned char)*left), b;
        if (!a) {
            left++;
            continue;
        }
        b = letter((unsigned char)*right);
        if (!b) {
            right--;
            continue;
        }
        if (a != b)
            return 0;
        left++;
        right--;
    }
    return 1;
}

int main(void)
{
    reverse_words(sentence);
    print_string(sentence);
    print_char('\n');
    for (const char **p = phrases; *p != 0; p++) {
        print_string(*p);
        print_string(palindrome_loose(*p) ? ": a palindrome\n" : ": not a palindrome\n");
    }
    return 0;
}
