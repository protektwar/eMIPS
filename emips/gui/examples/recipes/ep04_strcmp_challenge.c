/* MIPS by Example in eMIPS, episode 4, the challenge solved (the C version
 * of ep04_strcmp_challenge.asm). */
#include "emips.h"

int my_strncmp(const char *a, const char *b, int n)
{
    for (; n > 0; n--, a++, b++) {
        int d = (unsigned char)*a - (unsigned char)*b;
        if (d != 0 || *a == '\0')
            return d;
    }
    return 0;
}

static int fold(int c)
{
    return (unsigned)(c - 'A') < 26 ? c + 32 : c;
}

int my_strcasecmp(const char *a, const char *b)
{
    int x, y;
    do {
        x = fold((unsigned char)*a++);
        y = fold((unsigned char)*b++);
    } while (x == y && x != '\0');
    return x - y;
}

static void print_sign(const char *text, int r)
{
    print_string(text);
    print_char(r < 0 ? '<' : r > 0 ? '>' : '=');
    print_char('\n');
}

int main(void)
{
    print_sign("strncmp(apple, apply, 4): ", my_strncmp("apple", "apply", 4));
    print_sign("strncmp(apple, apply, 5): ", my_strncmp("apple", "apply", 5));
    print_sign("strcasecmp(Zebra, apple): ", my_strcasecmp("Zebra", "apple"));
    print_sign("strcasecmp(MIPS, mips): ", my_strcasecmp("MIPS", "mips"));
    return 0;
}
