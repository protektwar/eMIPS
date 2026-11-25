/* MIPS by Example in eMIPS, episode 3, the challenge solved (the C version
 * of ep03_strings_challenge.asm). */
#include "emips.h"

static const char text[] = "MIPS by Example in eMIPS";

int strlen_bottom(const char *s)
{
    int n = 0;
    if (*s == '\0')
        return 0;
    do {
        s++;
        n++;
    } while (*s != '\0');
    return n;
}

int strlen_ptr(const char *s)
{
    const char *p = s;
    while (*p++ != '\0')
        ;
    return p - s - 1;
}

int count_caps(const char *s)
{
    int n = 0;
    for (; *s != '\0'; s++)
        n += (unsigned char)(*s - 'A') < 26;
    return n;
}

static void print_result(const char *label, int n)
{
    print_string(label);
    print_int(n);
    print_char('\n');
}

int main(void)
{
    print_result("strlen_bottom: ", strlen_bottom(text));
    print_result("strlen_ptr: ", strlen_ptr(text));
    print_result("strlen_bottom(\"\"): ", strlen_bottom(""));
    print_result("strlen_ptr(\"\"): ", strlen_ptr(""));
    print_result("capitals: ", count_caps(text));
    return 0;
}
