/* MIPS by Example in eMIPS, episode 6 — Reading a line; upper and lower
 * case (the C version of ep06_case.asm). */
#include "emips.h"

static char buf[64];                     /* up to 63 characters + the 0 */

void chomp(char *s)                      /* cut the line at its newline */
{
    for (; *s != '\0'; s++)
        if (*s == '\n') {
            *s = '\0';
            return;
        }
}

void upper(char *s)
{
    for (; *s != '\0'; s++)
        if ((unsigned)(*s - 'a') < 26)
            *s -= 32;
}

void lower(char *s)
{
    for (; *s != '\0'; s++)
        if ((unsigned)(*s - 'A') < 26)
            *s += 32;
}

int main(void)
{
    print_string("Your name? ");
    read_string(buf, 64);
    chomp(buf);
    print_string("Hello, ");
    print_string(buf);
    print_string("!\n");
    upper(buf);
    print_string("UPPER: ");
    print_string(buf);
    lower(buf);
    print_string("\nlower: ");
    print_string(buf);
    print_char('\n');
    return 0;
}
