/* MIPS by Example in eMIPS, episode 6, the challenge solved (the C version
 * of ep06_case_challenge.asm). */
#include "emips.h"

static char buf[64];

static void chomp(char *s)
{
    for (; *s != '\0'; s++)
        if (*s == '\n') {
            *s = '\0';
            return;
        }
}

static int is_letter(int c)
{
    return (unsigned)((c | 0x20) - 'a') < 26;
}

void swapcase(char *s)
{
    for (; *s != '\0'; s++)
        if (is_letter(*s))
            *s ^= 0x20;                  /* bit 5: A <-> a */
}

void title(char *s)
{
    int start = 1;                       /* the next letter begins a word */
    for (; *s != '\0'; s++) {
        if (*s == ' ') {
            start = 1;
        } else if (is_letter(*s)) {
            *s |= 0x20;                  /* small ... */
            if (start)
                *s ^= 0x20;              /* ... or a capital at a word's start */
            start = 0;
        }
    }
}

int main(void)
{
    print_string("Your name? ");
    read_string(buf, 64);
    chomp(buf);
    swapcase(buf);
    print_string("Swapped: ");
    print_string(buf);
    title(buf);
    print_string("\nTitle: ");
    print_string(buf);
    print_char('\n');
    return 0;
}
