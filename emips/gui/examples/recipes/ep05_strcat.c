/* MIPS by Example in eMIPS, episode 5 — Copy, concatenate, and the
 * overflow (the C version of ep05_strcat.asm). */
#include "emips.h"

static char buf[16];                       /* 15 characters + the 0 */
static int visits = 7;

void my_strcpy(char *dst, const char *src)
{
    while ((*dst++ = *src++) != '\0')
        ;
}

void my_strcat(char *dst, const char *src)
{
    while (*dst != '\0')
        dst++;
    my_strcpy(dst, src);
}

int main(void)
{
    my_strcpy(buf, "Hello, ");
    my_strcat(buf, "world!");
    print_string(buf);
    print_string("\nvisits = ");
    print_int(visits);
    return 0;
}
