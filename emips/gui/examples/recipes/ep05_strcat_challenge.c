/* MIPS by Example in eMIPS, episode 5, the challenge solved (the C version
 * of ep05_strcat_challenge.asm). */
#include "emips.h"

static char buf[16];                       /* 15 characters + the 0 */
static int visits = 7;

int my_strlcat(char *dst, const char *src, int size)
{
    char *last = dst + size - 1;           /* the place left for the 0 */
    char *start = dst;
    int n;

    while (*dst != '\0')
        dst++;
    n = dst - start;
    for (; *src != '\0'; src++) {
        n++;                               /* counted, whether it fits or not */
        if (dst != last)
            *dst++ = *src;
    }
    *dst = '\0';
    return n;
}

int main(void)
{
    int wanted;

    my_strlcat(buf, "Hello, ", 16);
    wanted = my_strlcat(buf, "world, and more!", 16);
    print_string(buf);
    print_string("\nwanted = ");
    print_int(wanted);
    print_string("\nvisits = ");
    print_int(visits);
    return 0;
}
