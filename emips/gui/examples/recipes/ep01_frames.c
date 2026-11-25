/* MIPS by Example in eMIPS, episode 1 — Functions done right: the stack
 * frame (the C version of ep01_frames.asm).  At -O 0 GCC gives every
 * function the same frame: $ra, $fp, the frame pointer set, locals at
 * fixed offsets from it. */
#include "emips.h"

static const char *words[] = {"MIPS", "eMIPS", "assembly", 0};

int my_strlen(const char *s)
{
    int n = 0;
    while (s[n] != '\0')
        n++;
    return n;
}

void describe(const char *s)
{
    int n = my_strlen(s);                    /* a local variable */
    print_string(s);
    print_string(" has ");
    print_int(n);
    print_string(" characters\n");
}

int main(void)
{
    for (const char **w = words; *w != 0; w++)
        describe(*w);
    return 0;
}
