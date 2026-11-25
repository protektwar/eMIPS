/* MIPS by Example in eMIPS, episode 3 — Strings in memory: length and the
 * byte order (the C version of ep03_strings.asm). */
#include "emips.h"

static const char word[] = "MIPS";

int my_strlen(const char *s)
{
    int n = 0;
    while (s[n] != '\0')
        n++;
    return n;
}

int main(void)
{
    unsigned as_word;

    print_string(word);
    print_string(" has ");
    print_int(my_strlen(word));
    print_string(" characters\nas a word: ");
    memcpy(&as_word, word, 4);              /* the four bytes as one word */
    print_hex(as_word);
    print_char('\n');
    return 0;
}
