/* MIPS by Example in eMIPS, episode 43 — A hash table: counting words (the
 * C version of ep43_hashtable.asm).  16 slots, h = h * 31 + c, linear
 * probing on a collision. */
#include "emips.h"

#define SLOTS 16
static char text[] = "the cat and the dog and the bird saw the cat";
static char *keys[SLOTS];               /* 0 = empty */
static int counts[SLOTS];

static unsigned hash(const char *s)
{
    unsigned h = 0;
    while (*s)
        h = h * 31 + (unsigned char)*s++;
    return h;
}

static int streq(const char *a, const char *b)
{
    while (*a == *b) {
        if (*a == 0)
            return 1;
        a++, b++;
    }
    return 0;
}

/* one more for this word: its slot, or an empty one when it is new */
static void count(char *word)
{
    unsigned slot = hash(word) & (SLOTS - 1);
    while (keys[slot] && !streq(word, keys[slot]))
        slot = (slot + 1) & (SLOTS - 1);   /* a collision: the next slot */
    if (!keys[slot])
        keys[slot] = word;
    counts[slot]++;
}

int main(void)
{
    char *s = text;
    while (*s) {                        /* word by word: a space becomes its end */
        char *word = s;
        while (*s && *s != ' ')
            s++;
        if (*s)
            *s++ = 0;
        count(word);
    }
    int words = 0;
    for (int i = 0; i < SLOTS; i++)
        if (keys[i]) {
            words++;
            print_string("slot ");
            print_int(i);
            print_string(": ");
            print_string(keys[i]);
            print_char(' ');
            print_int(counts[i]);
            print_char('\n');
        }
    print_string("different words: ");
    print_int(words);
    print_char('\n');
    return 0;
}
