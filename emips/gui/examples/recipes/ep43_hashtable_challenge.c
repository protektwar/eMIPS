/* MIPS by Example in eMIPS, episode 43 — the challenge, solved: a good hash
 * and a bad one (the C version of ep43_hashtable_challenge.asm): the same
 * words with h = h * 31 + c and with the sum of the letters (every anagram
 * in one slot); probes counts the slots looked at. */
#include "emips.h"

#define SLOTS 16
static char text1[] = "the cat and the dog and the bird saw the cat act tac god was";
static char text2[] = "the cat and the dog and the bird saw the cat act tac god was";
static char *keys[SLOTS];               /* 0 = empty */
static int counts[SLOTS];
static int sumhash;                     /* 1: the sum of the letters as the hash */
static int probes;                      /* slots looked at */

static unsigned hash(const char *s)
{
    unsigned h = 0;
    while (*s)
        h = (sumhash ? h : h * 31) + (unsigned char)*s++;
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
    for (;;) {
        probes++;
        if (!keys[slot] || streq(word, keys[slot]))
            break;
        slot = (slot + 1) & (SLOTS - 1);   /* a collision: the next slot */
    }
    if (!keys[slot])
        keys[slot] = word;
    counts[slot]++;
}

static void words(char *s, int sum, const char *label)
{
    sumhash = sum;
    probes = 0;
    for (int i = 0; i < SLOTS; i++) {   /* an empty table */
        keys[i] = 0;
        counts[i] = 0;
    }
    while (*s) {                        /* word by word: a space becomes its end */
        char *word = s;
        while (*s && *s != ' ')
            s++;
        if (*s)
            *s++ = 0;
        count(word);
    }
    print_string(label);
    for (int i = 0; i < SLOTS; i++)
        if (keys[i]) {
            print_string("slot ");
            print_int(i);
            print_string(": ");
            print_string(keys[i]);
            print_char(' ');
            print_int(counts[i]);
            print_char('\n');
        }
    print_string("probes: ");
    print_int(probes);
    print_char('\n');
}

int main(void)
{
    words(text1, 0, "h = h * 31 + c\n");
    words(text2, 1, "the sum of the letters\n");
    return 0;
}
