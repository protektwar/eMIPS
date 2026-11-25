/* MIPS by Example in eMIPS, episode 37 — Playing a tune (the C version of
 * ep37_tune.asm): a table of (pitch, beats), the tempo, service 33.
 * Its checks are built with -O2: the time printed counts the instructions
 * too, and at -O0 they add 2 ms. */
#include "emips.h"

#define TEMPO 400                         /* milliseconds a beat */
static const int twinkle[] = {
    60, 1, 60, 1, 67, 1, 67, 1, 69, 1, 69, 1, 67, 2,   /* C C G G A A G- */
    65, 1, 65, 1, 64, 1, 64, 1, 62, 1, 62, 1, 60, 2,   /* F F E E D D C- */
    67, 1, 67, 1, 65, 1, 65, 1, 64, 1, 64, 1, 62, 2,   /* G G F F E E D- */
    67, 1, 67, 1, 65, 1, 65, 1, 64, 1, 64, 1, 62, 2,
    60, 1, 60, 1, 67, 1, 67, 1, 69, 1, 69, 1, 67, 2,
    65, 1, 65, 1, 64, 1, 64, 1, 62, 1, 62, 1, 60, 2,
    0                                     /* the end */
};

int play(const int *tune, int tempo, int instrument)
{
    int n = 0;
    for (; tune[0] != 0; tune += 2, n++)
        midi_out_sync(tune[0], tune[1] * tempo, instrument, 100);   /* play, wait */
    return n;
}

int main(void)
{
    unsigned start = time_ms();
    int n = play(twinkle, TEMPO, 0);      /* 0: piano */
    print_string("notes: ");
    print_int(n);
    print_string("\nms: ");
    print_int(time_ms() - start);
    print_char('\n');
    return 0;
}
