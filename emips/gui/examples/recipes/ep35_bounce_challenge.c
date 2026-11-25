/* MIPS by Example in eMIPS, episode 35's challenge, solved (the C version
 * of ep35_bounce_challenge.asm): a frame of 4000 cycles, clear unrolled.
 * Its checks are built with -O2: at -O0 a picture takes longer than a
 * frame. */
#include "emips.h"

#define FRAMES 60
/* the display's registers: volatile, every access really happens */
static volatile unsigned *const display = (unsigned *)0xffff0020;
enum { STATUS = 1, BASE = 2, FRAME = 4, PERIOD = 5 };

static void clear(unsigned *buf)
{
    for (unsigned *end = buf + 1024; buf != end; buf += 4)
        buf[0] = buf[1] = buf[2] = buf[3] = 0x00102040;   /* four a turn */
}

static void ball(unsigned *buf, int x, int y)
{
    unsigned *p = buf + y * 32 + x;
    for (int row = 0; row < 4; row++, p += 32)
        p[0] = p[1] = p[2] = p[3] = 0x00ffa020;
}

int main(void)
{
    unsigned *draw = (unsigned *)0x10009000, *shown = (unsigned *)0x10008000;
    int x = 3, y = 10, dx = 1, dy = 1;
    display[PERIOD] = 4000;               /* a frame every 4000 cycles */
    for (int n = 0; n < FRAMES; n++) {
        clear(draw);
        ball(draw, x, y);
        display[BASE] = (unsigned)draw;   /* shown from the next blank */
        display[STATUS] = 1;              /* clear VBLANK ... */
        while (display[STATUS] == 0)      /* ... and wait for the blank */
            ;
        unsigned *t = draw;               /* swap */
        draw = shown;
        shown = t;
        x += dx;
        if (x < 0 || x > 28) {
            dx = -dx;
            x += 2 * dx;
        }
        y += dy;
        if (y < 0 || y > 28) {
            dy = -dy;
            y += 2 * dy;
        }
    }
    print_string("frames: ");
    print_int(display[FRAME]);
    print_char('\n');
    return 0;
}
/* eMIPS: --display 256x256:8@gp */
