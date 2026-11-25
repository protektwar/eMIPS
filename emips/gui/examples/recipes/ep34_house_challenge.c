/* MIPS by Example in eMIPS, episode 34's challenge, solved (the C version
 * of ep34_house_challenge.asm): a sun, a filled circle. */
#include "emips.h"

#define SCREEN ((unsigned *)0x10008000)
static int pixels;                        /* how many pixels were written */

void hline(int x0, int x1, int y, unsigned colour)
{
    unsigned *p = SCREEN + y * 64 + x0;   /* pointer arithmetic: words, not bytes */
    for (; x0 <= x1; x0++) {
        *p++ = colour;
        pixels++;
    }
}

void rect(int x0, int y0, int x1, int y1, unsigned colour)
{
    for (int y = y0; y <= y1; y++)
        hline(x0, x1, y, colour);
}

void roof(int xc, int ytop, int rows, unsigned colour)
{
    for (int k = 0; k < rows; k++)       /* one wider each side, one row down */
        hline(xc - k, xc + k, ytop + k, colour);
}

void disc(int xc, int yc, int r, unsigned colour)
{
    for (int dy = -r; dy <= r; dy++) {
        int w = r;                         /* the widest w with w*w + dy*dy <= r*r */
        while (w * w > r * r - dy * dy)
            w--;
        hline(xc - w, xc + w, yc + dy, colour);
    }
}

int main(void)
{
    rect(0, 0, 63, 63, 0x0080c0ff);       /* the sky */
    disc(54, 9, 6, 0x00ffe040);           /* the sun */
    rect(0, 52, 63, 63, 0x0040a040);      /* the grass */
    rect(40, 14, 44, 24, 0x00a04040);     /* the chimney, before the roof */
    roof(31, 11, 19, 0x00c03020);
    rect(14, 30, 48, 51, 0x00e0c080);     /* the walls */
    rect(27, 40, 35, 51, 0x00804020);     /* the door */
    rect(18, 34, 24, 40, 0x00ffff80);     /* two windows */
    rect(38, 34, 44, 40, 0x00ffff80);
    print_string("pixels written: ");
    print_int(pixels);
    print_char('\n');
    return 0;
}
/* eMIPS: --display 256x256:4@gp */
