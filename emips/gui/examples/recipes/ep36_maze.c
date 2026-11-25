/* MIPS by Example in eMIPS, episode 36 — A maze game: a file, the display,
 * the keyboard (the C version of ep36_maze.asm). */
#include "emips.h"

#define SCREEN ((unsigned *)0x10008000)   /* 16 x 16 words */
#define ROW 17                            /* bytes a row in the file: 16 cells, '\n' */
static char maze[16 * ROW];

static int load(void)
{
    int fd = open_file("ep36_maze.txt", 0);
    if (fd < 0)
        return -1;
    int n = read_file(fd, maze, sizeof maze);
    close_file(fd);
    return n;
}

static void plot(int x, int y, unsigned colour)
{
    SCREEN[y * 16 + x] = colour;
}

static int getkey(void)
{
    while ((MMIO_RX_CONTROL & 1) == 0)    /* a key typed? */
        ;
    return MMIO_RX_DATA;
}

int main(void)
{
    int x = 0, y = 0, moves = 0;
    if (load() < 0) {
        print_string("cannot read the maze\n");
        return 0;
    }
    for (int r = 0; r < 16; r++)          /* the maze on the display */
        for (int c = 0; c < 16; c++) {
            char ch = maze[r * ROW + c];
            plot(c, r, ch == '#' ? 0x00304080 : ch == 'E' ? 0x0040c040 : 0x00101010);
            if (ch == 'S') {
                x = c;
                y = r;
            }
        }
    plot(x, y, 0x00ffd020);
    for (;;) {
        int key = getkey(), nx = x, ny = y;
        if (key == 'q') {
            print_string("gave up\n");
            return 0;
        }
        if (key == 'w')
            ny--;
        else if (key == 's')
            ny++;
        else if (key == 'a')
            nx--;
        else if (key == 'd')
            nx++;
        else
            continue;
        char ch = maze[ny * ROW + nx];
        if (ch == '#')                    /* a wall: stay */
            continue;
        plot(x, y, 0x00101010);
        x = nx;
        y = ny;
        moves++;
        plot(x, y, 0x00ffd020);
        if (ch == 'E') {
            print_string("out in ");
            print_int(moves);
            print_string(" moves\n");
            return 0;
        }
    }
}
/* eMIPS: --display 256x256:16@gp --file ep36_maze.txt */
