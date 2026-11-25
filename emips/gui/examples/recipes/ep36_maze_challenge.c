/* MIPS by Example in eMIPS, episode 36's challenge, solved (the C version
 * of ep36_maze_challenge.asm): a robot by the right-hand rule, a trail. */
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

static const int dx[4] = { 0, 1, 0, -1 }, dy[4] = { -1, 0, 1, 0 };  /* N E S W */
static const int turns[4] = { 1, 0, 3, 2 };   /* right, straight, left, back */

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
    int d = 1;                            /* facing east */
    for (;;) {
        int t = 0, nd;
        while (nd = (d + turns[t]) & 3, maze[(y + dy[nd]) * ROW + x + dx[nd]] == '#')
            t++;                          /* a wall: the next turn */
        char ch = maze[(y + dy[nd]) * ROW + x + dx[nd]];
        plot(x, y, 0x00806010);           /* the trail */
        d = nd;
        x += dx[d];
        y += dy[d];
        moves++;
        plot(x, y, 0x00ffd020);
        if (ch == 'E') {
            print_string("robot out in ");
            print_int(moves);
            print_string(" moves\n");
            return 0;
        }
    }
}
/* eMIPS: --display 256x256:16@gp --file ep36_maze.txt */
