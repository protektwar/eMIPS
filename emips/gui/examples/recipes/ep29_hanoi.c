/* MIPS by Example in eMIPS, episode 29 — Towers of Hanoi (the C version of
 * ep29_hanoi.asm). */
#include "emips.h"

static unsigned char pegs[3][4] = {{3, 2, 1, 0}, {0}, {0}};   /* bottom first */
static int height[3] = {3, 0, 0};
static int moves;

static void move_disk(int n, int from, int to)
{
    int d = pegs[from][--height[from]];   /* the top disk */
    pegs[from][height[from]] = 0;
    if (height[to] && pegs[to][height[to] - 1] < d) {   /* onto a smaller one? */
        pegs[from][height[from]++] = d;   /* put it back */
        print_string("illegal: a smaller disk under disk ");
        print_int(d);
        print_char('\n');
        return;
    }
    pegs[to][height[to]++] = d;
    moves++;
    print_string("disk ");
    print_int(d);
    print_string(": ");
    print_char('A' + from);
    print_string(" -> ");
    print_char('A' + to);
    print_char('\n');
    (void)n;
}

void hanoi(int n, int from, int to, int via)
{
    if (n == 0)
        return;
    hanoi(n - 1, from, via, to);          /* n - 1 out of the way */
    move_disk(n, from, to);               /* disk n */
    hanoi(n - 1, via, to, from);          /* n - 1 back on top of it */
}

int main(void)
{
    hanoi(3, 0, 2, 1);
    print_int(moves);
    print_string(" moves\n");
    return 0;
}
