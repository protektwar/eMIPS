/* Lab: a bug to find with GDB (python3 -m emips gdb c_gdb.elf, then
 * gdb-multiarch -ex 'target remote :1234').  It should print
 * "9 7 5 = 21": every score is set once and never changed again... */
#include "emips.h"

struct player {
    char name[8];
    int score;
};

static struct player players[3];

static void set_name(struct player *p, const char *name)
{
    int i = 0;
    while (name[i] != 0) {
        p->name[i] = name[i];
        i++;
    }
    p->name[i] = 0;
}

int main(void)
{
    static const char *const names[3] = {"ada", "elizabeth", "grace"};
    static const int points[3] = {9, 7, 5};
    int total = 0;

    for (int k = 0; k < 3; k++) {
        players[k].score = points[k];
        set_name(&players[k], names[k]);
    }
    for (int k = 0; k < 3; k++) {
        print_int(players[k].score);
        print_char(' ');
        total += players[k].score;
    }
    print_string("= ");
    print_int(total);
    print_char('\n');
    return 0;
}
