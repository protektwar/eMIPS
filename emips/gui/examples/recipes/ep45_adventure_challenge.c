/* MIPS by Example in eMIPS, episode 45 — the challenge, solved: a lamp, a
 * dark cellar and an inventory (the C version of ep45_adventure_challenge.asm):
 * six exits a room, an item table (the room each lies in, -1: carried). */
#include "emips.h"

enum { HALL, LIBRARY, KITCHEN, GARDEN, CELLAR };
enum { LAMP, KEY, ITEMS };

struct room {
    const char *name, *desc;
    int exits[6];                       /* north, south, east, west, up, down */
};

static const struct room rooms[] = {
    {"hall", "Doors lead north, east and south.", {LIBRARY, GARDEN, KITCHEN, -1, -1, -1}},
    {"library", "Shelves of old manuals; the door is south.", {-1, HALL, -1, -1, -1, -1}},
    {"kitchen", "It smells of coffee; the door is west, stairs go down.",
     {-1, -1, -1, HALL, -1, CELLAR}},
    {"garden", "Sunlight at last!", {HALL, -1, -1, -1, -1, -1}},
    {"cellar", "Dusty barrels; the stairs go up.", {-1, -1, -1, -1, KITCHEN, -1}},
};
static int items[ITEMS] = {LIBRARY, CELLAR};   /* where each lies (-1: carried) */
static const char *inames[ITEMS] = {"lamp", "key"};
static const char dirs[] = "nsewud";

static int dark_here(int room)
{
    return room == CELLAR && items[LAMP] >= 0;
}

static void describe(int room)
{
    if (dark_here(room)) {
        print_string("It is too dark to see.\n");
        return;
    }
    print_string("You are in the ");
    print_string(rooms[room].name);
    print_string(". ");
    print_string(rooms[room].desc);
    print_char('\n');
    for (int i = 0; i < ITEMS; i++)
        if (items[i] == room) {
            print_string("There is a ");
            print_string(inames[i]);
            print_string(" here.\n");
        }
}

static void take(int room)
{
    int n = 0;
    if (!dark_here(room))
        for (int i = 0; i < ITEMS; i++)
            if (items[i] == room) {
                items[i] = -1;
                n++;
                print_string("You take the ");
                print_string(inames[i]);
                print_string(".\n");
            }
    if (n == 0)
        print_string("There is nothing here.\n");
}

static void inventory(void)
{
    int n = 0;
    print_string("You carry:");
    for (int i = 0; i < ITEMS; i++)
        if (items[i] < 0) {
            n++;
            print_char(' ');
            print_string(inames[i]);
        }
    if (n == 0)
        print_string(" nothing");
    print_char('\n');
}

int main(void)
{
    char line[32];
    int here = HALL;
    describe(here);
    for (;;) {
        print_string("> ");
        read_string(line, sizeof line);
        char c = line[0];
        if (c == 'q') {
            print_string("Bye.\n");
            return 0;
        }
        if (c == 't') {
            take(here);
            continue;
        }
        if (c == 'i') {
            inventory();
            continue;
        }
        if (c != 'l') {
            int d = 0;
            while (dirs[d] && dirs[d] != c)
                d++;
            if (!dirs[d]) {
                print_string("I don't understand.\n");
                continue;
            }
            int to = rooms[here].exits[d];
            if (to < 0) {
                print_string("You can't go that way.\n");
                continue;
            }
            if (to == GARDEN) {
                if (items[KEY] >= 0) {
                    print_string("The door to the garden is locked.\n");
                    continue;
                }
                print_string("You open the door with the key.\n");
            }
            here = to;
        }
        describe(here);
        if (here == GARDEN) {
            print_string("You win!\n");
            return 0;
        }
    }
}
