/* MIPS by Example in eMIPS, episode 45 — A tiny text adventure (the C version
 * of ep45_adventure.asm): the world is a table of rooms, each with its exits
 * north, south, east and west (-1: a wall); the garden needs the key. */
#include "emips.h"

enum { HALL, LIBRARY, KITCHEN, GARDEN };

struct room {
    const char *name, *desc;
    int exits[4];                       /* north, south, east, west */
};

static const struct room rooms[] = {
    {"hall", "Doors lead north, east and south.", {LIBRARY, GARDEN, KITCHEN, -1}},
    {"library", "Shelves of old manuals; the door is south.", {-1, HALL, -1, -1}},
    {"kitchen", "It smells of coffee; the door is west.", {-1, -1, -1, HALL}},
    {"garden", "Sunlight at last!", {HALL, -1, -1, -1}},
};
static const char dirs[] = "nsew";      /* the commands that move, in exits order */

static void describe(int room, int keyroom)
{
    print_string("You are in the ");
    print_string(rooms[room].name);
    print_string(". ");
    print_string(rooms[room].desc);
    print_char('\n');
    if (keyroom == room)
        print_string("There is a key here.\n");
}

int main(void)
{
    char line[32];
    int here = HALL, key = LIBRARY;     /* key: the room it lies in (-1: we have it) */
    describe(here, key);
    for (;;) {
        print_string("> ");
        read_string(line, sizeof line);
        char c = line[0];
        if (c == 'q') {
            print_string("Bye.\n");
            return 0;
        }
        if (c == 't') {
            if (key == here) {
                key = -1;
                print_string("You take the key.\n");
            } else
                print_string("There is no key here.\n");
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
                if (key >= 0) {
                    print_string("The door to the garden is locked.\n");
                    continue;
                }
                print_string("You open the door with the key.\n");
            }
            here = to;
        }
        describe(here, key);
        if (here == GARDEN) {
            print_string("You win!\n");
            return 0;
        }
    }
}
