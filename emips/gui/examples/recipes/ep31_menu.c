/* MIPS by Example in eMIPS, episode 31 — A menu-driven program with
 * dialogs (the C version of ep31_menu.asm): a table of function pointers
 * picked by the menu choice. */
#include "emips.h"

static int count, total;

static int quit(void)
{
    if (confirm_dialog("Quit?") != 0)     /* No / Cancel: go on */
        return 1;
    print_string("bye\n");
    return 0;
}

static int add(void)
{
    int status;
    int n = input_dialog_int("A number:", &status);
    if (status != 0) {
        message_dialog("Not a number.", 2);
        return 1;
    }
    total += n;
    count++;
    print_string("added ");
    print_int(n);
    print_char('\n');
    return 1;
}

static int sum(void)
{
    message_dialog_int("The sum: ", total);
    print_string("sum ");
    print_int(total);
    print_char('\n');
    return 1;
}

static int average(void)
{
    if (count == 0) {
        message_dialog("No numbers yet.", 1);
        return 1;
    }
    message_dialog_int("The average: ", total / count);
    print_string("average ");
    print_int(total / count);
    print_char('\n');
    return 1;
}

static int (*const actions[])(void) = { quit, add, sum, average };

int main(void)
{
    int status;
    for (;;) {
        unsigned choice = input_dialog_int("1 add a number\n2 the sum\n3 the average\n0 quit",
                                           &status);
        int go;
        if (status == -2)                  /* Cancel = quit */
            go = quit();
        else if (status != 0) {
            message_dialog("Not a number.", 2);
            continue;
        } else if (choice >= 4) {          /* the range check */
            message_dialog("No such choice.", 2);
            continue;
        } else
            go = actions[choice]();        /* a call through the table */
        if (!go)
            return 0;
    }
}
