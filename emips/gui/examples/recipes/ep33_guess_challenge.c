/* MIPS by Example in eMIPS, episode 33's challenge, solved (the C version
 * of ep33_guess_challenge.asm): at most 7 guesses a game. */
#include "emips.h"

static void play(void)
{
    int secret = random_int_range(0, 100) + 1;   /* 1 ... 100 */
    int guesses = 0, status;
    for (;;) {
        int g = input_dialog_int("Guess my number, 1 to 100:", &status);
        if (status == -2) {                       /* Cancel: give up */
            print_string("gave up: ");
            print_int(secret);
            print_char('\n');
            message_dialog_int("It was ", secret);
            return;
        }
        if (status != 0) {
            message_dialog("Not a number.", 2);
            continue;
        }
        guesses++;
        print_int(g);
        if (g == secret) {
            print_string(" right, guesses: ");
            print_int(guesses);
            print_char('\n');
            message_dialog_int("Got it! Guesses: ", guesses);
            return;
        }
        if (g < secret) {
            message_dialog("Higher.", 1);
            print_string(" higher\n");
        } else {
            message_dialog("Lower.", 1);
            print_string(" lower\n");
        }
        if (guesses == 7) {                       /* the seventh wrong guess */
            print_string("out of guesses: ");
            print_int(secret);
            print_char('\n');
            message_dialog_int("Out of guesses. It was ", secret);
            return;
        }
    }
}

int main(void)
{
    set_seed(0, time_ms());                       /* eMIPS: the simulated clock */
    do
        play();
    while (confirm_dialog("Play again?") == 0);
    return 0;
}
