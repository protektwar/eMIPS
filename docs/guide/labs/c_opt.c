/* Lab: what the optimiser does — the same loop built with -O 0
 * (c_opt_O0.elf) and -O 2 (c_opt_O2.elf). */
#include "emips.h"

static int data[64];

int main(void)
{
    for (int i = 0; i < 64; i++)
        data[i] = i * 3;
    int sum = 0;
    for (int i = 0; i < 64; i++)
        sum += data[i];
    print_int(sum);
    print_char('\n');
    return 0;
}
