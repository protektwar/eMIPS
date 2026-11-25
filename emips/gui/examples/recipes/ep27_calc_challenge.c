/* MIPS by Example in eMIPS, episode 27, the challenge solved (the C version
 * of ep27_calc_challenge.asm). */
#include "emips.h"

static const int nums[5] = {3, 1, 4, 1, 5};

static int add2(int a, int b) { return a + b; }
static int mul2(int a, int b) { return a * b; }
static int max2(int a, int b) { return a >= b ? a : b; }

struct job {
    const char *name;
    int (*fn)(int, int);                  /* a function pointer */
    int start;
};

static const struct job jobs[3] = {
    {"sum ", add2, 0}, {"product ", mul2, 1}, {"largest ", max2, -2147483647 - 1},
};

int reduce(const int *a, int n, int (*fn)(int, int), int value)
{
    for (int i = 0; i < n; i++)
        value = fn(value, a[i]);          /* call whatever we were given */
    return value;
}

int main(void)
{
    for (int i = 0; i < 3; i++) {
        print_string(jobs[i].name);
        print_int(reduce(nums, 5, jobs[i].fn, jobs[i].start));
        print_char('\n');
    }
    return 0;
}
