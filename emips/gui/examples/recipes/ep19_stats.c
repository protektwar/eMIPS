/* MIPS by Example in eMIPS, episode 19 — Multiples and statistics (the C
 * version of ep19_stats.asm). */
#include "emips.h"

static const int scores[10] = {72, 85, 90, 64, 78, 95, 58, 81, 88, 74};

int sum_mult(int limit, int k)            /* k + 2k + 3k + ... below limit */
{
    int sum = 0;
    for (int i = k; i < limit; i += k)
        sum += i;
    return sum;
}

void minmax(const int *a, int n, int *min, int *max)
{
    *min = *max = a[0];
    for (int i = 1; i < n; i++) {
        if (a[i] < *min)
            *min = a[i];
        if (a[i] > *max)
            *max = a[i];
    }
}

int mean(const int *a, int n)             /* rounded to the nearest */
{
    unsigned sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    return (sum + n / 2) / n;
}

int above(const int *a, int n, int m)
{
    int count = 0;
    for (int i = 0; i < n; i++)
        if (a[i] > m)
            count++;
    return count;
}

int main(void)
{
    int min, max, m;

    print_string("multiples of 3 or 5 below 1000: ");
    print_int(sum_mult(1000, 3) + sum_mult(1000, 5) - sum_mult(1000, 15));
    print_char('\n');
    print_string("scores:");
    for (int i = 0; i < 10; i++) {
        print_char(' ');
        print_int(scores[i]);
    }
    print_char('\n');
    minmax(scores, 10, &min, &max);
    m = mean(scores, 10);
    print_string("min ");
    print_int(min);
    print_string(", max ");
    print_int(max);
    print_string(", mean ");
    print_int(m);
    print_string(", above the mean ");
    print_int(above(scores, 10, m));
    print_char('\n');
    return 0;
}
