/* MIPS by Example in eMIPS, episode 23, the challenge solved (the C version
 * of ep23_bsearch_challenge.asm). */
#include "emips.h"

static const int reps[8] = {1, 3, 3, 3, 5, 8, 8, 9};
static const int sqnums[3] = {144, 1000, 2};

int first(const int *a, int n, int x)     /* the first index of x, or -1 */
{
    int lo = 0, hi = n - 1, found = -1;
    while (lo <= hi) {
        int mid = (unsigned)(lo + hi) / 2;
        if (a[mid] < x)
            lo = mid + 1;
        else {
            if (a[mid] == x)
                found = mid;              /* remember it, look further left */
            hi = mid - 1;
        }
    }
    return found;
}

int isqrt(int n)                          /* the largest r with r * r <= n */
{
    int lo = 0, hi = n, r = 0;
    while (lo <= hi) {
        int mid = (unsigned)(lo + hi) / 2;
        if (mid * mid > n)
            hi = mid - 1;
        else {
            r = mid;
            lo = mid + 1;
        }
    }
    return r;
}

int main(void)
{
    int f3 = first(reps, 8, 3), f8 = first(reps, 8, 8);

    print_string("first 3 at ");
    print_int(f3);
    print_string(", first 8 at ");
    print_int(f8);
    print_char('\n');
    for (int i = 0; i < 3; i++) {
        print_string("isqrt(");
        print_int(sqnums[i]);
        print_string(") = ");
        print_int(isqrt(sqnums[i]));
        print_char('\n');
    }
    return 0;
}
