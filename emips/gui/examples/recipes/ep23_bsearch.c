/* MIPS by Example in eMIPS, episode 23 — Binary search, loop and recursive
 * (the C version of ep23_bsearch.asm). */
#include "emips.h"

static const int nums[15] = {2, 5, 8, 12, 16, 23, 38, 45, 56, 72, 91, 99, 104, 120, 150};
static const int wanted[3] = {23, 91, 7};

int bsearch(const int *a, int n, int x, int *probes)
{
    int lo = 0, hi = n - 1;
    *probes = 0;
    while (lo <= hi) {
        int mid = (unsigned)(lo + hi) / 2;
        ++*probes;
        if (a[mid] == x)
            return mid;
        if (a[mid] < x)
            lo = mid + 1;                 /* x is larger: the right half */
        else
            hi = mid - 1;                 /* x is smaller: the left half */
    }
    return -1;
}

int bsearch_r(const int *a, int lo, int hi, int x)
{
    int mid;
    if (lo > hi)
        return -1;
    mid = (unsigned)(lo + hi) / 2;
    if (a[mid] == x)
        return mid;
    if (a[mid] < x)
        return bsearch_r(a, mid + 1, hi, x);
    return bsearch_r(a, lo, mid - 1, x);
}

int main(void)
{
    for (int i = 0; i < 3; i++) {
        int probes, k = bsearch(nums, 15, wanted[i], &probes);
        print_int(wanted[i]);
        print_string(": index ");
        print_int(k);
        print_string(", probes ");
        print_int(probes);
        print_char('\n');
    }
    print_string("recursively:");
    for (int i = 0; i < 3; i++) {
        print_char(' ');
        print_int(bsearch_r(nums, 0, 14, wanted[i]));
    }
    print_char('\n');
    return 0;
}
