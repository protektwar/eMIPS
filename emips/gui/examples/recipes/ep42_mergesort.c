/* MIPS by Example in eMIPS, episode 42 — Merge sort (the C version of
 * ep42_mergesort.asm): halves sorted by recursion, then merged through tmp. */
#include "emips.h"

#define N 8
static int array[N] = {38, 27, 43, 3, 9, 82, 10, 3};
static int tmp[N];                      /* merge's room */

/* a[lo..mid-1] and a[mid..hi-1] are sorted: one sorted a[lo..hi-1] */
static void merge(int *a, int lo, int mid, int hi)
{
    int i = lo, j = mid, k = 0;
    while (i < mid && j < hi)
        tmp[k++] = a[i] <= a[j] ? a[i++] : a[j++];   /* equal: left first (stable) */
    while (i < mid)
        tmp[k++] = a[i++];
    while (j < hi)
        tmp[k++] = a[j++];
    for (k = 0; k < hi - lo; k++)
        a[lo + k] = tmp[k];
}

/* sorts a[lo..hi-1] */
static void mergesort(int *a, int lo, int hi)
{
    if (hi - lo < 2)
        return;                         /* 0 or 1 element: sorted already */
    int mid = (lo + hi) / 2;
    mergesort(a, lo, mid);
    mergesort(a, mid, hi);
    merge(a, lo, mid, hi);
}

static void print_array(const int *a, int n)
{
    for (int i = 0; i < n; i++) {
        print_int(a[i]);
        print_char(i < n - 1 ? ' ' : '\n');
    }
}

int main(void)
{
    print_string("before: ");
    print_array(array, N);
    mergesort(array, 0, N);
    print_string("after:  ");
    print_array(array, N);
    return 0;
}
