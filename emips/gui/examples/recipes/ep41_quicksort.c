/* MIPS by Example in eMIPS, episode 41 — Quicksort (the C version of
 * ep41_quicksort.asm): the last element as the pivot, Lomuto's partition. */
#include "emips.h"

#define N 10
static int array[N] = {34, 7, 23, 32, 5, 62, 32, 7, -1, 18};

/* the pivot a[hi] to its sorted place; returns that place */
static int partition(int *a, int lo, int hi)
{
    int pivot = a[hi], i = lo;
    for (int j = lo; j < hi; j++)
        if (a[j] < pivot) {
            int t = a[i]; a[i] = a[j]; a[j] = t;
            i++;
        }
    a[hi] = a[i];
    a[i] = pivot;
    return i;
}

static void quicksort(int *a, int lo, int hi)
{
    if (lo >= hi)
        return;                         /* 0 or 1 element: sorted already */
    int p = partition(a, lo, hi);
    quicksort(a, lo, p - 1);
    quicksort(a, p + 1, hi);
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
    quicksort(array, 0, N - 1);
    print_string("after:  ");
    print_array(array, N);
    return 0;
}
