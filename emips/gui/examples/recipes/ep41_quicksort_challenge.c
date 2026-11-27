/* MIPS by Example in eMIPS, episode 41 — the challenge, solved: how many
 * comparisons (the C version of ep41_quicksort_challenge.asm)?  The last
 * element as the pivot on a sorted array: n * n / 2; the median of three
 * splits it in halves. */
#include "emips.h"

#define N 10
static int mixed[N] = {34, 7, 23, 32, 5, 62, 32, 7, -1, 18};
static int sorted1[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static int sorted2[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static int median;                      /* 1: the median of three as the pivot */
static int compares;                    /* comparisons with the pivot */

static void swap(int *a, int i, int j)
{
    int t = a[i]; a[i] = a[j]; a[j] = t;
}

static int partition(int *a, int lo, int hi)
{
    if (median) {                       /* the median of a[lo], a[mid], a[hi] to a[hi] */
        int mid = (lo + hi) / 2;
        if (a[mid] < a[lo]) swap(a, lo, mid);
        if (a[hi] < a[lo]) swap(a, lo, hi);
        if (a[mid] < a[hi]) swap(a, mid, hi);
    }
    int pivot = a[hi], i = lo;
    for (int j = lo; j < hi; j++) {
        compares++;
        if (a[j] < pivot)
            swap(a, i++, j);
    }
    a[hi] = a[i];
    a[i] = pivot;
    return i;
}

static void quicksort(int *a, int lo, int hi)
{
    if (lo >= hi)
        return;
    int p = partition(a, lo, hi);
    quicksort(a, lo, p - 1);
    quicksort(a, p + 1, hi);
}

static void measure(int *a, int med, const char *label)
{
    median = med;
    compares = 0;
    quicksort(a, 0, N - 1);
    print_string(label);
    print_int(compares);
    print_string(" comparisons\n");
}

int main(void)
{
    measure(mixed, 0, "mixed,  last element:    ");
    measure(sorted1, 0, "sorted, last element:    ");
    measure(sorted2, 1, "sorted, median of three: ");
    return 0;
}
