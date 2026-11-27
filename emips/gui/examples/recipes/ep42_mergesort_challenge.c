/* MIPS by Example in eMIPS, episode 42 — the challenge, solved: counting
 * inversions with merge sort (the C version of ep42_mergesort_challenge.asm):
 * taking the right half's front jumps over every element left in the left
 * half — each one an inversion. */
#include "emips.h"

#define N 8
static int array[N] = {38, 27, 43, 3, 9, 82, 10, 3};
static int reversed[N] = {8, 7, 6, 5, 4, 3, 2, 1};
static int tmp[N];                      /* merge's room */
static int inversions;

/* a[lo..mid-1] and a[mid..hi-1] are sorted: one sorted a[lo..hi-1] */
static void merge(int *a, int lo, int mid, int hi)
{
    int i = lo, j = mid, k = 0;
    while (i < mid && j < hi)
        if (a[i] <= a[j])               /* equal: left first (stable) */
            tmp[k++] = a[i++];
        else {
            inversions += mid - i;      /* it jumps over mid - i of them */
            tmp[k++] = a[j++];
        }
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

static void sort_and_count(int *a)
{
    print_string("before: ");
    print_array(a, N);
    inversions = 0;
    mergesort(a, 0, N);
    print_string("after:  ");
    print_array(a, N);
    print_string("inversions: ");
    print_int(inversions);
    print_char('\n');
}

int main(void)
{
    sort_and_count(array);
    sort_and_count(reversed);           /* the most there can be: 8 * 7 / 2 */
    return 0;
}
