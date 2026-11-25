/* MIPS by Example in eMIPS, episode 22 — Sorting numbers: bubble,
 * insertion, into a second array (the C version of ep22_sort.asm). */
#include "emips.h"

static const int nums[7] = {64, 25, 12, 22, 11, 90, 35};
static int sorted1[7], sorted2[7];        /* each sort works on a copy */

void copy(const int *src, int *dst, int n)
{
    for (int i = 0; i < n; i++)
        dst[i] = src[i];
}

int bubble(int *a, int n, int *passes)    /* -> swaps */
{
    int swaps = 0;
    *passes = 0;
    for (int pairs = n - 1; pairs > 0; pairs--) {
        int swapped = 0;
        ++*passes;
        for (int i = 0; i < pairs; i++)
            if (a[i] > a[i + 1]) {
                int t = a[i];
                a[i] = a[i + 1];
                a[i + 1] = t;
                swaps++;
                swapped = 1;
            }
        if (!swapped)                     /* no swap: sorted */
            break;
    }
    return swaps;
}

int insertion(int *a, int n)              /* -> moves */
{
    int moves = 0;
    for (int i = 1; i < n; i++) {
        int key = a[i], j = i;
        while (j > 0 && a[j - 1] > key) {
            a[j] = a[j - 1];              /* move up one place */
            j--;
            moves++;
        }
        a[j] = key;
    }
    return moves;
}

static void print_array(const char *text, const int *a)
{
    print_string(text);
    for (int i = 0; i < 7; i++) {
        print_char(' ');
        print_int(a[i]);
    }
}

int main(void)
{
    int swaps, passes, moves;

    print_array("numbers:", nums);
    print_char('\n');
    copy(nums, sorted1, 7);
    swaps = bubble(sorted1, 7, &passes);
    print_array("bubble:", sorted1);
    print_string(", ");
    print_int(swaps);
    print_string(" swaps in ");
    print_int(passes);
    print_string(" passes\n");
    copy(nums, sorted2, 7);
    moves = insertion(sorted2, 7);
    print_array("insertion:", sorted2);
    print_string(", ");
    print_int(moves);
    print_string(" moves\n");
    print_array("numbers:", nums);
    print_char('\n');
    return 0;
}
