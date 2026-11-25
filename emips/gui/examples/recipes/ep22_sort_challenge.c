/* MIPS by Example in eMIPS, episode 22, the challenge solved (the C version
 * of ep22_sort_challenge.asm). */
#include "emips.h"

static int nums[7] = {64, 25, 12, 22, 11, 90, 35};
static const int left[4] = {3, 8, 15, 21};
static const int right[5] = {1, 9, 10, 30, 42};
static int merged[9];

int selection(int *a, int n)              /* -> swaps */
{
    int swaps = 0;
    for (int i = 0; i < n - 1; i++) {
        int m = i;
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[m])
                m = j;
        if (m != i) {                     /* already in place: no swap */
            int t = a[i];
            a[i] = a[m];
            a[m] = t;
            swaps++;
        }
    }
    return swaps;
}

void merge(const int *l, int nl, const int *r, int nr, int *out)
{
    while (nl || nr) {
        if (nr == 0 || (nl && *l <= *r)) {
            *out++ = *l++;
            nl--;
        } else {
            *out++ = *r++;
            nr--;
        }
    }
}

static void print_array(const char *text, const int *a, int n)
{
    print_string(text);
    for (int i = 0; i < n; i++) {
        print_char(' ');
        print_int(a[i]);
    }
}

int main(void)
{
    int swaps = selection(nums, 7);

    print_array("selection:", nums, 7);
    print_string(", ");
    print_int(swaps);
    print_string(" swaps\n");
    merge(left, 4, right, 5, merged);
    print_array("merged:", merged, 9);
    print_char('\n');
    return 0;
}
