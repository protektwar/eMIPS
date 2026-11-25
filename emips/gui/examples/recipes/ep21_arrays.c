/* MIPS by Example in eMIPS, episode 21 — Arrays: sum, search, reverse, two
 * score arrays into grades (the C version of ep21_arrays.asm). */
#include "emips.h"

static int midterm[6] = {78, 92, 65, 88, 71, 95};
static const int final[6] = {84, 88, 59, 92, 75, 99};
static char letters[6];                   /* one grade letter a student */

int sum(const int *a, int n)
{
    int s = 0;
    for (int i = 0; i < n; i++)
        s += a[i];
    return s;
}

int find(const int *a, int n, int x)      /* x's index, or -1 */
{
    for (int i = 0; i < n; i++)
        if (a[i] == x)
            return i;
    return -1;
}

static void show_find(int x)
{
    print_int(x);
    print_string(" is at index ");
    print_int(find(midterm, 6, x));
    print_char('\n');
}

char letter(unsigned avg)
{
    if (avg >= 90) return 'A';
    if (avg >= 80) return 'B';
    if (avg >= 70) return 'C';
    if (avg >= 60) return 'D';
    return 'F';
}

void grades(const int *m, const int *f, char *out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = letter((unsigned)(m[i] + f[i]) / 2);
}

void reverse(int *a, int n)               /* swap the ends, n / 2 times */
{
    for (int i = 0, j = n - 1; i < j; i++, j--) {
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

static void print_array(const char *text, const int *a, int n)
{
    print_string(text);
    for (int i = 0; i < n; i++) {
        print_char(' ');
        print_int(a[i]);
    }
    print_char('\n');
}

int main(void)
{
    int s;

    print_array("midterm:", midterm, 6);
    s = sum(midterm, 6);
    print_string("sum ");
    print_int(s);
    print_string(", average ");
    print_int((unsigned)s / 6);
    print_char('\n');
    show_find(88);
    show_find(100);
    grades(midterm, final, letters, 6);
    print_string("grades:");
    for (int i = 0; i < 6; i++) {
        print_char(' ');
        print_char(letters[i]);
    }
    print_char('\n');
    reverse(midterm, 6);
    print_array("reversed:", midterm, 6);
    return 0;
}
