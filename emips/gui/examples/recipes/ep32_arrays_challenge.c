/* MIPS by Example in eMIPS, episode 32's challenge, solved (the C version
 * of ep32_arrays_challenge.asm): arr_find added to the library. */
#include "emips.h"

/* --- the library (arrays.h would declare these four) --- */
int arr_sum(const int *a, int n)
{
    int s = 0;
    while (n-- > 0)
        s += *a++;
    return s;
}

int arr_max(const int *a, int n)
{
    int best = *a;
    while (--n > 0)
        if (*++a > best)
            best = *a;
    return best;
}

void arr_print(const int *a, int n)
{
    print_char('[');
    while (n > 0) {
        print_int(*a++);
        if (--n > 0)
            print_string(", ");
    }
    print_char(']');
}

void arr_reverse(int *a, int n)
{
    int *b = a + n - 1;
    for (; a < b; a++, b--) {
        int t = *a;
        *a = *b;
        *b = t;
    }
}

int arr_find(const int *a, int n, int x)
{
    for (int i = 0; i < n; i++)
        if (a[i] == x)
            return i;
    return -1;
}

/* --- the program --- */
static int primes[] = { 2, 3, 5, 7, 11, 13 };
static int temps[] = { -3, 4, -1, 7, 2 };

static void find(const int *a, int n, int x)
{
    print_int(x);
    print_string(" at ");
    print_int(arr_find(a, n, x));
    print_char('\n');
}

static void show(const int *a, int n)
{
    arr_print(a, n);
    print_string("  sum ");
    print_int(arr_sum(a, n));
    print_string("  max ");
    print_int(arr_max(a, n));
    print_char('\n');
}

int main(void)
{
    show(primes, 6);
    show(temps, 5);
    arr_reverse(temps, 5);
    print_string("reversed ");
    arr_print(temps, 5);
    print_char('\n');
    find(primes, 6, 7);
    find(primes, 6, 8);
    find(temps, 5, 4);
    return 0;
}
