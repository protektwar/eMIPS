/* MIPS by Example in eMIPS, episode 32 — A library of your own (the C
 * version of ep32_arrays.asm).  In a real project the arr_ functions would
 * be arrays.c, compiled once, with arrays.h declaring them for every
 * program that uses them; here they share the file. */
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

/* --- the program --- */
static int primes[] = { 2, 3, 5, 7, 11, 13 };
static int temps[] = { -3, 4, -1, 7, 2 };

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
    return 0;
}
