/* MIPS by Example in eMIPS, episode 14 — Sorting an array of strings (the C
 * version of ep14_sort.asm): an insertion sort that moves only addresses. */
#include "emips.h"

static const char *names[] = {"pear", "apple", "MIPS", "zebra", "banana", "Apple"};

static int my_strcmp(const char *a, const char *b)
{
    while (*a != '\0' && *a == *b) {
        a++;
        b++;
    }
    return (unsigned char)*a - (unsigned char)*b;
}

void isort(const char **a, int n)
{
    for (int i = 1; i < n; i++) {
        const char *key = a[i];
        int j = i - 1;
        while (j >= 0 && my_strcmp(a[j], key) > 0) {
            a[j + 1] = a[j];             /* one place up */
            j--;
        }
        a[j + 1] = key;
    }
}

static void show(const char **a, int n)
{
    for (int i = 0; i < n; i++) {
        print_char(' ');
        print_string(a[i]);
    }
    print_char('\n');
}

int main(void)
{
    print_string("before:");
    show(names, 6);
    isort(names, 6);
    print_string("after: ");
    show(names, 6);
    return 0;
}
