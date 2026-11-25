/* MIPS by Example in eMIPS, episode 14, the challenge solved (the C version
 * of ep14_sort_challenge.asm): the comparison passed as a function pointer. */
#include "emips.h"

static const char *names[] = {"pear", "apple", "MIPS", "zebra", "banana", "Apple"};
static const char *copy[] = {"pear", "apple", "MIPS", "zebra", "banana", "Apple"};

static int fold(int c)
{
    return (unsigned)(c - 'A') < 26 ? c + 32 : c;
}

static int my_strcasecmp(const char *a, const char *b)
{
    int x, y;
    do {
        x = fold((unsigned char)*a++);
        y = fold((unsigned char)*b++);
    } while (x == y && x != '\0');
    return x - y;
}

static int bylen(const char *a, const char *b)
{
    int n = 0;
    while (*a++ != '\0')
        n++;
    while (*b++ != '\0')
        n--;
    return n;
}

void isort(const char **a, int n, int (*cmp)(const char *, const char *))
{
    for (int i = 1; i < n; i++) {
        const char *key = a[i];
        int j = i - 1;
        while (j >= 0 && cmp(a[j], key) > 0) {
            a[j + 1] = a[j];
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
    isort(names, 6, my_strcasecmp);
    print_string("ignoring case:");
    show(names, 6);
    isort(copy, 6, bylen);
    print_string("by length:    ");
    show(copy, 6);
    return 0;
}
