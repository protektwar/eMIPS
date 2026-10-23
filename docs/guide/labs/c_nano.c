/* Lab: the same C for MIPS32 and for nanoMIPS (python3 -m emips cc
 * c_nano.c -O 2 and cc c_nano.c -O 2 --arch nanomips; prebuilt:
 * c_nano_32.elf, c_nano.elf).  It sorts a table of records by key and
 * prints the keys and a checksum of the values: compare the code. */
#include "emips.h"

struct record {
    int key;
    short value;
    unsigned char flags;
};

struct record table[8] = {
    {42, 7, 1}, {17, -3, 0}, {99, 12, 1}, {5, 30, 0},
    {64, -8, 1}, {23, 1, 0}, {81, 19, 1}, {10, 4, 0},
};

__attribute__((noinline)) void sort(struct record *r, int n)
{
    for (int i = 1; i < n; i++) {                 /* insertion sort */
        struct record x = r[i];
        int j = i - 1;
        while (j >= 0 && r[j].key > x.key) {
            r[j + 1] = r[j];
            j--;
        }
        r[j + 1] = x;
    }
}

__attribute__((noinline)) int checksum(const struct record *r, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
        if (r[i].flags)
            sum = sum * 31 + r[i].value;
    return sum;
}

int main(void)
{
    sort(table, 8);
    for (int i = 0; i < 8; i++) {
        print_int(table[i].key);
        print_char(' ');
    }
    print_char('\n');
    print_int(checksum(table, 8));
    print_char('\n');
    return 0;
}
