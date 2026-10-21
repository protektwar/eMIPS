/* Lab: the same C for MIPS32 (the o32 ABI) and for MIPS64 (n64)
 * (python3 -m emips cc c_long.c -O 1 and cc c_long.c -O 1 --arch mips64;
 * prebuilt: c_long_32.elf, c_long_64.elf).  main prints the sizes of a
 * few types, then how many bytes a million 4 KB pages hold, computed twice. */
#include "emips.h"

struct entry {
    int  key;
    long value;
};

int pages = 1000000;

__attribute__((noinline)) long bytes_int(int n)  { return n * 4096; }        /* int x int */
__attribute__((noinline)) long bytes_long(int n) { return (long)n * 4096; }  /* long x int */

int main(void)
{
    print_int(sizeof(int));          print_char(' ');
    print_int(sizeof(long));         print_char(' ');
    print_int(sizeof(void *));       print_char(' ');
    print_int(sizeof(struct entry)); print_char('\n');
    print_long(bytes_int(pages));    print_char('\n');
    print_long(bytes_long(pages));   print_char('\n');
    return 0;
}
