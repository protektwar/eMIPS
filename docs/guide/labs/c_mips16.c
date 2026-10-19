/* Lab: the same C as MIPS32 and as MIPS16e (python3 -m emips cc c_mips16.c -O 1
 * and cc c_mips16.c -O 1 --mips16; prebuilt: c_mips16_32.elf, c_mips16_16.elf).
 * checksum() mixes the bytes of a string; main prints the sum. */
#include "emips.h"

static const char text[] = "MIPS16e makes code smaller";

unsigned checksum(const char *s)
{
    unsigned sum = 0;
    while (*s) {
        sum = (sum << 5) + sum + (unsigned char)*s;   /* sum * 33 + c */
        s++;
    }
    return sum;
}

int main(void)
{
    print_unsigned(checksum(text));
    print_char('\n');
    return 0;
}
