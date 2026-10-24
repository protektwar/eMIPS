/* Lab 77: the same C for RV32 and RV64 (python3 -m emips cc c_rv.c -O 2
 * --arch rv32 and --arch rv64; prebuilt: c_rv_32.elf, c_rv_64.elf).
 * fnv64() is the 64-bit FNV-1a hash; main hashes a text 100 times and
 * prints the hash's two halves. */
#include "emips.h"

typedef unsigned long long u64;

static const char text[] = "the quick brown fox jumps over the lazy dog";

__attribute__((noinline)) u64 fnv64(const char *s, u64 h)
{
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 0x100000001b3ULL;            /* the FNV prime */
    }
    return h;
}

int main(void)
{
    u64 h = 0xcbf29ce484222325ULL;        /* the FNV offset basis */
    for (int k = 0; k < 100; k++)
        h = fnv64(text, h);
    print_hex((unsigned)(h >> 32));
    print_char(' ');
    print_hex((unsigned)h);
    print_char('\n');
    return 0;
}
