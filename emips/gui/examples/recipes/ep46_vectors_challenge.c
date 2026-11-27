/* MIPS by Example in eMIPS, episode 46 — the challenge, solved: brighter
 * pixels (the C version of ep46_vectors_challenge.asm): sixteen bytes in one
 * MSA register, + 0x40 wrapping (addv.b) and saturating (adds_u.b). */
#include "emips.h"
#include <msa.h>

static unsigned char pixels[16] __attribute__((aligned(16))) = {
    0, 32, 64, 96, 128, 160, 192, 200, 208, 216, 224, 232, 240, 248, 250, 255};
static unsigned char wrapped[16] __attribute__((aligned(16)));
static unsigned char bright[16] __attribute__((aligned(16)));

static void print_bytes(const unsigned char *p)
{
    for (int i = 0; i < 16; i++) {
        print_int(p[i]);
        print_char(i < 15 ? ' ' : '\n');
    }
}

int main(void)
{
    v16u8 px = (v16u8)__builtin_msa_ld_b(pixels, 0);
    v16u8 add = (v16u8)__builtin_msa_ldi_b(0x40);
    __builtin_msa_st_b((v16i8)__builtin_msa_addv_b((v16i8)px, (v16i8)add), wrapped, 0);
    __builtin_msa_st_b((v16i8)__builtin_msa_adds_u_b(px, add), bright, 0);
    print_string("pixels:     ");
    print_bytes(pixels);
    print_string("wrapping:   ");
    print_bytes(wrapped);
    print_string("saturating: ");
    print_bytes(bright);
    return 0;
}
// eMIPS: --release R5
