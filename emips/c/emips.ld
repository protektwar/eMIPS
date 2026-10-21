/* emips.ld — the eMIPS memory map for C programs (python3 -m emips cc):
 * the MARS / eMIPS assembler layout — code at 0x00400000, data at
 * 0x10010000, the stack below 0x7FFFF000 — so the views, the examples and
 * the labs find things where they are for assembly programs.  Both ABIs:
 * o32 (ELF32) and n64 (ELF64, emips cc --arch mips64; the addresses are
 * in the 32-bit compatibility space). */
OUTPUT_ARCH(mips)
ENTRY(__start)

SECTIONS
{
  . = 0x00400000;
  .text : {
    *crt0.o(.text)
    *(.text .text.* .gnu.linkonce.t.*)
  }
  .rodata : { *(.rodata .rodata.* .gnu.linkonce.r.*) }
  /* the ABI flags: the ISA and the ASEs the code needs (the DSP Module) */
  .MIPS.abiflags : { *(.MIPS.abiflags) }

  . = 0x10010000;
  .data : { *(.data .data.* .gnu.linkonce.d.*) }
  _gp = ALIGN(16) + 0x7FF0;
  .sdata : { *(.sdata .sdata.*) }
  . = ALIGN(4);
  __bss_start = .;
  .sbss : { *(.sbss .sbss.* .scommon) }
  .bss : { *(.bss .bss.* COMMON) }
  . = ALIGN(4);
  _end = .;

  /* the stack below MARS's $sp (0x7FFFEFFC), aligned as the ABIs want it:
   * 8 bytes for o32 (a double on the stack: SDC1 / LDC1), 16 for n64 */
  __stack_top = 0x7FFFEFF0;

  /DISCARD/ : {
    *(.reginfo) *(.pdr) *(.comment) *(.note.*)
    *(.gnu.attributes) *(.MIPS.options)
  }
}
