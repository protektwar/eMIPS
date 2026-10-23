/* emips_nano.ld — the eMIPS memory map for nanoMIPS C programs (python3
 * -m emips cc --arch nanomips; gold, nanomips-elf-ld): the layout the
 * nanoMIPS assembler gives assembly programs — code at 0x00400000, at
 * 0x10010000 the data, then (the toolchain's uhi32.ld order) _gp and the
 * small data the p32 ABI reaches from $gp, .sbss / .bss — and the stack
 * below 0x7FFFF000. */
ENTRY(__start)

SECTIONS
{
  . = 0x00400000;
  .text : {
    *crt0_nano.o(.text)
    *(.text .text.* .gnu.linkonce.t.*)
  }

  . = 0x10010000;
  .data : { *(.data .data.* .gnu.linkonce.d.*) }
  .rodata : { *(.rdata .rodata .rodata.* .gnu.linkonce.r.*) }
  /* the ABI flags: the ISA and the ASEs the code needs (DN00179 5.2) */
  .nanoMIPS.abiflags : { *(.nanoMIPS.abiflags) }
  . = ALIGN(8);
  _gp = .;
  .ssdata : { *(.ssdata .ssdata.*) }
  .ssbss : { *(.ssbss .ssbss.*) }
  .sdata : { *(.sdata .sdata.* .gnu.linkonce.s.*) }
  . = ALIGN(4);
  __bss_start = .;
  .sbss : { *(.sbss .sbss.* .scommon) }
  .bss : { *(.bss .bss.* COMMON) }
  . = ALIGN(4);
  _end = .;

  /* below MARS's $sp (0x7FFFEFFC), 16-byte aligned (the p32 ABI) */
  __stack_top = 0x7FFFEFF0;

  /DISCARD/ : { *(.comment) *(.note.*) *(.gnu.attributes) *(.eh_frame) }
}
