/* emips_rv.ld — the eMIPS memory map for RISC-V C programs (python3 -m
 * emips cc --arch rv32 / rv64): RARS's map, the layout eMIPS's RISC-V
 * assembler gives assembly programs (emips.rvld) — code at 0x00400000
 * (crt0 first) and the read-only data after it; at 0x10010000 the data,
 * the small data, .sbss, .bss — with GNU ld's __global_pointer$ (its
 * default script's formula) and the stack below 0x7FFFF000. */
ENTRY(_start)

SECTIONS
{
  . = 0x00400000;
  .text : {
    *crt0_rv.o(.text)
    *(.text .text.* .gnu.linkonce.t.*)
  }
  .rodata : { *(.rodata .rodata.* .gnu.linkonce.r.*) }

  . = 0x10010000;
  .data : {
    __DATA_BEGIN__ = .;
    *(.data .data.* .gnu.linkonce.d.*)
  }
  .sdata : {
    __SDATA_BEGIN__ = .;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2)
    *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  }
  __bss_start = .;
  .sbss : { *(.sbss .sbss.* .gnu.linkonce.sb.*) *(.scommon) }
  .bss : {
    *(.bss .bss.* .gnu.linkonce.b.*) *(COMMON)
    . = ALIGN(. != 0 ? 8 : 1);
  }
  . = ALIGN(8);
  __BSS_END__ = .;
  __global_pointer$ = MIN(__SDATA_BEGIN__ + 0x800,
                          MAX(__DATA_BEGIN__ + 0x800, __BSS_END__ - 0x800));
  _end = .;

  /* RARS's sp (0x7FFFEFFC) rounded down to the 16 bytes the psABI wants */
  __stack_top = 0x7FFFEFF0;

  /DISCARD/ : { *(.comment) *(.note.*) *(.eh_frame) }
}
