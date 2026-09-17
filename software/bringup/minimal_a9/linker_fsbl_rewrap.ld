/* Rewrap an OCM-resident FSBL raw image in an ARM ELF for Bootgen. */
ENTRY(_start)
PHDRS { fsbl PT_LOAD FLAGS(5); }
SECTIONS {
  . = 0x00000000;
  .text : { _start = .; KEEP(*(.data)) } :fsbl
}
