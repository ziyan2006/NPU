/* Probe only: pad the loadable segment to a 32-bit boundary for legacy FSBLs. */
ENTRY(_start)

MEMORY {
  DDR (rwx) : ORIGIN = 0x10000000, LENGTH = 256M
}

SECTIONS {
  .text : {
    *(.text.start*)
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
  } > DDR
  .data : { *(.data*) } > DDR
  .bss (NOLOAD) : { *(.bss*) *(COMMON) } > DDR
  . = ALIGN(8);
  _stack_top = ORIGIN(DDR) + LENGTH(DDR);
}
