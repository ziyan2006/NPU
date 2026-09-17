/* SD cold-boot application: FSBL loads and enters it at 0x10000000. */
ENTRY(_start)

MEMORY {
  DDR (rwx) : ORIGIN = 0x10000000, LENGTH = 256M
}

SECTIONS {
  .text : { *(.text.start*) *(.text*) *(.rodata*) } > DDR
  .data : { *(.data*) } > DDR
  .bss (NOLOAD) : { *(.bss*) *(COMMON) } > DDR
  . = ALIGN(8);
  _stack_top = ORIGIN(DDR) + LENGTH(DDR);
}
