ENTRY(_start)

MEMORY {
  DDR (rwx) : ORIGIN = 0x12200000, LENGTH = 64K
}

SECTIONS {
  .text : { *(.text.start*) *(.text*) *(.rodata*) } > DDR
  .data : { *(.data*) } > DDR
  .bss (NOLOAD) : { *(.bss*) *(COMMON) } > DDR
  . = ALIGN(8);
  _stack_top = ORIGIN(DDR) + LENGTH(DDR);
}
