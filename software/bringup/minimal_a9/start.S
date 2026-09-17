.syntax unified
.cpu cortex-a9
.arm

.global _start
.type _start, %function
.extern main
.extern _stack_top

_start:
    ldr sp, =_stack_top
    bl main
1:  wfi
    b 1b
