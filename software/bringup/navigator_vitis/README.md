# Navigator Z7020 Vitis standalone NPU runner

This directory is the first PS-side driver application.  It deliberately uses
only JTAG, volatile DDR and UART0; it has no QSPI, eMMC or SD-card write path.

`npu_vitis_platform.[ch]` adapts the portable CSR driver to the Xilinx
standalone BSP.  The runner copies a complete task image into a 2 MiB,
64-byte-aligned DDR buffer, submits its physical address through the NPU CSR,
waits by polling, and compares the NPU-written output with a generated golden
buffer.  HP0 is non-coherent: the portable driver calls the adapter's clean
callback before the doorbell and invalidate callback after completion.

## Generate the local task payload

Use the exact task image and golden output used for the intended board test.
The following uses the deterministic nonzero test that has already passed via
JTAG; it only creates ignored local generated C source.

```powershell
python scripts/63_generate_vitis_task_payload.py `
  --image hardware/build/navigator_seeded_task/task_image.bin `
  --golden hardware/build/navigator_seeded_task/expected_output.bin `
  --program-dir hardware/generated/bott2_mir1k_v1_program `
  --output-dir software/bringup/navigator_vitis/generated
```

This verifies the task's 64-byte alignment and derives the output offset from
`program.json` plus `task_image.json`.  The generated header/source expose the
task image, expected output, output offset and command count.

## Build and run in Vitis

1. In Vivado, use the candidate XSA exported by
   `scripts/43_export_navigator_candidate.tcl`; do **not** use the vendor
   audio-loopback XSA.
2. Create a Zynq-7000 standalone domain/application.  Add
   `software/src/npu_driver.c`, `software/include/`, and every `.c` in this
   directory (including `generated/navigator_npu_task_payload.c`).  The runner
   uses the normal Vitis `platform.h` hooks to initialise/clean up the BSP.
3. Confirm `xparameters.h` supplies one of the `STEM_NPU_0_*BASEADDR` macros
   and that its value is `0x43C00000`.  The application refuses another base.
4. Leave the BSP caches enabled.  Program the candidate bitstream over JTAG,
   then download the ELF over JTAG and open UART0 at 115200-8-N-1.
5. The success criterion is `NPU TASK PASS`, completed tag `0x56544953`,
   1,869 retired commands and a golden output match.  A task failure prints
   error code/PC/instruction tag and counters.

The 2 MiB buffer is a linker-owned static object, not a guessed hard-coded
DDR address.  In Zynq standalone the A9 pointer is identity-mapped to its HP0
DMA address.  If a future linker script places it outside PS DDR, stop and
fix the linker placement rather than passing a CPU virtual address to the NPU.

The runner is intentionally polling-only for this first software milestone.
The RTL IRQ is wired, but GIC instance/device IDs must be taken from the final
XSA before enabling interrupt-driven completion.
