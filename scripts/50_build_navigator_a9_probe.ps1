$ErrorActionPreference = 'Stop'
$toolchain = 'C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin'
$gcc = Join-Path $toolchain 'arm-none-eabi-gcc.exe'
$objcopy = Join-Path $toolchain 'arm-none-eabi-objcopy.exe'
if (!(Test-Path -LiteralPath $gcc) -or !(Test-Path -LiteralPath $objcopy)) {
  throw 'GNU Arm Embedded Toolchain 14.2.rel1 is required.'
}
$out = 'hardware/build/navigator_a9_probe'
New-Item -ItemType Directory -Force -Path $out | Out-Null
& $gcc -mcpu=cortex-a9 -marm -mfloat-abi=soft -ffreestanding -fno-builtin -fdata-sections -ffunction-sections -nostdlib `
  '-Wl,-T,software/bringup/minimal_a9/linker.ld' '-Wl,--gc-sections' "-Wl,-Map,$out/navigator_a9_probe.map" `
  software/bringup/minimal_a9/start.S software/bringup/minimal_a9/navigator_a9_probe.c -o $out/navigator_a9_probe.elf
& $objcopy -O binary $out/navigator_a9_probe.elf $out/navigator_a9_probe.bin
Write-Output "NPU_A9_PROBE_ELF $(Resolve-Path $out/navigator_a9_probe.elf)"
