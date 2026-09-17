$ErrorActionPreference = 'Stop'
$toolchain = 'C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin'
$gcc = Join-Path $toolchain 'arm-none-eabi-gcc.exe'
$objcopy = Join-Path $toolchain 'arm-none-eabi-objcopy.exe'
$payload = 'software/bringup/navigator_vitis/generated'
$out = 'hardware/build/navigator_sd_coldboot_runner'
foreach ($file in @($gcc, $objcopy, "$payload/navigator_npu_task_payload.c", "$payload/navigator_npu_task_payload.h")) {
  if (!(Test-Path -LiteralPath $file)) { throw "missing required input: $file" }
}
New-Item -ItemType Directory -Force -Path $out | Out-Null
& $gcc -mcpu=cortex-a9 -marm -mfloat-abi=soft -O2 -ffreestanding -fno-builtin -fdata-sections -ffunction-sections -nostdlib `
  '-Isoftware/include' "-I$payload" '-Wl,-T,software/bringup/minimal_a9/linker_sd_coldboot.ld' '-Wl,--gc-sections' `
  "-Wl,-Map,$out/navigator_sd_coldboot_runner.map" `
  software/bringup/minimal_a9/start.S software/bringup/minimal_a9/navigator_sd_coldboot_runner.c `
  software/src/npu_driver.c "$payload/navigator_npu_task_payload.c" `
  -lgcc -o $out/navigator_sd_coldboot_runner.elf
if ($LASTEXITCODE -ne 0) { throw 'SD coldboot runner link failed.' }
& $objcopy -O binary $out/navigator_sd_coldboot_runner.elf $out/navigator_sd_coldboot_runner.bin
if ($LASTEXITCODE -ne 0) { throw 'SD coldboot runner binary conversion failed.' }
Write-Output "NPU_SD_COLDBOOT_RUNNER_ELF $(Resolve-Path $out/navigator_sd_coldboot_runner.elf)"
