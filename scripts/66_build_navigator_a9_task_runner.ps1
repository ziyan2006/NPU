$ErrorActionPreference = 'Stop'
$toolchain = 'C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin'
$gcc = Join-Path $toolchain 'arm-none-eabi-gcc.exe'
$objcopy = Join-Path $toolchain 'arm-none-eabi-objcopy.exe'
if (!(Test-Path -LiteralPath $gcc) -or !(Test-Path -LiteralPath $objcopy)) {
  throw 'GNU Arm Embedded Toolchain 14.2.rel1 is required.'
}
$out = 'hardware/build/navigator_a9_task_runner'
New-Item -ItemType Directory -Force -Path $out | Out-Null
& $gcc -mcpu=cortex-a9 -marm -mfloat-abi=soft -ffreestanding -fno-builtin -fdata-sections -ffunction-sections -nostdlib `
  '-Isoftware/include' '-Wl,-T,software/bringup/minimal_a9/linker.ld' '-Wl,--gc-sections' "-Wl,-Map,$out/navigator_a9_task_runner.map" `
  software/bringup/minimal_a9/start.S software/bringup/minimal_a9/navigator_a9_task_runner.c software/src/npu_driver.c `
  -lgcc -o $out/navigator_a9_task_runner.elf
if ($LASTEXITCODE -ne 0) { throw 'A9 task-runner link failed.' }
& $objcopy -O binary $out/navigator_a9_task_runner.elf $out/navigator_a9_task_runner.bin
if ($LASTEXITCODE -ne 0) { throw 'A9 task-runner binary conversion failed.' }
Write-Output "NPU_A9_TASK_RUNNER_ELF $(Resolve-Path $out/navigator_a9_task_runner.elf)"
