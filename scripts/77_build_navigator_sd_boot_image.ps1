$ErrorActionPreference = 'Stop'
$toolchain = 'C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi\14.2 rel1\bin'
$objcopy = Join-Path $toolchain 'arm-none-eabi-objcopy.exe'
$ld = Join-Path $toolchain 'arm-none-eabi-ld.exe'
$bootgen = 'C:\AMDDesignTools\2026.1\Vivado\bin\bootgen.bat'
$out = 'hardware/build/navigator_sd_coldboot_runner'
$rawFsbl = "$out/fsbl_raw.bin"
$fsblObj = "$out/fsbl_raw.o"
$fsblElf = "$out/fsbl_rewrapped.elf"
$runner = "$out/navigator_sd_coldboot_runner.elf"
$bif = 'software/bringup/minimal_a9/navigator_sd_coldboot.bif'
foreach ($file in @($objcopy, $ld, $bootgen, $rawFsbl, $runner, $bif, 'software/bringup/minimal_a9/linker_fsbl_rewrap.ld', 'hardware/build/navigator_z7020_vendor_v37_export/stem_npu_navigator_z7020_candidate.bit')) {
  if (!(Test-Path -LiteralPath $file)) { throw "missing required input: $file" }
}
if ((Get-Item -LiteralPath $rawFsbl).Length -ne 0x18008) {
  throw 'unexpected FSBL raw length; expected 0x18008 bytes'
}
& $objcopy -I binary -O elf32-littlearm -B arm $rawFsbl $fsblObj
if ($LASTEXITCODE -ne 0) { throw 'FSBL raw-to-object conversion failed.' }
& $ld -m armelf -T software/bringup/minimal_a9/linker_fsbl_rewrap.ld -o $fsblElf $fsblObj
if ($LASTEXITCODE -ne 0) { throw 'FSBL ELF rewrap failed.' }
& $bootgen -arch zynq -image $bif -o "$out/BOOT.BIN" -w
if ($LASTEXITCODE -ne 0) { throw 'Bootgen failed.' }
& $bootgen -arch zynq -read "$out/BOOT.BIN" | Tee-Object -FilePath "$out/BOOT.read.txt"
if ($LASTEXITCODE -ne 0) { throw 'Bootgen readback failed.' }
Write-Output "NPU_SD_BOOT_IMAGE $(Resolve-Path "$out/BOOT.BIN")"
