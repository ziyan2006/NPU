#!/usr/bin/env python3
"""Compile/run the Vitis adapter against a tiny host-side Xilinx BSP stub."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


TEST_SOURCE = r'''#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "xil_types.h"
#include "npu_vitis_platform.h"

unsigned clean_calls;
unsigned invalidate_calls;
size_t clean_bytes;
size_t invalidate_bytes;

void Xil_DCacheFlushRange(INTPTR address, u32 bytes) {
    (void)address;
    ++clean_calls;
    clean_bytes += bytes;
}
void Xil_DCacheInvalidateRange(INTPTR address, u32 bytes) {
    (void)address;
    ++invalidate_calls;
    invalidate_bytes += bytes;
}

int main(void) {
    uint32_t registers[0x100 / sizeof(uint32_t)];
    uint8_t image[512];
    uint8_t expected[4] = { 1u, 2u, 3u, 4u };
    static uint8_t task[1024] __attribute__((aligned(64)));
    npu_device_t device;
    npu_vitis_task_buffer_t buffer;
    npu_completion_t completion;

    memset(registers, 0, sizeof(registers));
    memset(image, 0xa5, sizeof(image));
    registers[NPU_REG_IP_ID >> 2] = NPU_IP_ID_VALUE;
    registers[NPU_REG_VERSION >> 2] = 0x00010000u;
    registers[NPU_REG_ISA_VERSION >> 2] = 0x00010000u;
    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE;
    assert(npu_vitis_device_init_at(&device, (uintptr_t)registers) == NPU_OK);
    assert(npu_vitis_task_buffer_init(&buffer, task, sizeof(task)) == NPU_OK);
    assert(npu_vitis_stage_task(&buffer, image, sizeof(image)) == NPU_OK);
    assert(memcmp(task, image, sizeof(image)) == 0);
    assert(npu_vitis_submit_staged(&device, &buffer, 0x56495453u, 1234u)
           == NPU_OK);
    assert(clean_calls == 1u && clean_bytes == sizeof(image));
    assert(registers[NPU_REG_TASK_BASE_LO >> 2] == (uint32_t)(uintptr_t)task);
    assert(registers[NPU_REG_TASK_BYTES >> 2] == sizeof(image));
    assert(registers[NPU_REG_DOORBELL >> 2] == 1u);

    task[64] = 1u; task[65] = 2u; task[66] = 3u; task[67] = 4u;
    registers[NPU_REG_COMPLETED_TAG >> 2] = 0x56495453u;
    registers[NPU_REG_STATUS >> 2] = NPU_STATUS_IDLE | NPU_STATUS_DONE;
    assert(npu_wait(&device, 1u, &completion) == NPU_OK);
    assert(invalidate_calls == 1u && invalidate_bytes == sizeof(image));
    assert(npu_vitis_verify_output(&buffer, 64u, expected, sizeof(expected)));
    assert(!npu_vitis_verify_output(&buffer, 510u, expected, sizeof(expected)));
    puts("npu_vitis_platform: PASS");
    return 0;
}
'''


def main() -> None:
    compiler = next(
        (path for name in ("cc", "gcc", "clang") if (path := shutil.which(name))),
        None,
    )
    cross = Path(r"C:\AMDDesignTools\2026.1\gnu\microblaze\nt\bin\mb-gcc.exe")
    if compiler is None and not cross.exists():
        raise AssertionError("no native or Vitis C compiler available")
    with tempfile.TemporaryDirectory(prefix="npu_vitis_platform_") as raw:
        root = Path(raw)
        (root / "xil_types.h").write_text(
            "#include <stdint.h>\ntypedef uint32_t u32; typedef uintptr_t INTPTR;\n",
            encoding="utf-8",
        )
        (root / "xil_cache.h").write_text(
            "#include \"xil_types.h\"\n"
            "void Xil_DCacheFlushRange(INTPTR address, u32 bytes);\n"
            "void Xil_DCacheInvalidateRange(INTPTR address, u32 bytes);\n",
            encoding="utf-8",
        )
        (root / "xparameters.h").write_text(
            "#define XPAR_STEM_NPU_0_S_AXI_CTRL_BASEADDR 0x43C00000u\n",
            encoding="utf-8",
        )
        (root / "xil_printf.h").write_text(
            "int xil_printf(const char *format, ...);\n", encoding="utf-8"
        )
        (root / "platform.h").write_text(
            "void init_platform(void);\nvoid cleanup_platform(void);\n",
            encoding="utf-8",
        )
        (root / "fake_payload.h").write_text(
            "#include <stddef.h>\n#include <stdint.h>\n"
            "extern const uint8_t navigator_npu_task_image[];\n"
            "extern const size_t navigator_npu_task_image_bytes;\n"
            "extern const uint8_t navigator_npu_expected_output[];\n"
            "extern const size_t navigator_npu_expected_output_bytes;\n"
            "extern const size_t navigator_npu_output_offset;\n"
            "extern const uint32_t navigator_npu_expected_command_count;\n",
            encoding="utf-8",
        )
        test = root / "test.c"
        test.write_text(TEST_SOURCE, encoding="utf-8")
        common = [
            "-std=c11", "-Wall", "-Wextra", "-Werror",
            f"-I{root}", f"-I{ROOT / 'software' / 'include'}",
            f"-I{ROOT / 'software' / 'bringup' / 'navigator_vitis'}",
        ]
        sources = [
            ROOT / "software" / "src" / "npu_driver.c",
            ROOT / "software" / "bringup" / "navigator_vitis"
            / "npu_vitis_platform.c", test,
        ]
        if compiler is not None:
            executable = root / "test.exe"
            subprocess.run([compiler, *common, *map(str, sources), "-o",
                            str(executable)], check=True)
            result = subprocess.run([str(executable)], check=True,
                                    text=True, capture_output=True)
            if "npu_vitis_platform: PASS" not in result.stdout:
                raise AssertionError("adapter test did not report PASS")
            outcome = f"native compile/run ({compiler})"
        else:
            for source in sources:
                subprocess.run([str(cross), *common, "-ffreestanding", "-c",
                                str(source), "-o",
                                str(root / f"{source.stem}.o")], check=True)
            outcome = f"cross compile ({cross})"
        # The full standalone main is intentionally compiled separately: it
        # defines main and refers to a generated local payload, neither of
        # which belongs in the host runtime test above.
        runner = ROOT / "software" / "bringup" / "navigator_vitis" \
            / "navigator_npu_runner.c"
        runner_compiler = compiler if compiler is not None else str(cross)
        subprocess.run([
            runner_compiler, *common, "-ffreestanding", "-c",
            '-DNPU_VITIS_PAYLOAD_HEADER=\\"fake_payload.h\\"', str(runner),
            "-o", str(root / "navigator_npu_runner.o"),
        ], check=True)
    print(f"npu_vitis_platform_contract: PASS ({outcome})")


if __name__ == "__main__":
    main()
