from pathlib import Path
import unittest

from tools.check_translator_gpu import CHECKS, ROOT, command_for, device_filter, printed_block, vulkan_device, vulkan_device_name


class TranslatorCheckListTests(unittest.TestCase):
    def test_every_listed_tool_emitter_and_unit_test_exists(self):
        compiler = ROOT / "tools/shader-compiler"
        sources = "".join(path.read_text(encoding="utf-8") for path in (compiler / "src").rglob("*.rs"))
        for name, check in CHECKS.items():
            self.assertTrue((ROOT / "tools" / f"{check.tool}.py").is_file(), name)
            if check.emitter and check.emitter != "xshader":
                self.assertTrue((compiler / "examples" / f"{check.emitter}.rs").is_file(), name)
            if check.capture:
                self.assertIn(f"fn {check.capture[0]}", sources, name)

    def test_passes_each_tool_only_what_it_takes(self):
        tools = {"llvm-mc": "mc", "dxc": "dxc", "dxv": "dxv"}
        emitters = {check.emitter: Path("/emitters") / str(check.emitter) for check in CHECKS.values()}
        for name in CHECKS:
            command = command_for(name, emitters, tools, {"adapter": "10de:2705"}, Path("/work"))
            source = Path(command[1]).read_text(encoding="utf-8")
            for option in (word for word in command[2:] if word.startswith("--")):
                self.assertIn(option.removeprefix("--"), source, f"{name} {option}")

    def test_names_the_vulkan_device_for_an_adapter(self):
        summary = ("Devices:\n========\nGPU0:\n\tvendorID           = 0x10de\n\tdeviceID           = 0x2b85\n"
                   "\tdeviceName         = NVIDIA GeForce RTX 5090\nGPU1:\n\tvendorID           = 0x1002\n"
                   "\tdeviceID           = 0x13c0\n\tdeviceName         = AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)\n")
        self.assertEqual(vulkan_device_name("10de:2b85", summary), "NVIDIA GeForce RTX 5090")
        self.assertEqual(vulkan_device_name("1002:13C0", summary), "AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)")
        self.assertIsNone(vulkan_device_name("10de:2705", summary))

    def test_filters_by_the_device_uuid(self):
        # A Proton guest can see the device under another name, so the name cannot select it.
        summary = ("Devices:\n========\nGPU0:\n\tvendorID           = 0x1002\n\tdeviceID           = 0x73bf\n"
                   "\tdeviceName         = AMD Radeon RX 6900 XT (RADV NAVI21)\n"
                   "\tdeviceUUID         = 00000000-0300-0000-0000-000000000000\n"
                   "GPU1:\n\tvendorID           = 0x1002\n\tdeviceID           = 0x164e\n"
                   "\tdeviceName         = AMD Ryzen 9 7900X 12-Core Processor (RADV RAPHAEL_MENDOCINO)\n"
                   "\tdeviceUUID         = 00000000-1300-0000-0000-000000000000\n")
        self.assertEqual(device_filter(vulkan_device("1002:73bf", summary)),
                         {"DXVK_FILTER_DEVICE_UUID": "00000000030000000000000000000000"})
        self.assertEqual(device_filter(vulkan_device("1002:164e", summary)),
                         {"DXVK_FILTER_DEVICE_UUID": "00000000130000000000000000000000"})
        self.assertIsNone(vulkan_device("10de:2705", summary))
        # Without a UUID in the summary, the name is all there is.
        self.assertEqual(device_filter({"deviceName": "NVIDIA GeForce RTX 5090"}),
                         {"VKD3D_FILTER_DEVICE_NAME": "NVIDIA GeForce RTX 5090",
                          "DXVK_FILTER_DEVICE_NAME": "NVIDIA GeForce RTX 5090"})
        self.assertEqual(device_filter({}), {})

    def test_reads_the_program_a_unit_test_prints(self):
        output = "running 1 test\nBEGIN_HLSL\nvoid main() {}\n\nEND_HLSL\ntest ok\nBEGIN_HLSL\nsecond\nEND_HLSL\n"
        self.assertEqual(printed_block(output), "void main() {}\n\n")
        self.assertEqual(printed_block("test a::b ... BEGIN_ATOMIC\nbody\nEND_ATOMIC\n"), "body\n")
        with self.assertRaises(ValueError):
            printed_block("running 0 tests\n")


if __name__ == "__main__":
    unittest.main()
