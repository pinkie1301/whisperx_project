import unittest

from scripts.run import (
    _release_memory,
    resolve_runtime_options,
)


class FakeCuda:
    def __init__(self, available):
        self.available = available
        self.empty_cache_called = False

    def is_available(self):
        return self.available

    def empty_cache(self):
        self.empty_cache_called = True


class FakeMps:
    def __init__(self, available):
        self.available = available
        self.empty_cache_called = False

    def is_available(self):
        return self.available

    def empty_cache(self):
        self.empty_cache_called = True


class FakeTorch:
    def __init__(self, cuda_available=False, mps_available=False):
        self.cuda = FakeCuda(cuda_available)
        self.mps = FakeMps(mps_available)


class RuntimeOptionsTest(unittest.TestCase):
    def test_auto_uses_cuda_float16_when_cuda_is_available(self):
        runtime = resolve_runtime_options(FakeTorch(cuda_available=True), "auto", "auto")

        self.assertEqual(runtime.device, "cuda")
        self.assertEqual(runtime.compute_type, "float16")

    def test_auto_uses_cpu_int8_when_cuda_is_unavailable(self):
        runtime = resolve_runtime_options(FakeTorch(cuda_available=False), "auto", "auto")

        self.assertEqual(runtime.device, "cpu")
        self.assertEqual(runtime.compute_type, "int8")

    def test_explicit_cuda_errors_when_cuda_is_unavailable(self):
        with self.assertRaisesRegex(SystemExit, "CUDA was requested"):
            resolve_runtime_options(FakeTorch(cuda_available=False), "cuda", "auto")

    def test_explicit_compute_type_overrides_device_default(self):
        runtime = resolve_runtime_options(FakeTorch(cuda_available=True), "cuda", "int8")

        self.assertEqual(runtime.device, "cuda")
        self.assertEqual(runtime.compute_type, "int8")

    def test_release_memory_clears_available_cuda_cache(self):
        torch_module = FakeTorch(cuda_available=True)

        _release_memory(torch_module)

        self.assertTrue(torch_module.cuda.empty_cache_called)


if __name__ == "__main__":
    unittest.main()
