"""Launch immutable catalog cubins with the same argument ABI and PDL settings as Zig."""
import ctypes as C
from pathlib import Path


class AttributeValue(C.Union):
    _fields_ = [('pad', C.c_uint8 * 64), ('integer', C.c_int), ('alignment', C.c_uint64)]


class Attribute(C.Structure):
    _fields_ = [('id', C.c_uint), ('pad', C.c_uint), ('value', AttributeValue)]


class Config(C.Structure):
    _fields_ = [(name, C.c_uint) for name in ('gx', 'gy', 'gz', 'bx', 'by', 'bz', 'shared')]
    _fields_ += [('stream', C.c_void_p), ('attributes', C.POINTER(Attribute)), ('count', C.c_uint)]


class Kernel:
    def __init__(self, directory, entry):
        self.entry = entry
        self.driver = C.CDLL('libcuda.so.1')
        self.module, self.function = C.c_void_p(), C.c_void_p()
        data = (Path(directory) / 'cubins' / (entry['hash'] + '.cubin')).read_bytes()
        self.data = C.create_string_buffer(data)
        self.check(self.driver.cuModuleLoadDataEx(C.byref(self.module), self.data, 0, None, None))
        self.check(self.driver.cuModuleGetFunction(C.byref(self.function), self.module, entry['name'].encode()))
        self.driver.cuLaunchKernelEx.argtypes = [C.POINTER(Config), C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        assert C.sizeof(Config) == 56 and Config.stream.offset == 32 and C.sizeof(Attribute) == 72

    def check(self, result):
        if result:
            message = C.c_char_p()
            self.driver.cuGetErrorString(result, C.byref(message))
            raise RuntimeError(f'CUDA {result}: {message.value!r}')

    def launch(self, values, grid, stream):
        storage = []
        for parameter in self.entry['params']:
            value = values[parameter['name']]
            ty = parameter['type']
            if ty.startswith('*'):
                storage.append(C.c_uint64(value.data_ptr()))
            else:
                storage.append({'i32': C.c_int32, 'u64': C.c_uint64, 'fp32': C.c_float}[ty](value))
        divisor = self.entry.get('grid_x_divisor', 1)
        if divisor > 1:
            storage.append(C.c_uint32(grid[0]))
        storage += [C.c_uint64(0), C.c_uint64(0)]
        pointers = (C.c_void_p * len(storage))(*(C.cast(C.pointer(value), C.c_void_p) for value in storage))
        attributes = (Attribute * 1)()
        attributes[0].id, attributes[0].value.integer = 6, 1
        grid = ((grid[0] + divisor - 1) // divisor, grid[1], grid[2])
        config = Config(*grid, 32 * self.entry['num_warps'], 1, 1, self.entry['shared'], stream,
                        attributes, int(self.entry['pdl']))
        self.check(self.driver.cuLaunchKernelEx(C.byref(config), self.function, pointers, None))

    def close(self):
        self.check(self.driver.cuModuleUnload(self.module))
