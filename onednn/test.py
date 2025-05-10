import csrc
import psutil
import numpy as np

proc = psutil.Process()

print(csrc.data_type.s4)

print(f"Memory: {proc.memory_info()}")


md = csrc.memory_desc([2, 1024*1024], csrc.data_type.s8, csrc.format_tag.ab)
mem = csrc.memory(md)


csrc.memory(np.ones([128, 128], np.float32))
csrc.memory(np.ones([128, 128], np.float16))
csrc.memory(np.ones([128, 128], np.int32))
csrc.memory(np.ones([128, 128], np.int8))
csrc.memory(np.ones([128, 128], np.uint8))

org = np.random.randint(-128, 128, size=(2, 3), dtype=np.int8)
mem = csrc.memory(org.transpose().copy())
print(org)
m1np = mem.numpy()
m2 = mem.reorder(csrc.memory_desc([3,2],
                 csrc.data_type.s32,
                 csrc.format_tag.ba))
m2np = m2.numpy()

print(m1np, m1np.shape, m1np.strides, mem.md)
print(m2np, m2np.shape, m2np.strides, m2.md)

o2 = np.random.randint(-128, 128, size=(3, 2)).astype(np.float32)
m2 = csrc.memory(o2)
#n2 = np.array(m2, copy=False)
n2 = m2.numpy()

print(o2, o2.__array_interface__)
print(n2, n2.__array_interface__, m2)

print(f"Memory: {proc.memory_info()}")
