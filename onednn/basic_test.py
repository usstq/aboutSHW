import dnnl
import numpy as np

def test_mm_i4():
    M = 128
    OC = 256
    IC = 512
    IC_group_size = 128
    assert IC % IC_group_size == 0
    IC_groups = IC // IC_group_size
    
    src = np.random.randint(-1, 2, size=(M, IC)).astype(np.float32)
    w_qi4 = np.random.randint(0, 16, size=(OC, IC_groups, IC_group_size)).astype(np.uint8)
    w_scale = np.random.randint(-8, 8, size=(OC, IC_groups, 1)).astype(np.float32) / 8
    w_zp = np.random.randint(5, 10, size=(OC, IC_groups, 1)).astype(np.uint8)
    
    w_scale = w_scale*0 + 1
    #w_zp = w_zp*0 + 1

    w_dequant = (w_qi4.astype(np.float32) - w_zp.astype(np.float32))*w_scale
    w_dequant = w_dequant.reshape([OC, -1])
    ref = src @ w_dequant.transpose()
    
    mm = dnnl.onednn_matmul()
    mm.init(dnnl.data_type.f32, dnnl.data_type.u4, M, IC, OC, IC_group_size)
    wei_md = dnnl.memory_desc()
    #wei_md = dnnl.memory_desc([OC, IC], dnnl.data_type.u4, dnnl.format_tag.ab)
    mm.create(wei_md)
    print(mm)
    
    src_mem = dnnl.memory(src)
    dst_mem = dnnl.memory(dnnl.memory_desc([M, OC], dnnl.data_type.f32, dnnl.format_tag.ab))
    wei_mem = dnnl.memory(w_qi4.reshape(OC, -1)).to_u4().reorder(mm.wei_md)
    sc_mem = dnnl.memory(w_scale.reshape(OC, IC_groups).transpose().copy())
    zp_mem = dnnl.memory(w_zp.reshape(OC, IC_groups).transpose().copy())
    
    mm.exec(src_mem, dst_mem, wei_mem, sc_mem, zp_mem, dnnl.memory())
    res = dst_mem.numpy()
    print(ref)
    print(res)
    print(zp_mem)
    assert np.allclose(ref, res)
    print(f"wei_mem={wei_mem}")
    
    
test_mm_i4()

def test_basic():
    md = dnnl.memory_desc([2, 1024*1024], dnnl.data_type.s8, dnnl.format_tag.ab)
    mem = dnnl.memory(md)

def test_memory():
    dnnl.memory(np.ones([128, 128], np.float32))
    dnnl.memory(np.ones([128, 128], np.float16))
    dnnl.memory(np.ones([128, 128], np.int32))
    dnnl.memory(np.ones([128, 128], np.int8))
    dnnl.memory(np.ones([128, 128], np.uint8))

    org = np.random.randint(-128, 128, size=(2, 3), dtype=np.int8)
    mem = dnnl.memory(org.transpose().copy())
    print(org)
    m1np = mem.numpy()
    m2 = mem.reorder(dnnl.memory_desc([3,2],
                    dnnl.data_type.s32,
                    dnnl.format_tag.ba))
    m2np = m2.numpy()

    print(m1np, m1np.shape, m1np.strides, mem.md)
    print(m2np, m2np.shape, m2np.strides, m2.md)

    o2 = np.random.randint(-128, 128, size=(3, 2)).astype(np.float32)
    m2 = dnnl.memory(o2)
    #n2 = np.array(m2, copy=False)
    n2 = m2.numpy()

    print(o2, o2.__array_interface__)
    print(n2, n2.__array_interface__, m2)

