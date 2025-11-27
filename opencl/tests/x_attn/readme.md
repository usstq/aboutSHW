# GEMM opt record
## base
### config:
```
EU tile: 32x32
XeCore: 4x4
query shape: [8k, 128], aka [512, 128*16]
key shape: [128k, 128], aka[8192, 128*16]
HQ: 32
HK: 8
precision: f16
GPU: B580, 20 Xecores @ 2.9G
```
### performance
 `88T/s`
### Why using this config?
EU tile is small, so there are enough free registers for extending.

## TRY1: add decompress before dot simply(ac3c652fc)
`28T/s`, `51T/s if remove '-noschedule'`
### Finding1: The following decompression code will be translated inefficently:
```c++
// Line 1279:  A1[i].select<BLOCK_REG_K, 1>(m * BLOCK_REG_K) = A1_i8[i].select<BLOCK_REG_K, 1>(m * BLOCK_REG_K);
(W)     mov (16|M0)              r11.0<2>:w    r39.48<1;1,0>:b                                       //  ALU pipe: int; $602
(W)     mov (16|M0)              r54.0<2>:hf   r11.0<2;1,0>:w                   {I@1}                //  ALU pipe: float; $602
(W)     mov (16|M0)              r53.16<1>:uw  r54.0<2;1,0>:uw                  {F@1}                //  ALU pipe: int; $602

// Line 1280:  A1[i].select<BLOCK_REG_K, 1>(m * BLOCK_REG_K) = (A1[i].select<BLOCK_REG_K, 1>(m * BLOCK_REG_K) - zps[1][i * 8 + m]) * scales[1][i * 8 + m];
(W)     mov (16|M0)              r4.0<1>:uw    r53.16<1;1,0>:uw                 {I@1}                //  ALU pipe: int; $604
(W)     add (16|M0)              r12.0<1>:hf   r4.0<1;1,0>:hf    -r7.23<0;1,0>:hf {I@1}              //  ALU pipe: float; $604
(W)     mul (16|M0)              r56.0<1>:hf   r12.0<1;1,0>:hf   r10.23<0;1,0>:hf {F@1}              //  ALU pipe: float; $605
(W)     mov (16|M0)              r53.16<1>:uw  r56.0<1;1,0>:uw                  {F@1}                //  ALU pipe: int; $605
```
### Finding2: Major problem should be the computation for decompression:
|metric|f16|i8(base)|comment|
|---|---:|---:|---|
|GPU_MEMORY_BYTE_READ|124,811,492,864|54,139,320,320|i8 is already lower than f16|
|XVE_INST_EXECUTED_ALU0_ALL|4,564,036,949|163,528,650,752|There are too many instructions in ALU0(FP) |
|XVE_INST_EXECUTED_ALU1_ALL|13,776,432,528|178,796,282,624|There are too many instructions in ALU1(INT)|
|XVE_INST_EXECUTED_ALU2_ALL|104,674,538,883|105,906,176,000|almost same in ALU2(XMX)|

## TRY2.0: ~~use pin-pong for A matrix~~, decompression concurrency from 16->32(4da5a594e)
`44T/s`, `63T/s if remove 'no-schedule'`
### Test data:
|metric|cur|i8(base)|comment|
|---|---:|---:|---|
|GPU_MEMORY_BYTE_READ|54,054,478,848|54,139,320,320|no change|
|XVE_INST_EXECUTED_ALU0_ALL|136,892,360,472|163,528,650,752|reduced but not enough |
|XVE_INST_EXECUTED_ALU1_ALL|125,696,550,983|178,796,282,624|reduced but not enough|
|XVE_INST_EXECUTED_ALU2_ALL|105,782,711,890|105,906,176,000|no change|

### reason for delay pin-pong:
Instructions number for ALU0/1 are still bigger than XMX, reduce them should be the right way.

## TRY2.1: decompression mul+add->mad(be8c1b3af2)
`47T/s`, `66T/s if remove 'no-schedule'`
### Test data:
|metric|cur|TRY2.0|comment|
|---|---:|---:|---|
|GPU_MEMORY_BYTE_READ|54,130,486,784|54,054,478,848|no change|
|XVE_INST_EXECUTED_ALU0_ALL|110,782,410,752|136,892,360,472|reduced ~26G, theoretical=128* 1024* 128/32* 32(heads)* 16(repeat)* 100(test)=26.8G |
|XVE_INST_EXECUTED_ALU1_ALL|125,843,194,624|125,696,550,983|no change|
|XVE_INST_EXECUTED_ALU2_ALL|105,906,176,000|105,782,711,890|no change|
NOTE: there may be **accuarcy** impact due to per-calc zp*scale:
```c++
origin:
in(half) = in(int8)
out(half) = (in(half) - zp(half)) * scale(half)
new:
temp(half) = zp(half) * scale(half)             // <-- acc lost here
in(half) = in(int8)
out(half) = in(int8) * scale(half) - temp(half) // aka mad
```

## TRY2.2: use uint8 instead of int8, use denormal * 32768 * 512 to convert uw to half(ded89bf85)
`59T/s`, `75T/s if remove 'no-schedule'`
### Test data:
|metric|cur|TRY2.1|comment|
|---|---:|---:|---|
|XVE_INST_EXECUTED_ALU0_ALL|79,548,362,752|110,782,410,752|reduced |
|XVE_INST_EXECUTED_ALU1_ALL|77,854,458,624|125,843,194,624|reduced |

New decompression asm code:
```c++
// Line 1276:  d0.format<ushort>() = A0_i8[m];
(W)     mov (32|M0)              r5.0<1>:w     r19.0<1;1,0>:ub                  {$6.dst}             //  ALU pipe: int; $943

// Line 1277:  d0 *= half{32768.0};
(W)     mul (32|M0)              acc0.0<1>:hf  r5.0<1;1,0>:hf    32768.0:hf              {I@1}       //  ALU pipe: float; $945

// Line 1278:  d0 = d0 * scales[0][m] - zps[0][m];
(W)     mad (32|M0)              r5.0<1>:hf    -r7.0<0;0>:hf     acc0.0<1;0>:hf    r10.0<0>:hf       //  ALU pipe: float; $947

// Line 1279:  A0[0 + m / 8].select<BLOCK_REG_K, 1>(m % 8 * BLOCK_REG_K) = d0.select<16, 1>(0);
(W)     mov (16|M0)              r63.0<1>:hf   r5.0<1;1,0>:hf                   {F@1}                //  ALU pipe: float; $949

// Line 1280:  A0[2 + m / 8].select<BLOCK_REG_K, 1>(m % 8 * BLOCK_REG_K) = d0.select<16, 1>(16);
(W)     mov (16|M0)              r59.0<1>:uw   r5.16<1;1,0>:uw                                       //  ALU pipe: int; $951
```

NOTE: there may be **accuarcy** impact due to per-upscale zp:
```c++
based on try 2.1:
temp(half) = zp(half) * scale(half)
in(half) = in(int8)
out(half) = in(half) * scale(half) - temp(half)
new:
temp(half) = zp(half) * scale(half)             // <-- acc lost here
scale_up(half) = scale(half) * 512              // <-- acc lost here
in(word) = in(uint8)
in(half) = reinterpret_cast<half>(in(word)) * 32768;
out(half) = in(half) * scale_up(half) - temp(half)
```

## TRY2.3: use uint8 instead of int8, zp&scale keep unchanged(98146fc05)
`55T/s`, `69T/s if remove 'no-schedule'`
### Test data:
|metric|cur|TRY2.2|comment|
|---|---:|---:|---|
|XVE_INST_EXECUTED_ALU0_ALL|92,786,634,752|79,548,362,752|increased |
|XVE_INST_EXECUTED_ALU1_ALL|90,885,882,624|77,854,458,624|increased |

Asm code:
```c++
// Line 1276:  d0.format<ushort>() = A0_i8[m];
(W)     mov (32|M0)              r6.0<1>:w     r30.32<1;1,0>:ub                                      //  ALU pipe: int; $548

// Line 1277:  d0 *= half{32768.0};
(W)     mul (32|M0)              acc0.0<1>:hf  r6.0<1;1,0>:hf    32768.0:hf              {I@1}       //  ALU pipe: float; $550

//             r2.0 = 512.0
(W)     mad (32|M0)              acc0.0<1>:hf  -r7.1<0;0>:hf     acc0.0<1;0>:hf    r2.0<0>:hf        //  ALU pipe: float; $552
// Line 1279:  d0 = (d0 - zps[0][m]) * scales[0][m];
(W)     mul (32|M0)              r6.0<1>:hf    r10.1<0;1,0>:hf   acc0.0<1;1,0>:hf                    //  ALU pipe: float; $553

// Line 1280:  A0[0 + m / 8].select<BLOCK_REG_K, 1>(m % 8 * BLOCK_REG_K) = d0.select<16, 1>(0);
(W)     mov (16|M0)              r47.16<1>:uw  r6.0<1;1,0>:uw                   {F@1}                //  ALU pipe: int; $555

// Line 1281:  A0[2 + m / 8].select<BLOCK_REG_K, 1>(m % 8 * BLOCK_REG_K) = d0.select<16, 1>(16);
(W)     mov (16|M0)              r51.16<1>:uw  r6.16<1;1,0>:uw                                       //  ALU pipe: int; $557
```

## TRY3.0: reuse decompression result: increase B tile from 32->64(553b73d36dc)
32x32 tile #reg(try2.3):
```
C: 32*32*4/64=64
A: i8data: 32*32/64=16, i8->f16: 32*16*2/64=16
B: 16*16*2(half)*2(32tile)*2(copy)/64=32
all: 128+
```
32x64 tile #reg(cur):
```
C: 32*64*4/64=128
A: i8data: 32*32/64=16, i8->f16: 32*16*2/64=16
B: 16*16*2(half)*4(64tile)*2(copy)/64=64
all: 224+
```
`78T/s`, `83T/s if remove 'no-schedule'`
### Test data:
|metric|cur|TRY2.3|comment|
|---|---:|---:|---|
|GPU_MEMORY_BYTE_READ|54,130,486,784|54,158,295,040|no change|
|XVE_INST_EXECUTED_ALU0_ALL|49,701,618,432|92,786,634,752|decreased |
|XVE_INST_EXECUTED_ALU1_ALL|48,216,760,064|90,885,882,624|decreased |
|XVE_INST_EXECUTED_ALU2_ALL|106,753,425,408|105,906,176,000|no change|

## TRY3.1: ~~use pin-pong for A matrix~~
32x64 tile #reg(based on try3.0):
```
C: 32*64*4/64=128
A: i8data: 32*32/64=16, i8->f16: 32*16*2/64=16, another copy +16
B: 16*16*2(half)*4(64tile)*2(copy)/64=64
other: 5(desc), 2(scale/zp)+1(aux), 2(temp for dec)
all: 250+
```
The free registers are not enough, only free some registers for i8->f16 then the compilation could pass but this will make decompression cost increased. NA.

## TRY3.2: use q*k' + pin-pong for B matrix(54e0af3a91)
u8: `101T/s`, `102T/s if remove 'no-schedule'`
f16: `105T/s`, `104T/s if remove 'no-schedule'`
### major changes:
- use q * k'(tile: 64x32) instead of k * q':
  - prons:
    - pin-pong can be better to hide latency of loading B
    - decoding B and load A can be parallel
    - save regs comparing to TRY3.1
  - cons:
    - need transpose or reduce to compute softmax+sum, both are inefficent
- ues block load for scale and zp, scatter load will cost more
- remove atomic to get max
### headroom for next:
hardware roofline: `2.9G*20eu*8*32*4*2=118T/s`, current hits about 85%/89%, the headroom is small. Current hotsopt should be `reduce2d` in softmax stage which may get `108T/s` if remove them. But, there is no good way to optimize them now.

## perf tools
### intel-gpu-tools
- record: `xe-perf-recorder --metric ComputeBasic`
- decode: `xe-perf-reader xe_perf.record -c all`



# Porting to Xe1

## Statefull block load/store/prefetch
Hardware will return 0 for the out-of-bound bytes.
```
template <typename T, int NElts, DataSize DS = DataSize::Default,
          CacheHint L1H = CacheHint::Default,
          CacheHint L2H = CacheHint::Default>
vector<RetTy, NElts> cm_load(SurfaceIndex Idx, unsigned Offset);

cm_prefetch
```

## Stateless block load/store/prefetch
Hardware will return 0 for the out-of-bound bytes.
```
template <typename T, int NElts, DataSize DS = DataSize::Default,
          CacheHint L1H = CacheHint::Default,
          CacheHint L2H = CacheHint::Default>
vector<RetTy, NElts> cm_ptr_load(const T *const Ptr, unsigned Offset);

cm_ptr_prefetch
```

## charpter4.19 Shared Virtual Memory (SVM)

Return 0 if out of boundary. The larger N is, performance improves. (Mannually verified on DG2.)

```
cm_svm_block_read(svmptr_t v_Addr, vector_ref<TYPE, N> v_Src);
```

## clang/lib/Headers/cm/include/cm/cm_pointer.h

HW hangs when trying to read memory out of boundary. (Mannually verified on LNL.)

```
template <typename T0, int N, int M>
CM_NODEBUG CM_INLINE void
cm_ptr_block_read(const T0 *const addr,
                  matrix_ref<details::remove_address_space_t<T0>, N, M> dst)
```


# The following load methods are not applicable to Xe1.

## Untyped 2D block load/store/prefetch
target-dependent and only available when CM_HAS_LSC_UNTYPED_2D macro is defined.
```
template <typename T, int Width, int Height = 1, int NumBlocks = 1,
          bool Transposed = false, bool Transformed = false,
          CacheHint L1H = CacheHint::Default,
          CacheHint L2H = CacheHint::Default,
vector<T, N> cm_ptr_load(T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
                         unsigned SurfacePitch, int X, int Y);

cm_ptr_prefetch
```

## Untyped descriptor based 2D block load/store/prefetch
target-dependent and only available when CM_HAS_LSC_UNTYPED_2D macro is defined.
```
template <lsc::LoadOp Op = lsc::LoadOp::Normal,
          CacheHint L1H = CacheHint::Default,
          CacheHint L2H = CacheHint::Default,
          int OffsetX = 0, int OffsetY = 0>
void cm_load(details::Block2DRefTy<T, BlockH, BlockW, NBlocks, Op> Res,
            const lsc::block_2d_desc<T, NBlocks, BlockH, BlockW> &Desc,
            int16_t Pred = 1);

cm_prefetch
```

## Typed load/store/prefetch
target-dependent and only available when CM_HAS_LSC_TYPED macro is defined.
```
cm_load4_typed
cm_prefetch4_typed
```

## Typed 2D block load/store
target-dependent and only available when CM_HAS_LSC_TYPED_2D macro is defined.
```
cm_load(SurfaceIndex Idx, int X, int Y, matrix_ref<T, Height, Width> Data);
cm_prefetch(SurfaceIndex Idx, int X, int Y);
```