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

## TRY2.3: use uint8 instead of int8, zp&scale keep unchanged()
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

## TRY3: reuse decompression result: increase B tile(TODO)
32x32 tile #reg(cur):
```
C: 32*32*4/64=64
A: i8data: 32*32/64=16, i8->f16: 32*16*2/64=16
B: 16*16*2(half)*2(32tile)*2(copy)/64=32
all: 128+
```
32x64 tile #reg(<span style="color:red;">NA</span>):
```
C: 32*64*4/64=128
A: i8data: 32*32/64=16, i8->f16: 32*16*2/64=16, another copy: 16+16
B: 16*16*2(half)*4(64tile)*2(copy)/64=64
all: >256
```

## perf tools
### intel-gpu-tools
- record: `xe-perf-recorder --metric ComputeBasic`
- decode: `xe-perf-reader xe_perf.record -c all`
