## JIT

[./jit.h](./jit.h) : a jit base class based on [xbayk](https://github.com/herumi/xbyak), providing some additional features:
 - `JIT_DEBUG=name` or `JIT_DEBUG=*` will show disassembly code of jit and triger `INT3` to cause GDB to stop for runtime debuging.
 - runtime logging infrastructure which can log`zmm` registers and show after kernel invokation, very helpful for debugging.

## Compute bound Analysis
[./test-cpi.cpp](./test-cpi.cpp) using jit to test instruction/instruction-sequence throughput using PMU.

This is measuring actual throughput rather than prediction done by static code analyzer like [uiCA](https://uica.uops.info/), this is helpful to understand & optimize computational bound.


