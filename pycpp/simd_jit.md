# SIMDJit

code generation of register expression:
 - register allocation: all free registers are scratch registers that can be used w/o actual allocation
   the challenge is, how to allocate the minimal number of scratch regsiter, since at runtime, the 
   available number of scratch registers maybe extreamly limited.

   so the idea is : 
   take a snapshot of free registers as the scratch register pool
   this pool is changing dynamically:
       - since all temp register can be used only once, so after it has been passed to next
         instruction as source, it can be freed and reused.
         for example:
          - `rhs >> 6` => `rhs >>= 6`
          - `7 + (rhs)`, if rhs is itself a scratch-register, then we can do `rhs += 7` w/o 
             allocate new scratch register at all.
          - `(rhs) - 7`, if rhs is itself a scratch-register, then we can do `rhs -= 7`
          - `7 - (rhs)`, if rhs is itself a scratch-register, then we can do `rhs = neg(rhs); rhs += 7`
          - `6 >> rhs` => `scatch=6; scatch >>= rhs; free rhs`
       - if dst register is not used as source register, Or after the dst-register has been accessed
         the last time as source register, it can be added into scratch buffer list. this optimization
         requires one-pass to mark the last op accessing dst-register. and also mark the last temp register
         so it can be replaced with dst register

   so we will do one-pass temp register coloring, mark last op accessing dst-register, and switch last temp
   register to this dst-register if possible (this may broke the scratch register coloring).

   Addressing pattern matching: `[base + index*(1,2,4,8) + disp]` this pattern can be calculated using
   `LEA` very efficently, and it will not override base or index register, so it's more powerful than
   normal add instruction.

   basically, we are doing instruction lowering like LLVM:
    - 3-OP IR:
        `t0 = r0 + r2`
        `t1 = t0 * 4`
        `t2 = t1 + 89`
    - LEA pattern matching:
        `t2 = LEA(r0 + t0*4 + 89)`
    - reuse scratch reg, since after scratch register is being accessed lastly, it can be reused:
      the same is true for dst register:
        `t0 = r0 + r2`
        `t0 = t0 * 4`
    - convert it into 2-OP IR (`+=/-=/*=/>>=/<<=`):
        - `t0 = r0`
        - `t0 += r2`
        - `t0 *= 4`
    - temp register renaming

  and the IR data structure is RegExprImpl, which is binary tree like data structure
  and it can represent 3-OP
   

# Comparing operation and boolean expression
  when a regexpr is used in control-flow context, additional compare with zero will be required
  to check if it's boolean value 'false' or not, depending on which control flow is changed.

```bash
        cmp     edi, 1
        jle     .L4
        cmp     esi, 2
        jg      .L7
.L4:
        ret
.L7:
```

  in this context, if the last binary-OP in expression is compare OP, we can directly remove the
  step that convert EFLAGs into boolean value, this is a context-dependent optimization.

  but when compare-OP appears with-in normal context, it needs to be converted into value using
  https://www.felixcloutier.com/x86/setcc, and 
```bash
        cmp     edi, 1
        setg    bl        # a0 > 1
        cmp     esi, 2
        setg    al        # a1 > 2
        and     ebx, eax  # a0 > 1 && a1 > 2;
```

  this context dependent code generation is complex, we can instead always using the normal context
  in control-flow context case, add an additional `test bl, bl` to set ZF accordingly.

  and also add following optimizations:
   - in-context of control-flow, evaluate will got a target Label which will be jump target
     when condition is true(or false):
       - if the last-OP is binary comparing, we will generate `cmp` & `Jcc` directly
       - if the last-OP is not binary comparing, we have-to add a `test` to set true-false flag
         so that ZF is set according to the value of boolean expression.
  
  in `evaluate` function, we can pass a special flag to indicate that we are generating EFLAGS instead
  of assign final boolean value to a register.
  
  also we will support bit-wise `|` only, comparing op generate 1 or 0 results, but `||` and `&&` also accept normal int value and convert to 1 or 0 before `||` and `&&`, for example `(0x80 & 0x01)` is 0, but `0x80 || 0x01` is true instead !!!
  
  so `||` will check if lhs or rhs is result of comparing OP, if not we simply report error (programer will change their writing style and insert comparing OP (for example `a != 0` will convert a into boolean) )


  Loss of short-circuiting: to be simpler, we ignore short-circuiting behaviour.


