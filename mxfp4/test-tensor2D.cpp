#define JIT_DEBUG
#include "../include/jit.h"


int main() {
    tensor2D<int> ti(32, 64);
    for(int i0 = 0; i0 < 32; i0++) {
        for(int i1 = 0; i1 < 64; i1++) {
            ti(i0, i1) = i0 + i1;
        }
    }
    tensor2D<int> ti2(32, 64);
    for(int i0 = 0; i0 < 32; i0++) {
        for(int i1 = 0; i1 < 64; i1++) {
            ti2(i0, i1) = i0 + i1;
        }
    }
    ti2(10, 20) = 0;
    show_tensor2D("ti", ti);

    compare(ti, ti2, true);
    return 0;
}
