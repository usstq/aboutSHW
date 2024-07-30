#include <immintrin.h>

__m512i test(char x) {
    auto a = _mm512_set1_epi8(x);
    auto b = _mm512_set1_epi8(127);
    return _mm512_add_epi8(a, b);
}


int test2(int x) {
    return x*64;
}

char buffer[1024] = {};

void test3(__m512i x) {
    _mm512_storeu_si512(&buffer[0], x);
}

