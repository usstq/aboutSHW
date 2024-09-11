/*
    following test shows that on average:
        2MB generate 1 SW_PAGE_FAULT which roughly take 0.5ms

*/

#include "../include/linux_perf.hpp"

void test(int64_t size) {
    char * p = reinterpret_cast<char*>(malloc(size));
    {
        auto prof = LinuxPerf::Profile("memset", 0, size);
        for(int64_t i = 0; i < size; i++) p[i] = i;
    }
    free(p);
}

int main() {
    for(int64_t sz = 4096; sz < 32ll*1024*1024*1024; sz *= 2) {
        printf("test sz=%ld\n", sz);
        test(sz);
    }
    return 0;
}
