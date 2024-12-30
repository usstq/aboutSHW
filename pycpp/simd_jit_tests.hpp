#include "simd_jit.hpp"

std::vector<std::function<void(void)>> test_exprs;

#if defined(__AVX2__) || defined(__AVX512F__)

#define TEST_EXPR(expr_name)                                                                 \
    test_exprs.push_back([] {                                                                \
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);                       \
        {                                                                                    \
            auto dst = jit->get_arg(0);                                                      \
            auto a = jit->get_arg(1);                                                        \
            auto b = jit->get_arg(2);                                                        \
            auto c = jit->get_arg(3);                                                        \
            auto d = jit->get_arg(4);                                                        \
            auto e = jit->get_arg(5);                                                        \
            auto f = jit->get_arg(6);                                                        \
            expr_name(dst, a, b, c, d, e, f);                                                \
            jit->return_(dst);                                                               \
            jit->finalize();                                                                 \
        }                                                                                    \
        int a = 2;                                                                           \
        int b = 3;                                                                           \
        int c = 4;                                                                           \
        int d = 5;                                                                           \
        int e = 6;                                                                           \
        int f = 7;                                                                           \
        int dst = 10;                                                                        \
        auto result = (*jit)(dst, a, b, c, d, e, f);                                         \
        expr_name(dst, a, b, c, d, e, f);                                                    \
        if (result != dst) {                                                                 \
            std::cout << #expr_name << ":" << result << " != expected " << dst << std::endl; \
            OPENVINO_ASSERT(false);                                                          \
        } else {                                                                             \
            std::cout << #expr_name << ":" << result << " == expected " << dst << std::endl; \
        }                                                                                    \
    });

#define TEST_WHILE_EXPR(cond_expr, body_expr)                                                \
    test_exprs.push_back([] {                                                                \
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);                       \
        {                                                                                    \
            auto dst = jit->get_arg(0);                                                      \
            auto a = jit->get_arg(1);                                                        \
            auto b = jit->get_arg(2);                                                        \
            auto c = jit->get_arg(3);                                                        \
            auto d = jit->get_arg(4);                                                        \
            auto e = jit->get_arg(5);                                                        \
            auto f = jit->get_arg(6);                                                        \
            jit->while_(cond_expr(dst, a, b, c, d, e, f), [&] {                              \
                body_expr(dst, a, b, c, d, e, f);                                            \
            });                                                                              \
            jit->return_(dst);                                                               \
            jit->finalize();                                                                 \
        }                                                                                    \
        int a = 2;                                                                           \
        int b = 3;                                                                           \
        int c = 4;                                                                           \
        int d = 5;                                                                           \
        int e = 6;                                                                           \
        int f = 7;                                                                           \
        int dst = 10;                                                                        \
        auto result = (*jit)(dst, a, b, c, d, e, f);                                         \
        while (cond_expr(dst, a, b, c, d, e, f)) {                                           \
            body_expr(dst, a, b, c, d, e, f);                                                \
        }                                                                                    \
        if (result != dst) {                                                                 \
            std::cout << #cond_expr << ":" << result << " != expected " << dst << std::endl; \
            OPENVINO_ASSERT(false);                                                          \
        } else {                                                                             \
            std::cout << #cond_expr << ":" << result << " == expected " << dst << std::endl; \
        }                                                                                    \
    });

#define EXPR0(dst, a, b, c, d, e, f) dst = (a + 6)
#define EXPR1(dst, a, b, c, d, e, f) dst = (a - b * 4)
#define EXPR2(dst, a, b, c, d, e, f) dst = (a * ((b << 2) ^ (c >> 1)))
#define EXPR3(dst, a, b, c, d, e, f) dst = (a + b | c - c & 8)
#define EXPR4(dst, a, b, c, d, e, f) dst = (a + b * (c + d) * 8 + e)
#define EXPR5(dst, a, b, c, d, e, f) dst = (a + b * (c - (d + e)) * 8 + e * (f - a))
#define EXPR6(dst, a, b, c, d, e, f) dst += 2
#define EXPR7(dst, a, b, c, d, e, f) dst = a + (dst * b) * 4
#define EXPR8(dst, a, b, c, d, e, f) dst = a * 3 * sizeof(float)
#define EXPR9(dst, a, b, c, d, e, f) dst = dst + 4 + 9 + a + 3 * sizeof(float) + 8
#define EXPRA(dst, a, b, c, d, e, f) dst++
#define EXPRB(dst, a, b, c, d, e, f) dst--
#define EXPRC(dst, a, b, c, d, e, f) dst = ((a > 1) || (b < 2)) && (!(f == 0))

#define COND_EXPR0(dst, a, b, c, d, e, f)  (dst == 100)
#define WHILE_EXPR0(dst, a, b, c, d, e, f) dst += 1

#define COND_EXPR1(dst, a, b, c, d, e, f)  (dst + a * 4) < (80 * f >> 2)
#define WHILE_EXPR1(dst, a, b, c, d, e, f) dst += (a >> 1)

#define COND_EXPR2(dst, a, b, c, d, e, f)  (dst == 100) || (dst == 80) && !(dst < 70)
#define WHILE_EXPR2(dst, a, b, c, d, e, f) dst += (a >> 1)

void unit_test() {
    TEST_EXPR(EXPR0);
    TEST_EXPR(EXPR1);
    TEST_EXPR(EXPR2);
    TEST_EXPR(EXPR3);
    TEST_EXPR(EXPR4);
    TEST_EXPR(EXPR5);
    TEST_EXPR(EXPR6);
    TEST_EXPR(EXPR7);
    TEST_WHILE_EXPR(COND_EXPR0, WHILE_EXPR0);
    TEST_WHILE_EXPR(COND_EXPR1, WHILE_EXPR1);
    TEST_WHILE_EXPR(COND_EXPR2, WHILE_EXPR2);
    TEST_EXPR(EXPR8);
    TEST_EXPR(EXPR9);
    TEST_EXPR(EXPRA);
    TEST_EXPR(EXPRB);
    TEST_EXPR(EXPRC);

    for (auto& func : test_exprs) {
        func();
    }
}

struct AAA {
    AAA() {
        std::cout << "Default constructor is called\n";
    }
    AAA(const AAA& rhs) {
        std::cout << "Copy constructor is called\n";
    }
    const AAA& operator=(const AAA& rhs) {
        std::cout << "Assign operator is called\n";
    }
};
void test_operator_overload() {
    AAA a;
    AAA b = a;
    AAA c[4] = {a, a, a, a};
    c[0] = a;
}

extern "C" void test() {
    unit_test();
    // test_operator_overload();
}
#endif

#if defined(__ARM_NEON)

#include "../include/misc.hpp"

extern "C" void test() {
    ECOUT("start-compiling...");
    auto jit = std::make_shared<ov::intel_cpu::SIMDJit>();
    {
        auto a = jit->get_arg(0);
        auto b = jit->get_arg(1);
        auto c = jit->get_sreg();

        // jit->add(c, a, b);
        //c = a + b;
        c *= 0x7fff1234;
        jit->return_(c);
    }
    jit->ready();
    auto f = jit->getCode<int (*)(int, int)>();
    ECOUT("start run...");
    int a = 3;
    int b = 4;
    ECOUT(a, "+", b, "=", f(a, b));
}
#endif