#include "simd_jit.hpp"

std::vector<std::function<void(void)>> test_exprs;

void unit_tests() {
    auto do_test = [&](std::string op, int dst, int a, int b, int answer) {
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
        {
            auto dst = jit->get_arg(0);
            auto a = jit->get_arg(1);
            auto b = jit->get_arg(2);
            if (op == "=")
                dst = a;
            if (op == "+")
                dst = a + b;
            if (op == "-")
                dst = a - b;
            if (op == "*")
                dst = a * b;
            if (op == "&")
                dst = a & b;
            if (op == "|")
                dst = a | b;
            if (op == "^")
                dst = a ^ b;
            if (op == "&&")
                dst = a && b;
            if (op == "||")
                dst = a || b;
            if (op == "+=")
                dst += a;
            if (op == "-=")
                dst -= a;
            if (op == "*=")
                dst *= a;
            if (op == "++")
                dst++;
            if (op == "--")
                dst--;
            if (op == "n")
                dst = -a;
            if (op == "~")
                dst = ~a;
            if (op == "!")
                dst = !a;
            jit->return_(dst);
        }
        jit->finalize();
        auto result = (*jit)(dst, a, b);
        OPENVINO_ASSERT(result == answer, "unit_test ", op, " failed! ", result, " != ", answer);
        return jit;
    };
    do_test("n", 0, 1, -2, -1);
    do_test("~", 0, (int64_t)0x1234, -2, ~(int64_t)0x1234);

    do_test("=", 0, 1, 2, 1);
    do_test("=", 0, -1, 2, -1);
    do_test("+", 0, 1, 2, 3);
    do_test("+", 0, 1, -2, -1);
    do_test("+", 0, 0x11111111, 0x33333333, 0x44444444);
    do_test("-", 0, 1, 2, -1);
    do_test("-", 0, 1, -2, 3);
    do_test("-", 0, 0x33333333, 0x11111111, 0x22222222);
    do_test("*", 0, 4, 2, 8);
    do_test("*", 0, 1024, -48, -1024 * 48);
    do_test("&", 0, 0x1234, 0x1234, 0x1234);
    do_test("&", 0, 0x1234, 0x0000, 0x1234 & 0x0000);
    do_test("|", 0, 0x1234, 0x1234, 0x1234);
    do_test("|", 0, 0x1234, 0x0000, 0x1234);

    do_test("+=", 10, 1, 0, 11);
    do_test("-=", 10, 1, 0, 9);
    do_test("*=", 10, 2, 0, 20);

    do_test("++", 10, 1, 0, 11);
    do_test("--", 10, 1, 0, 9);
}

using ExprKPI = ov::intel_cpu::SIMDJit::ExprStatus;

template <class T>
ExprKPI complex_expression(int id, T& dst, T& a, T& b, T& c, T& d, T& e, T& f) {
    switch (id) {
    case 0:
        dst = a - b * 4;
        return ExprKPI(0, 1);
    case 1:
        dst = (a * ((b << 2) ^ (c >> 1)));
        return ExprKPI(0, 3);
    case 2:
        dst = (a + (b|c) - c & 8);
        return ExprKPI(0, 4);
    case 3:
        dst = (a + b * (c + d) * 8 + e);
        return ExprKPI(0, 4);
    case 4:
        dst = (a + b * (c - (d + e)) * 8 + e * (f - a));
        return ExprKPI(1, 7);
    case 5:
        dst = a + (dst * b) * 4;
        return ExprKPI(0, 2);
    case 6:
        dst = a * 3 * sizeof(float);
        return ExprKPI(0, 2);
    case 7:
        dst = dst + 4 + 9 + a + 3 * sizeof(float) + 8;
        return ExprKPI(0, 3);
    case 8:
        dst = ((a > 1) || (b < 2)) && (!(f == 0));
        return ExprKPI(1, 6);
    default:
        OPENVINO_ASSERT(false, "not supported expression id : ", id);
    }
    return ExprKPI(0, 0);
}

void expr_tests() {
    auto do_test = [&](int id) {
        std::cout << ">>>>>>>>>>>>> expr_tests " << id << " <<<<<<<<<<<<<<<" << std::endl;
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
        {
            auto dst = jit->get_arg(0);
            auto a = jit->get_arg(1);
            auto b = jit->get_arg(2);
            auto c = jit->get_arg(3);
            auto d = jit->get_arg(4);
            auto e = jit->get_arg(5);
            auto f = jit->get_arg(6);
            complex_expression(id, dst, a, b, c, d, e, f);

            jit->return_(dst);
        }
        jit->finalize();
        int64_t dst = 1;
        int64_t a = 2;
        int64_t b = 3;
        int64_t c = 4;
        int64_t d = 5;
        int64_t e = 6;
        int64_t f = 7;
        auto result = (*jit)(dst, a, b, c, d, e, f);
        auto answer = dst;
        auto expect_st = complex_expression(id, answer, a, b, c, d, e, f);
        OPENVINO_ASSERT(result == answer, "expr_test[", id, "] failed! ", result, " != ", answer);
        OPENVINO_ASSERT(jit->expr_stat.scratch_reg_cnt <= expect_st.scratch_reg_cnt,
                        " expr_test[", id, "] used ",
                        jit->expr_stat.scratch_reg_cnt,
                        " scratch registers, expecting ",
                        expect_st.scratch_reg_cnt);
        OPENVINO_ASSERT(jit->expr_stat.ops_cnt <= expect_st.ops_cnt,
                        " expr_test[", id, "] used ",
                        jit->expr_stat.ops_cnt,
                        " ops, expecting ",
                        expect_st.ops_cnt);
        return jit;
    };
    // do_test(0);
    for (int i = 0; i < 9; i++)
        do_test(i);
}

void ctrlflow_unit_tests() {
    //=========================== for loop =======================================
    auto test_for_loop = [&](int start, int stop, int step, int answer) {
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
        {
            auto dst = jit->get_arg(0);
            auto a = jit->get_arg(1);
            auto b = jit->get_arg(2);
            auto c = jit->get_arg(3);
            auto idx = jit->get_sreg();
            jit->for_loop(idx, a, b, c, [&]{
                dst ++;
            });
            jit->return_(dst);
        }
        jit->finalize();
        int dst = 0;
        auto result = (*jit)(dst, start, stop, step);
        OPENVINO_ASSERT(result == answer, "test_for_loop(", start, ",", stop, ",", step, ") failed! ", result, " != ", answer);
    };
    test_for_loop(0, 10, 1, 10);
    test_for_loop(-10, 10, 1, 20);
    test_for_loop(-8192, 8192, 1, 8192*2);
    test_for_loop(0, 10, 4, 2);

    auto test_for_loop2 = [&](int start, int stop, int step, int answer) {
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
        {
            auto dst = jit->get_arg(0);
            auto a = jit->get_arg(1);
            auto b = jit->get_arg(2);
            auto c = jit->get_arg(3);
            auto idx = jit->get_sreg();
            jit->for_loop(idx, start, b, step, [&]{
                dst ++;
            });
            jit->return_(dst);
        }
        jit->finalize();
        int dst = 0;
        auto result = (*jit)(dst, start, stop, step);
        OPENVINO_ASSERT(result == answer, "test_for_loop2(", start, ",", stop, ",", step, ") failed! ", result, " != ", answer);
    };
    test_for_loop2(0, 10, 1, 10);
    test_for_loop2(-10, 10, 1, 20);
    test_for_loop2(-8192000, 8192000, 1, 8192000*2);
    test_for_loop2(0, 10, 4, 2);
    //=========================== while_ =======================================
    auto test_while = [&](int start, int stop, int step, int answer) {
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
        {
            auto dst = jit->get_arg(0);
            auto a = jit->get_arg(1);
            auto b = jit->get_arg(2);
            auto c = jit->get_arg(3);
            auto idx = jit->get_sreg();
            idx = start;
            jit->while_(idx + step <= stop, [&]{
                dst ++;
                idx += step;
            });
            jit->return_(dst);
        }
        jit->finalize();
        int dst = 0;
        auto result = (*jit)(dst, start, stop, step);
        OPENVINO_ASSERT(result == answer, "test_while(", start, ",", stop, ",", step, ") failed! ", result, " != ", answer);
    };
    test_while(0, 10, 1, 10);
    test_while(-8192000, 8192000, 1, 8192000*2);
    test_while(0, 10, 4, 2);
    //=========================== do_while =======================================
    auto test_do_while = [&](int start, int stop, int step, int answer) {
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
        {
            auto dst = jit->get_arg(0);
            auto a = jit->get_arg(1);
            auto b = jit->get_arg(2);
            auto c = jit->get_arg(3);
            auto idx = jit->get_sreg();
            idx = start;
            jit->do_while_(idx + step <= stop, [&]{
                dst ++;
                idx += step;
            });
            jit->return_(dst);
        }
        jit->finalize();
        int dst = 0;
        auto result = (*jit)(dst, start, stop, step);
        OPENVINO_ASSERT(result == answer, "test_do_while(", start, ",", stop, ",", step, ") failed! ", result, " != ", answer);
    };
    test_do_while(0, 10, 1, 10);
    test_do_while(-8192000, 8192000, 1, 8192000*2);
    test_do_while(0, 10, 4, 2);
    //=========================== if else =======================================
    auto test_if_else = [&](int va, int vb, int answer) {
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
        {
            auto dst = jit->get_arg(0);
            auto a = jit->get_arg(1);
            auto b = jit->get_arg(2);
            jit->if_(a < b, [&]{
                dst = a;
            }, [&]{
                dst = b;
            });
            jit->return_(dst);
        }
        jit->finalize();
        int dst = 0;
        auto result = (*jit)(dst, va, vb);
        OPENVINO_ASSERT(result == answer, "test_if_else(", va, ",", vb, ") failed! ", result, " != ", answer);
    };
    test_if_else(10, 20, 10);
    test_if_else(20, 10, 10);
}

#if defined(__AVX2__) || defined(__AVX512F__)

#    define TEST_EXPR(expr_name)                                                                 \
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

#    define TEST_WHILE_EXPR(cond_expr, body_expr)                                                \
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

#    define EXPR0(dst, a, b, c, d, e, f) dst = (a + 6)
#    define EXPR1(dst, a, b, c, d, e, f) dst = (a - b * 4)
#    define EXPR2(dst, a, b, c, d, e, f) dst = (a * ((b << 2) ^ (c >> 1)))
#    define EXPR3(dst, a, b, c, d, e, f) dst = (a + b | c - c & 8)
#    define EXPR4(dst, a, b, c, d, e, f) dst = (a + b * (c + d) * 8 + e)
#    define EXPR5(dst, a, b, c, d, e, f) dst = (a + b * (c - (d + e)) * 8 + e * (f - a))
#    define EXPR6(dst, a, b, c, d, e, f) dst += 2
#    define EXPR7(dst, a, b, c, d, e, f) dst = a + (dst * b) * 4
#    define EXPR8(dst, a, b, c, d, e, f) dst = a * 3 * sizeof(float)
#    define EXPR9(dst, a, b, c, d, e, f) dst = dst + 4 + 9 + a + 3 * sizeof(float) + 8
#    define EXPRA(dst, a, b, c, d, e, f) dst++
#    define EXPRB(dst, a, b, c, d, e, f) dst--
#    define EXPRC(dst, a, b, c, d, e, f) dst = ((a > 1) || (b < 2)) && (!(f == 0))

#    define COND_EXPR0(dst, a, b, c, d, e, f)  (dst == 100)
#    define WHILE_EXPR0(dst, a, b, c, d, e, f) dst += 1

#    define COND_EXPR1(dst, a, b, c, d, e, f)  (dst + a * 4) < (80 * f >> 2)
#    define WHILE_EXPR1(dst, a, b, c, d, e, f) dst += (a >> 1)

#    define COND_EXPR2(dst, a, b, c, d, e, f)  (dst == 100) || (dst == 80) && !(dst < 70)
#    define WHILE_EXPR2(dst, a, b, c, d, e, f) dst += (a >> 1)

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

#    include "../include/misc.hpp"

extern "C" void test() {
    //unit_tests();
    //expr_tests();
    ctrlflow_unit_tests();
}
#endif