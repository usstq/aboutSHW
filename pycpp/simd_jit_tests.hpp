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

#ifdef __x86_64__
#    define ExprKPI_X64(a, b) ExprKPI(a, b)
#else
#    define ExprKPI_X64(a, b)
#endif

#ifdef __aarch64__
#    define ExprKPI_AARCH64(a, b) ExprKPI(a, b)
#else
#    define ExprKPI_AARCH64(a, b)
#endif

template <class T>
ExprKPI complex_expression(int id, T& dst, T& a, T& b, T& c, T& d, T& e, T& f) {
    switch (id) {
    case 0:
        dst = a - b * 4;
        return ExprKPI_AARCH64(0, 1) ExprKPI_X64(0, 2);
    case 1:
        dst = (a * ((b << 2) ^ (c >> 1)));
        return ExprKPI_AARCH64(0, 3) ExprKPI_X64(1, 6);
    case 2:
        dst = (a + (b | c) - c & 8);
        return ExprKPI_AARCH64(0, 4) ExprKPI_X64(0, 5);
    case 3:
        dst = (a + b * (c + d) * 8 + e);
        return ExprKPI_AARCH64(0, 4) ExprKPI_X64(0, 6);
    case 4:
        dst = (a + b * (c - (d + e)) * 8 + e * (f - a));
        return ExprKPI_AARCH64(1, 7) ExprKPI_X64(1, 10);
    case 5:
        dst = a + (dst * b) * 4;
        return ExprKPI_AARCH64(0, 2) ExprKPI_X64(0, 3);
    case 6:
        dst = a * 3 * sizeof(float);
        return ExprKPI(0, 2);
    case 7:
        dst = dst + 4 + 9 + a + 3 * sizeof(float) + 8;
        return ExprKPI(0, 3);
    case 8:
        dst = ((a > 1) || (b < 2)) && (!(f == 0));
        return ExprKPI_AARCH64(1, 6) ExprKPI_X64(1, 9);
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
                        " expr_test[",
                        id,
                        "] used ",
                        jit->expr_stat.scratch_reg_cnt,
                        " scratch registers, expecting ",
                        expect_st.scratch_reg_cnt);
        OPENVINO_ASSERT(jit->expr_stat.ops_cnt <= expect_st.ops_cnt,
                        " expr_test[",
                        id,
                        "] used ",
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
            jit->for_loop(idx, a, b, c, [&] {
                dst++;
            });
            jit->return_(dst);
        }
        jit->finalize();
        int dst = 0;
        auto result = (*jit)(dst, start, stop, step);
        OPENVINO_ASSERT(result == answer,
                        "test_for_loop(",
                        start,
                        ",",
                        stop,
                        ",",
                        step,
                        ") failed! ",
                        result,
                        " != ",
                        answer);
    };
    test_for_loop(0, 10, 1, 10);
    test_for_loop(-10, 10, 1, 20);
    test_for_loop(-8192, 8192, 1, 8192 * 2);
    test_for_loop(0, 10, 4, 2);

    auto test_for_loop2 = [&](int start, int stop, int step, int answer) {
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
        {
            auto dst = jit->get_arg(0);
            auto a = jit->get_arg(1);
            auto b = jit->get_arg(2);
            auto c = jit->get_arg(3);
            auto idx = jit->get_sreg();
            jit->for_loop(idx, start, b, step, [&] {
                dst++;
            });
            jit->return_(dst);
        }
        jit->finalize();
        int dst = 0;
        auto result = (*jit)(dst, start, stop, step);
        OPENVINO_ASSERT(result == answer,
                        "test_for_loop2(",
                        start,
                        ",",
                        stop,
                        ",",
                        step,
                        ") failed! ",
                        result,
                        " != ",
                        answer);
    };
    test_for_loop2(0, 10, 1, 10);
    test_for_loop2(-10, 10, 1, 20);
    test_for_loop2(-8192000, 8192000, 1, 8192000 * 2);
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
            jit->while_(idx + step <= stop, [&] {
                dst++;
                idx += step;
            });
            jit->return_(dst);
        }
        jit->finalize();
        int dst = 0;
        auto result = (*jit)(dst, start, stop, step);
        OPENVINO_ASSERT(result == answer,
                        "test_while(",
                        start,
                        ",",
                        stop,
                        ",",
                        step,
                        ") failed! ",
                        result,
                        " != ",
                        answer);
    };
    test_while(0, 10, 1, 10);
    test_while(-8192000, 8192000, 1, 8192000 * 2);
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
            jit->do_while_(idx + step <= stop, [&] {
                dst++;
                idx += step;
            });
            jit->return_(dst);
        }
        jit->finalize();
        int dst = 0;
        auto result = (*jit)(dst, start, stop, step);
        OPENVINO_ASSERT(result == answer,
                        "test_do_while(",
                        start,
                        ",",
                        stop,
                        ",",
                        step,
                        ") failed! ",
                        result,
                        " != ",
                        answer);
    };
    test_do_while(0, 10, 1, 10);
    test_do_while(-8192000, 8192000, 1, 8192000 * 2);
    test_do_while(0, 10, 4, 2);
    //=========================== if else =======================================
    auto test_if_else = [&](int va, int vb, int answer) {
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);
        {
            auto dst = jit->get_arg(0);
            auto a = jit->get_arg(1);
            auto b = jit->get_arg(2);
            jit->if_(
                a < b,
                [&] {
                    dst = a;
                },
                [&] {
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

extern "C" void test() {
    unit_tests();
    expr_tests();
    ctrlflow_unit_tests();
}

#include "simple_perf.hpp"

extern "C" void tput(const char* op_name, const int UNROLL) {
    LinuxPerf p({{"C", 0}, {"I", 0}});

    auto get_tput_kernel = [](std::string op, const int UNROLL) {
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>("tput");
        auto cnt = jit->get_arg(0);
        auto idx = jit->get_sreg();

        auto vr0 = jit->get_vreg();
        auto vr1 = jit->get_vreg();
        auto vr2 = jit->get_vreg();
        auto vr3 = jit->get_vreg();

        auto vr4 = jit->get_vreg();
        auto vr5 = jit->get_vreg();
        auto vr6 = jit->get_vreg();
        auto vr7 = jit->get_vreg();

        XbyakVReg v0 = vr0;
        XbyakVReg v1 = vr1;
        XbyakVReg v2 = vr2;
        XbyakVReg v3 = vr3;
        XbyakVReg v4 = vr4;
        XbyakVReg v5 = vr5;
        XbyakVReg v6 = vr6;
        XbyakVReg v7 = vr7;

        idx = 0;
        jit->do_while_(idx < cnt, [&] {
            for (int i = 0; i < UNROLL; i++) {
                if (op == "fabs") {
                    jit->fabs(v1.s4, v0.s4);
                    jit->fabs(v3.s4, v2.s4);
                    jit->fabs(v5.s4, v4.s4);
                    jit->fabs(v7.s4, v6.s4);
                }
                if (op == "fmla") {
                    jit->fmla(v1.s4, v0.s4, v0.s4);
                    jit->fmla(v3.s4, v2.s4, v2.s4);
                    jit->fmla(v5.s4, v4.s4, v4.s4);
                    jit->fmla(v7.s4, v6.s4, v6.s4);
                }
                if (op == "fadd") {
                    jit->fadd(v1.s4, v1.s4, v1.s4);
                    jit->fadd(v2.s4, v2.s4, v2.s4);
                    jit->fadd(v3.s4, v3.s4, v3.s4);
                    jit->fadd(v4.s4, v4.s4, v4.s4);
                }
            }
            idx++;
        });
        jit->return_();
        jit->finalize();
        return jit;
    };

    auto do_test = [&](std::string op, const int UNROLL) {
        auto jit = get_tput_kernel(op, UNROLL);
        // warm-up
        (*jit)(1000);

        p.start();
        (*jit)(1000);
        auto evs = p.stop();

        auto loop_count = 1000 * UNROLL * 4;
        auto cycles = (double)evs["C"] / loop_count;
        auto instructions = (double)evs["I"] / loop_count;
        printf("\e[0;36m %s %llu (ns) CPI : %.2f Instructions & Cycles: %.2f, %.2f (per iteration of %d loops) "
               "\e[0m\n",
               op.c_str(),
               evs["ns"],
               cycles / instructions,
               instructions,
               cycles,
               loop_count);
    };
    std::string op(op_name);
    if (op == "all") {
        do_test("fabs", UNROLL);
        do_test("fmla", UNROLL);
        do_test("fadd", UNROLL);
    } else {
        do_test(op, UNROLL);
    }
}
