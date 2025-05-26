/*
* Copyright (c) 2020-2023, Intel Corporation
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
* OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/

#include <iostream>
#include <cassert>
#include <math.h>
#include <vector>

#include <CL/cl.h>
#include "float16.hpp"
#include "tensor.hpp"
#include "common.hpp"

#if defined(BINNAME)
# define KERNEL BINNAME
#endif

#define SZ 160
#define KERNEL_SZ 16
#define CHECK(a) do { \
    err = (a); \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "FAIL: err=%d @ line=%d (%s)\n", err, __LINE__, (#a)); \
        exit(err); \
    } \
}while (0)
#define CHECK2(a) do { \
    (a); \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "FAIL: err=%d @ line=%d (%s)\n", err, __LINE__, (#a)); \
        exit(err); \
    } \
}while (0)
#ifndef KERNEL
#error "Error: KERNEL must be defined with location of kernel binary"
#endif

using namespace ov;

void matmul(const PlainTensor& a, const PlainTensor& b, PlainTensor& c) {
    PlainTensor tmp;
    tmp.resize<float>({ c.size(0), c.size(1) });
    tmp = 0.0f;
    for (int i = 0; i < c.size(0); ++i) {
        for (int j = 0; j < c.size(1); ++j) {
            for (int k = 0; k < a.size(1); ++k) {
                tmp.ptr<float>(i)[j] += a.ptr<float16_t>(i)[k] * b.ptr<float16_t>(k)[j];
            }
        }
    }
    for (int i = 0; i < c.size(0); ++i) {
        for (int j = 0; j < c.size(1); ++j) {
            c.ptr<float16_t>(i)[j] = tmp.ptr<float>(i)[j];
        }
    }
}

void repack_f16(int K, int N, float16_t* src, float16_t* dst) {
    const int DEPTH = 8;
    const int BLOCK_REG_M = 8;
    const int BLOCK_REG_N = x::SG_SIZE;
    // half is 2, u8 is 4
    const int VNNI = 2;
    const int BLOCK_REG_K = 32 / VNNI;
    const int BLOCK_WG_M = x::BLOCK_SG_M * x::SG_M;
    const int BLOCK_WG_N = x::BLOCK_SG_N * x::SG_N;
    // register blocking
    const int REG_M = x::BLOCK_SG_M / BLOCK_REG_M;
    const int REG_N = x::BLOCK_SG_N / BLOCK_REG_N;
    // packed memory layout:
    struct reg {                                        // --> one DPAS op
        float16_t data[BLOCK_REG_K/2][BLOCK_REG_N*2];   // BLOCK_REG_K=16, BLOCK_REG_N=8, aka half[8,16]
    };
    struct sg_block {                                   // --> one EU
        reg regs[x::BLOCK_WG_K/BLOCK_REG_K][REG_N];     // REG_N==2, BLOCK_WG_K/BLOCK_REG_K==4
    };
    struct wg_block {                                   // --> one WG
        sg_block block[x::SG_N];
    };
    //struct weight {                                   // --> weight
    //    wg_block wg[WG_N][K/BLOCK_WG_K];
    //};

    const int WG_N = N / BLOCK_WG_N;
    const int WG_K = K / x::BLOCK_WG_K;
    wg_block* wg_block_ptr = (wg_block*)dst;
    for (int wg_n = 0; wg_n < WG_N; wg_n++) {
        for (int wg_k = 0; wg_k < WG_K; wg_k++) {
            wg_block& wg = *wg_block_ptr++;
            auto src_wg_block = src + wg_k * x::BLOCK_WG_K * N + wg_n * BLOCK_WG_N;
            for (int sg_n = 0; sg_n < x::SG_N; sg_n++) {
                sg_block& sg = wg.block[sg_n];
                auto src_sg_block = src_wg_block + sg_n * x::BLOCK_SG_N;
                for (int reg_k = 0; reg_k < x::BLOCK_WG_K / BLOCK_REG_K; reg_k++) {
                    for (int reg_n = 0; reg_n < REG_N; reg_n++) {
                        reg& cur_reg = sg.regs[reg_k][reg_n];
                        auto src_reg = src_sg_block + reg_k * BLOCK_REG_K * N + reg_n * BLOCK_REG_N;
                        for (int k = 0; k < BLOCK_REG_K / 2; k++) {
                            for (int n = 0; n < BLOCK_REG_N; n++) {
                                cur_reg.data[k][2 * n + 0] = *(src_reg + (k * 2 + 0) * N + n);
                                cur_reg.data[k][2 * n + 1] = *(src_reg + (k * 2 + 1) * N + n);
                            }
                        }
                    }
                }
            }
        }
    }
}

void fill(PlainTensor& t) {
    uint32_t M = t.size(0), N = t.size(1);
    float16_t* p = (float16_t*)t.m_ptr.get();
    for (uint32_t i = 0; i < M * N; i += 5) {
#define SET(idx, val) if (i + idx < M * N) p[i + idx] = val;
        SET(0, -2.0f);
        SET(1, -1.0f);
        SET(2, 0.0f);
        SET(3, 1.0f);
        SET(4, 2.0f);
#undef SET
    }
}

int main( int argc, char* argv[])
{
    PlainTensor a, b, c_ref, c, b_repack_ref, b_repack;
    uint32_t M = 128*2, N = 128*2, K = 64*2;
    a.resize<float16_t>({ M, K });
    b.resize<float16_t>({ K, N });
    b_repack_ref.resize<float16_t>({ K, N });
    b_repack.resize<float16_t>({ K, N });
    c_ref.resize<float16_t>({ M, N });
    c.resize<float16_t>({ M, N });

    //a = float16_t{ 1.0f };
    fill(a);
    //b = float16_t{ 1.0f };
    fill(b);
    repack_f16(K, N, (float16_t*)b.m_ptr.get(), (float16_t*)b_repack_ref.m_ptr.get());
    matmul(a, b, c_ref);

    // initialize GPU
    cl_platform_id platform;  // OpenCL platform
    cl_device_id device;      // device ID
    cl_context context;       // context
    cl_command_queue queue;   // command queue
    cl_program program;       // program
    cl_kernel kernel_gemm;    // kernel
    cl_kernel kernel_repack;  // kernel
    cl_int err;

    CHECK(clGetPlatformIDs(1, &platform, NULL));
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    CHECK2(context = clCreateContext(NULL, 1, &device, NULL, NULL, &err));
    CHECK2(queue = clCreateCommandQueueWithProperties(context, device, 0, &err));

    // diagnostic info
    char name_buffer[256];
    CHECK(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name_buffer), name_buffer, NULL));
    fprintf(stderr, "INFO: using platform: %s\n", name_buffer);
    CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name_buffer), name_buffer, NULL));
    fprintf(stderr, "INFO: using device: %s\n", name_buffer);

    // read in and initialize kernel
    FILE *fp = fopen(KERNEL, "rb");
    if (fp == NULL) {
        fprintf(stderr, "FAIL: unable to open %s\n", KERNEL);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    size_t sz = ftell(fp);
    rewind(fp);

    unsigned char *code = (unsigned char *)malloc(sz);
    fread(code, 1, sz, fp);
    fclose(fp);

    cl_int errNum = 0;
    const unsigned char *codes[1] = {code};
    size_t sizes[1] = {sz};
    CHECK2(program = clCreateProgramWithBinary(context, 1, &device, sizes, codes, &err, &errNum));
    CHECK(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
    CHECK2(kernel_gemm = clCreateKernel(program, "gemm", &err));
    CHECK2(kernel_repack = clCreateKernel(program, "repack_f16", &err));

    // gemm
    // kernel parameter initialization
    // void gemm(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t dst ATTR, uint M, uint N, uint K, uint lda, uint ldb, uint ldc)
    cl_mem d_a, d_b, d_c;
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, M * K * sizeof(float16_t), NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, K * N * sizeof(float16_t), NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, M * N * sizeof(float16_t), NULL, NULL);
    CHECK(clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, M * K * sizeof(float16_t), a.m_ptr.get(), 0, NULL, NULL));
    CHECK(clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, K * N * sizeof(float16_t), b_repack_ref.m_ptr.get(), 0, NULL, NULL));
    CHECK(clSetKernelArg(kernel_gemm, 0, sizeof(cl_mem), &d_a));
    CHECK(clSetKernelArg(kernel_gemm, 1, sizeof(cl_mem), &d_b));
    CHECK(clSetKernelArg(kernel_gemm, 2, sizeof(cl_mem), &d_c));
    CHECK(clSetKernelArg(kernel_gemm, 3, sizeof(M), &M));
    CHECK(clSetKernelArg(kernel_gemm, 4, sizeof(N), &N));
    CHECK(clSetKernelArg(kernel_gemm, 5, sizeof(K), &K));
    CHECK(clSetKernelArg(kernel_gemm, 6, sizeof(K), &K));
    CHECK(clSetKernelArg(kernel_gemm, 7, sizeof(N), &N));
    CHECK(clSetKernelArg(kernel_gemm, 8, sizeof(N), &N));

    // send to GPU
    size_t BLOCK_SG_M = x::BLOCK_SG_M;
    size_t BLOCK_SG_N = x::BLOCK_SG_N;
    size_t SG_M = x::SG_M, SG_N = x::SG_N;
    size_t globalSize[2] = { M / BLOCK_SG_M, N / BLOCK_SG_N };
    size_t localSize[2] = { SG_M, SG_N };
    CHECK(clEnqueueNDRangeKernel(queue, kernel_gemm, 2, NULL, globalSize, localSize, 0, NULL, NULL));
    clFinish(queue);

    // repack
    // void repack(int K, int N, half* src ATTR, half* dst ATTR)
    cl_mem d_b_org, d_b_pack;
    {
        d_b_org = clCreateBuffer(context, CL_MEM_READ_ONLY, K * N * sizeof(float16_t), NULL, NULL);
        d_b_pack = clCreateBuffer(context, CL_MEM_WRITE_ONLY, K * N * sizeof(float16_t), NULL, NULL);
        CHECK(clEnqueueWriteBuffer(queue, d_b_org, CL_TRUE, 0, N * K * sizeof(float16_t), b.m_ptr.get(), 0, NULL, NULL));
        CHECK(clSetKernelArg(kernel_repack, 0, sizeof(K), &K));
        CHECK(clSetKernelArg(kernel_repack, 1, sizeof(N), &N));
        CHECK(clSetKernelArg(kernel_repack, 2, sizeof(cl_mem), &d_b_org));
        CHECK(clSetKernelArg(kernel_repack, 3, sizeof(cl_mem), &d_b_pack));

        // send to GPU
        size_t BLOCK_SG_M = x::BLOCK_SG_M;
        size_t BLOCK_SG_N = x::BLOCK_SG_N;
        size_t SG_M = x::SG_M, SG_N = x::SG_N;
        size_t BLOCK_WG_N = BLOCK_SG_N * SG_N;
        size_t globalSize[2] = { N / BLOCK_WG_N, K / x::BLOCK_WG_K };
        size_t localSize[2] = { 1, 1 };
        CHECK(clEnqueueNDRangeKernel(queue, kernel_repack, 2, NULL, globalSize, localSize, 0, NULL, NULL));
        clFinish(queue);
    }

    // process output and cleanup
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, M * N * sizeof(float16_t), c.m_ptr.get(), 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_b_pack, CL_TRUE, 0, K * N * sizeof(float16_t), b_repack.m_ptr.get(), 0, NULL, NULL);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel_gemm);
    clReleaseKernel(kernel_repack);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    auto cmp = [](const char* prefix, PlainTensor& cur_t, PlainTensor& ref_t) {
        for (int i = 0; i < cur_t.size(0); ++i) {
            for (int j = 0; j < cur_t.size(1); ++j) {
                float cur = cur_t.ptr<float16_t>(i)[j];
                float ref = ref_t.ptr<float16_t>(i)[j];
                if (std::abs(cur - ref) > 0.01f) {
                    fprintf(stderr, "FAIL: comparison '%s' at index[%d, %d], cur: %f, ref: %f\n", prefix, i, j, cur, ref);
                    exit(-1);
                }
            }
        }
    };
    // verify results
    cmp("result", c, c_ref);
    cmp("pack", b_repack, b_repack_ref);

    fprintf(stderr, "PASSED\n");
    return 0;
}
