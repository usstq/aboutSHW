
# Floating point basic

Following image shows how floating point values are distributed on number axis. According to [wiki](https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Exponent_encoding), following picture shows how normal & subnormal values are distributed. we can see that:
 - without mantissa/significand, exponent along can represent :
   - 0          : subnormal 
   - 1~(Emax-1) : geometric sequence with common ratio of 2:
   - Emax       : infinity or NaN

In following example, geometric sequence $[\frac{1}{8}, \frac{1}{4}, \frac{1}{2}, 1]$ are encoded by exponent, and special subnormal range $[0,\frac{1}{8})$ and $[2,\infty)$  also encoded by exponent values, thus totally 6 exponent values are required (which is not friendly to binary encoding which usually has power of 2 number of values).

![Floating point values visualized](./fp_vis.jpg)

we can think of exponent value 1~(Emax-1) as encoding a sequence of length increasing geometrically, and these length are concatenated together to cover a continuous range in number axis, and significand/mantissa part encodes a series of discrete points equally distributed along each geometric length encoded by exponent.

Thus the whole floating point encoding is a combination of geometric & arithmetic sequence.

# rounding

In narrow sense, rounding means conversion from float point to integer. but broadly speaking it happens in conversion between any two representations who's element set are not fully coincides. for example, conversion between different float-point types. 

why roundTiesToEven is more accurate than Round-away-from-zero(四舍五入) taught in school? read [this](https://www.gnu.org/software/gawk/manual/html_node/Setting-the-rounding-mode.html)

different float-point representations are highly coincides with each other, for example, normal values of bfloat16 & float16 & fp8 & fp4 are subset of float32 normal values.

# Microscaling Formats (MX)

 - [OCP Microscaling Formats (MX) Specification Version 1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
 - [Microscaling Data Formats for Deep Learning](https://arxiv.org/pdf/2310.10537)

| Format Name |  Block Size | Scale Data Format | Element Data Format|
| ----------- | ----------- | ----------------- | -------------------------- |
|   MXFP4     |      32     |    E8M0 (8bit)    | E2M1 (4bit)                |

definitions of E2M1 & E8M0 are modified from IEEE 754, the expression capability wasted on infinities and NaN are avoided, we can check 5.3.3 & 5.4.1 of [OCP MX specification][1]

 - Only scale (E8M0) can represent NaN
 - Infinities are not expressed by both scales & element

LUT for E2M1 element is defined [here](https://github.com/openvinotoolkit/openvino/blob/4a5bd43723eeaff934d58e188c23629b65189778/src/core/src/type/float4_e2m1.cpp#L20). notice that no NaN nor Infinity can be encoded, so ALU can handle them easier.

E8M0 to float is defined [here](https://github.com/openvinotoolkit/openvino/blob/3056b53056d6319666f3fc250bebefb0c4b1a91e/src/core/src/type/float8_e8m0.cpp#L49), notice :
 - subnormal float are required to represent scale when exponent is -127
 - NaN is possible
 - no sign bit, thus no negative number is encoded.

As show in 6.1 of [OCP MX specification][1], dot product is done efficiently if input vectors are group aligned, thus for MatMul to benefit from this format, row of A matrix & column of B matrix must be converted to same MX format.

In case no special ALU is available, columns of B must be decompressed/converted_back to float (bf16/fp16) and A don't need to be encoded into MX formats at all.



[1]: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
