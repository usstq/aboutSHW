# ABI (Application Binary interface)
 - [What Is an ABI, and Why Should You Care? - Shung-Hsi Yu, SUSE](https://www.youtube.com/watch?v=90fwlfon3Hk)
 - [什么是应用程序二进制接口ABI
](https://zhuanlan.zhihu.com/p/386106883)

How one binary (application) links (dynamically) and calls another existing binary (library or system call).

Typical ABI related scenarios:
 - Linking errors : Undefined Symbol
 - name demangle
 - call-convention
 - Kernel system call convention & behavior
 - Procfs/sysfs behavior
 - C Data-structure definition
 - C++Object
 - Library backward compatibility
 - Symbol versioning

API is language specific concept, and since assembly is the closest to binary, so some ABI concept can be understood by assembly. All concept of APIs are mapped into ABI finally

 - Usually Compiler-backend-developer/OS developer will more involved
 - C++ ABI is much more complicated than C

## ABI in STL
[Dual ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)
```bash
# by-default, function return a string-related datatype will have [abi:cxx11], use 
# string as parameter will have `std::__cxx11` prefix, use string internally has 
# nothing special, these extra symbol naming decorations make sure that when
# caller pass string/list object to or from our lib, they are expecting the
# same C++11 standard string/list implementation.
#
$ gcc -std=c++11 -fPIC -shared test.cpp -o libtest.so && nm -g  -C -D ./libtest.so | grep XXX::
0000000000006298 T XXX::use_string_arg(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
00000000000062c6 T XXX::use_string_inside(int)
000000000000602e T XXX::to_vector_of_double()
00000000000060fe T XXX::to_vector_of_string[abi:cxx11]()
000000000000600e T XXX::to_float()
0000000000005f7a T XXX::to_string[abi:cxx11]()

# -D_GLIBCXX_USE_CXX11_ABI=0 can remove this ABI symbol name changes
$ gcc -D_GLIBCXX_USE_CXX11_ABI=0  -std=c++11 -fPIC -shared test.cpp -o libtest.so && nm -g  -C -D ./libtest.so | grep XXX::
000000000000528c T XXX::use_string_arg(std::string)
00000000000052ba T XXX::use_string_inside(int)
000000000000502e T XXX::to_vector_of_double()
00000000000050fe T XXX::to_vector_of_string()
000000000000500e T XXX::to_float()
0000000000004f7a T XXX::to_string()


```

## inline namespace
inline namespace is a tool for C++ ABI: https://www.foonathan.net/2018/11/inline-namespaces/


