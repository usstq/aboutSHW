
#define CL_HPP_ENABLE_EXCEPTIONS

#include "common.hpp"

int main() {
    tensorND<float> t;
    t.resize({2,3,4});
    t.at(0,0,1) = 1.11f;
    t.at(0,1,2) = 1.22f;
    t.at(1,2,3) = 1.23f;
    std::cout << t.repr(3,6) << std::endl;
    return 0;
}