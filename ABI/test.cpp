#include <vector>
#include <string>
#include <iostream>

class __attribute__((__visibility__("default"))) XXX {
    XXX() {
        std::cout << "XXX" << std::endl;
    }
    std::string sxx;
    std::string to_string();
    double to_float();
    std::vector<double> to_vector_of_double();
    std::vector<std::string> to_vector_of_string();
    void use_string_arg(std::string x);
    void use_string_inside(int x);
};

std::string XXX::to_string() {
    return "HSSS";
}
double XXX::to_float() {
    return 1.234;
}
std::vector<double> XXX::to_vector_of_double() {
    return {1.23, 2.344};
}
std::vector<std::string> XXX::to_vector_of_string() {
    return {"hello", "world"};
}
void XXX::use_string_arg(std::string x) {
    std::cout << x;
}
void XXX::use_string_inside(int x) {
    std::string xs = std::to_string(x);
    std::cout << xs;
}

void __attribute__((__visibility__("hidden"))) test() {
    std::cout << "test" << std::endl;
}