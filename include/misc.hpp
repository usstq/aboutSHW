
#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>

//========================================================================
// ASSERT
#define ASSERT(cond) if (!(cond)) {\
    std::stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
    throw std::runtime_error(ss.str()); \
}

//========================================================================
// ECOUT
template<int id = 0>
inline float get_delta_ms() {
    static auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    t0 = t1;
    return dt.count();
}

template <typename... Ts>
void easy_cout(const char* file, const char* func, int line, Ts... args) {
    std::string file_path(file);
    std::string file_name(file);

    auto last_sep = file_path.find_last_of('/');
    if (last_sep == std::string::npos)
        last_sep = file_path.find_last_of('\\');
    if (last_sep != std::string::npos)
        file_name = file_path.substr(last_sep + 1);

    std::string file_name_with_line = file_name + ":" + std::to_string(line);
    auto tag = file_name_with_line + " " + func + "()";

    std::stringstream ss;
    int dummy[sizeof...(Ts)] = {(ss << args, 0)...};
    auto dt_value = get_delta_ms();
    std::string dt_unit = "ms";
    if (dt_value > 1000.0f) {
        dt_value /= 1000.0f;
        dt_unit = "sec";
        if (dt_value > 60.0f) {
            dt_value /= 60.0f;
            dt_unit = "min";
        }
    }
    std::cout << " \033[37;100m+" << std::fixed << std::setprecision(3) << dt_value << " " << dt_unit << "\033[36;40m " << tag << " \033[0m " << ss.str() << "" << std::endl;
}

#define ECOUT(...) easy_cout(__FILE__, __func__, __LINE__, __VA_ARGS__)


//===============================================================
// getenv
inline int64_t getenv(const char * var, int64_t default_value) {
    static std::map<std::string, int64_t> envs;
    if (envs.count(var) == 0) {
        const char * p = std::getenv(var);
        if (p) {
            char str_value[256];
            int len = 0;
            while(p[len] >= '0' && p[len] <= '9') {
                str_value[len] = p[len];
                len++;
            }
            str_value[len] = 0;

            char unit = p[len];
            int64_t unit_value = 1;
            // has unit?
            if (unit == 'K' || unit == 'k') unit_value = 1024;
            if (unit == 'M' || unit == 'm') unit_value = 1024*1024;
            if (unit == 'G' || unit == 'g') unit_value = 1024*1024*1024;

            default_value = std::atoi(str_value) * unit_value;
        }
        printf("\033[32mENV:\t %s = %lld %s\033[0m\n", var, default_value, p?"":"(default)");
        envs[var] = default_value;
    }
    return envs[var];
}

static std::vector<std::string> str_split(const std::string& s, std::string delimiter) {
    std::vector<std::string> ret;
    size_t last = 0;
    size_t next = 0;
    if (s.empty()) return ret;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        std::cout << last << "," << next << "=" << s.substr(last, next-last) << "\n";
        ret.push_back(s.substr(last, next-last));
        last = next + 1;
    }
    ret.push_back(s.substr(last));
    return ret;
}

// multiple values separated by ,
inline std::vector<int>& getenvs(const char * var, size_t count = 0, int default_v = 0) {
    static std::map<std::string, std::vector<int>> envs;
    
    if (envs.count(var) == 0) {
        std::vector<int> ret;
        const char * p = std::getenv(var);
        if (p) {
            auto vec = str_split(p, ",");
            for(auto& v : vec)
                ret.push_back(std::atoi(v.c_str()));
        }
        while(ret.size() < count)
            ret.push_back(default_v);
        printf("\033[32mENV:\t %s = ", var);
        const char * sep = "";
        for(int v : ret) {
            printf("%s%d", sep, v);
            sep = ",";
        }
        printf("\033[0m\n");
        envs[var] = ret;
    }
    return envs[var];
}

