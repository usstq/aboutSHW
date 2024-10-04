
#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <cstdlib>

//========================================================================
// ASSERT
#ifndef ASSERT
#define ASSERT(cond) if (!(cond)) {\
    std::stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
    throw std::runtime_error(ss.str()); \
}
#endif

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

//========================================================================
// ostream
template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& nd) {
    os << "std::array<T=" << typeid(nd[0]).name() << ", N=" << N << ">{";
    const char * sep = "";
    for (int i= 0; i < N; i++) {
        os << sep << nd[i];
        sep = ",";
    }
    os << "}";
    return os;
}

//========================================================================
// allocate
template<typename T>
std::shared_ptr<T> alloc_cache_aligned(int count, T default_value) {
    auto ret = std::shared_ptr<T>(
#ifdef __NVCC__
            reinterpret_cast<T*>(malloc(count * sizeof(T))),
#else
            reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))),
#endif
            [](void * p) { ::free(p); });
    
    for(int i = 0; i < count; i++) {
        ret.get()[i] = default_value;
    }
    return ret;
}

template<typename T>
std::shared_ptr<T> alloc_cache_aligned(int count) {
    auto ret = std::shared_ptr<T>(
#ifdef __NVCC__
            reinterpret_cast<T*>(malloc(count * sizeof(T))),
#else
            reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))),
#endif
            [](void * p) { ::free(p); });
    return ret;
}

//========================================================================
// tensorND
template <typename T, int TENSORND_MAXDIMS=8>
struct tensorND {
    size_t m_shape[TENSORND_MAXDIMS] = {0};
    size_t m_strides[TENSORND_MAXDIMS] = {0};
    size_t m_ndims = 0;
    size_t m_offset = 0;
    size_t m_numel = 0;
    std::shared_ptr<T> m_data;

    bool operator==(const tensorND<T, TENSORND_MAXDIMS>& rhs) const {
        if (m_numel != rhs.m_numel)
            return false;

        for(int i = 0; i < m_ndims; i++)
            if (m_shape[i] != rhs.m_shape[i])
                return false;

        auto * p_lhr = m_data.get();
        auto * p_rhr = rhs.m_data.get();
        
        auto coord0 = get_coordinate();
        auto coord1 = rhs.get_coordinate();
        for(size_t i = 0; i < m_numel; i++) {
            auto value0 = m_data.get()[coord0.offset()];
            auto value1 = rhs.m_data.get()[coord1.offset()];
            if (value0 != value1) {
                std::cout << "mismatch at " << coord0.to_string() << "  " << value0 << " != " << value1 << std::endl;
                return false;
            }
            coord0 += 1;
            coord1 += 1;
        }
        return true;
    }

    size_t ndims() const { return m_ndims; }
    size_t numel() const  { return m_numel; }
    size_t size(int i) const {
        while (i < 0) i += m_ndims;
        while (i >= m_ndims) i -= m_ndims;
        return m_shape[i];
    }
    const size_t* shape() const { return m_shape; }
    const size_t* strides() const { return m_strides; }

    // coordinate (can be designed as iterator too)
    struct coordinate {
        const tensorND<T, TENSORND_MAXDIMS>* m_tensor;
        int64_t m_value[TENSORND_MAXDIMS] = {0};
        size_t m_offset = 0;

        size_t offset() { return m_offset; }
        int64_t operator[](int i) {
            auto ndims = m_tensor->ndims();
            if (i < 0) i += ndims;
            return m_value[i];
        }

        coordinate(tensorND<T, TENSORND_MAXDIMS>* p_tensor, const std::vector<int>& idx = {}) : m_tensor(p_tensor) {
            for(int i = 0; i < TENSORND_MAXDIMS; i++) {
                if (i < idx.size())
                    m_value[i] = idx[i];
                else
                    m_value[i] = 0;
            }
        }
        coordinate& operator+=(int64_t carry) {
            auto* shape = m_tensor->shape();
            auto ndims = m_tensor->ndims();
            for (int r = ndims - 1; r >= 0; r--) {
                m_value[r] += carry;
                carry = 0;

                assert(m_value[r] >= 0);
                // get carry for next-higher digit
                while(m_value[r] >= shape[r]) {
                    m_value[r] -= shape[r];
                    carry ++;
                }

                if (carry == 0) break;
            }
            // update offset
            auto* strides = m_tensor->strides();
            m_offset = 0;
            for (int r = 0; r < ndims; r++) {
                m_offset += m_value[r] * strides[r];
            }
            return *this;
        }

        std::string to_string() {
            auto ndims = m_tensor->ndims();
            std::stringstream ss;
            ss << "[";
            const char *sep = "";
            for(int n = 0; n < ndims; n++) {
                ss << sep << m_value[n];
                sep = ",";
            }
            ss << "]";
            return ss.str();
        }
    };

    coordinate get_coordinate() {
        return coordinate(this);
    }

    void resize(const std::vector<size_t>& dims) {
        assert(dims.size() <= TENSORND_MAXDIMS);
        m_ndims = dims.size();
        size_t s = 1;
        m_numel = 1;
        for(int64_t i = static_cast<int64_t>(m_ndims - 1); i >= 0; i--) {
            m_shape[i] = dims[i];
            m_strides[i] = s;
            s *= m_shape[i];
            m_numel *= m_shape[i];
        }
        m_data = alloc_cache_aligned<T>(m_numel);
    }

    void resize(const std::vector<size_t>& dims, T init) {
        resize(dims);
        auto* ptr = m_data.get();
        for(int i = 0; i < m_numel; i++) ptr[i] = init;
    }

    tensorND() = default;

    tensorND(const std::initializer_list<size_t>& dims) {
        resize(dims);
    }
    tensorND(const std::initializer_list<size_t>& dims, T init) {
        resize(dims, init);
    }
//===================== OpenCL ==========================
#ifdef CL_TARGET_OPENCL_VERSION
    cl::Buffer ocl_buffer;
    bool m_on_gpu;
    cl::Buffer& to_gpu() {
        if (ocl_buffer() == NULL) {
            ocl_buffer = cl::Buffer(m_data.get(), m_data.get() + m_numel, false);
            m_on_gpu = false;
        }
        if (!m_on_gpu) {
            cl::copy(m_data.get(), m_data.get() + m_numel, ocl_buffer);
            m_on_gpu = true;
        }
        return ocl_buffer;
    }
    T* to_cpu() {
        if (m_on_gpu) {
            cl::copy(ocl_buffer, m_data.get(), m_data.get() + m_numel);
            m_on_gpu = false;
        }
        return m_data.get();
    }
#endif
//===================== CUDA ==========================
#ifdef __NVCC__
    std::shared_ptr<T> m_dev_buff;
    bool m_on_gpu;
    T* to_gpu() {
        if (!m_dev_buff) {
            T* ptr;
            ASSERT(cudaMalloc((void**)&ptr, m_numel * sizeof(T)) == cudaSuccess);
            m_dev_buff = std::shared_ptr<T>(ptr, [](void * p){cudaFree(p);});
            m_on_gpu = false;
        }
        if (!m_on_gpu) {
            ASSERT(cudaMemcpy(m_dev_buff.get(), m_data.get(), m_numel * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
            m_on_gpu = true;
        }
        return m_dev_buff.get();
    }

    T* to_cpu() {
        // maybe we need wait for device code to finish the work here?
        if (m_on_gpu) {
            ASSERT(cudaDeviceSynchronize() == cudaSuccess);
            ASSERT(cudaMemcpy(m_data.get(), m_dev_buff.get(), m_numel * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
            m_on_gpu = false;
        }
        return m_data.get();
    }
#endif

    tensorND<T> clone() {
        // deep copy
        tensorND<T> ret;
        memcpy(ret.m_shape, m_shape, sizeof(m_shape));
        memcpy(ret.m_strides, m_strides, sizeof(m_strides));
        ret.m_ndims = m_ndims;
        ret.m_offset = m_offset;
        ret.m_numel = m_numel;
        ret.m_data = alloc_cache_aligned<T>(m_numel);
        memcpy(ret.m_data.get(), m_data.get(), m_numel * sizeof(T));
        return ret;
    }

    // reference element using index directly
    template <int dim>
    int64_t offset() const {
        return m_offset;
    }
    template <int dim, typename I>
    int64_t offset(I i) const {
        return m_offset + i * m_strides[dim];
    }
    template <int dim, typename I, typename... Is>
    int64_t offset(I i, Is... indices) const {
        return i * m_strides[dim] + offset<dim + 1>(indices...);
    }
    template <typename... Is>
    T* ptr(Is... indices) const {
        return reinterpret_cast<T*>(m_data.get()) + offset<0>(indices...);
    }
    /*
    template <typename... Is>
    void* ptr_v(Is... indices) const {
        return reinterpret_cast<void*>(m_data.get() + offset<0>(indices...) * m_element_size);
    }*/
    template <typename... Is>
    T& at(Is... indices) const {
#if 0
        std::cout << "at" << " ("; 
        int dummy[sizeof...(Is)] = {(std::cout << indices << ",", 0)...};
        std::cout << ") offset=" << offset<0>(indices...) << "" << std::endl;
#endif
        return *(m_data.get() + offset<0>(indices...));
    }
    //T& operator[](const coordinate<T, TENSORND_MAXDIMS>& coord) {
    //    return m_data.get()[coord.offset()];
    //}


    std::string repr(int precision = 0, int width = 5) {
        std::stringstream ss;
        auto* pdata = m_data.get();
        ss << " shape=(";
        const char * sep = "";
        for(int i = 0; i < m_ndims; i++) {
            ss << sep << m_shape[i];
            sep = ",";
        }
        ss << ") strides=(";
        sep = "";
        for(int i = 0; i < m_ndims; i++) {
            ss << sep << m_strides[i];
            sep = ",";
        }
        ss << ") dtype=" << typeid(*pdata).name() << " [\n";
        if (precision >= 0) ss << std::setprecision(precision);

        auto coord = get_coordinate();

        for(int i = 0; i < m_numel; ) {
            if (m_ndims > 1) {
                ss << " [";
                sep = "";
                for(int n = 0; n < m_ndims - 1; n++) {
                    ss << sep << coord[n];
                    sep = ",";
                }
                ss << ", ...] : ";
            }
            for(int n = 0; n < m_shape[m_ndims - 1]; n++) {
                if (width >= 0)
                    ss << std::fixed << std::setw(width);
                ss << m_data.get()[coord.offset()];
                // std::cout << coord.offset() << "\n";
                coord += 1;
                i += 1;
            }
            ss << "\n";
        }
        ss << "]";
        return ss.str();
    }
};


template<class T>
struct CArray {
    const T* data;
    std::size_t N;
    CArray(const T* data, std::size_t N) : data(data), N(N) {}
};

template<class T>
CArray<T> carray(const T* data, std::size_t N) {
    return CArray<T>(data, N);
}

template<class T>
CArray<T> carray(const tensorND<T>& t) {
    return CArray<T>(t.m_data.get(), t.numel());
}

template<class T>
std::ostream& operator<<(std::ostream& os, const CArray<T>& arr) {
    T last_v = arr.data[0];
    int last_c = 1;
    const char * sep = "";
    //os << "CArray<" << typeid(arr.data[0]).name() << "," << arr.N << ">{";
    os << "{";
    for(int i = 1; i < arr.N; i++) {
        auto cur_v = arr.data[i];
        if (last_v != cur_v) {
            os << sep << last_v;
            if (last_c > 1) os << "...x" << last_c;
            last_v = cur_v;
            last_c = 1;
        } else {
            last_c ++;
        }
    }
    os << sep << last_v;
    if (last_c > 1) os << "...x" << last_c;
    os << "}";
    return os;
}

