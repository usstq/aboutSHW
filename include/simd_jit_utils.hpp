#pragma once
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "misc.hpp"

//#include "openvino/core/except.hpp"
//#define ASSERT OPENVINO_ASSERT

namespace ov {
namespace intel_cpu {

static const int SIMDJIT_DEBUG = std::getenv("SIMDJIT_DEBUG") ? std::atoi(std::getenv("SIMDJIT_DEBUG")) : 0;

inline int jit_dump_asm(const char* name, const void* pcode, size_t size) {
    std::cout << "\033[32m::::    " << name << "    ::::\033[0m" << std::endl;
    std::ofstream outfile;
    outfile.open("temp.bin", std::ios_base::binary);
    outfile.write(reinterpret_cast<const char*>(pcode), size);
    outfile.close();
#ifdef __x86_64__
    auto ret = std::system("objdump -D -b binary -mi386:x86-64 -M intel temp.bin");
#endif
#ifdef __aarch64__
    auto ret = std::system("objdump -D -b binary -maarch64 temp.bin");
#endif
    return ret;
}

class reg_pool {
public:
    reg_pool(const char* name, size_t max_num_regs = 0) : m_name(name) {
        if (max_num_regs > 0) {
            add_range(0, max_num_regs - 1);
        }
    }

    void add_range(const std::vector<int>& reg_indices) {
        for (auto& reg_index : reg_indices) {
            if (m_reg_slot.size() < reg_index)
                m_reg_slot.resize(reg_index + 1, -1);

            m_reg_slot[reg_index] = m_reg_status.size();
            m_reg_status.push_back(reg_index);
        }
    }

    void add_range(int reg_index_from, int reg_index_to) {
        // m_reg_slot is a mapping table from reg-index to status-index
        if (m_reg_slot.size() < reg_index_to)
            m_reg_slot.resize(reg_index_to + 1, -1);

        for (int i = reg_index_from; i <= reg_index_to; i++) {
            m_reg_slot[i] = m_reg_status.size();
            m_reg_status.push_back(i);
        }
    }

    bool is_used(int slot_index) {
        ASSERT(static_cast<size_t>(slot_index) < m_reg_status.size(), slot_index, " > ", m_reg_status.size());
        return m_reg_status[slot_index] & mark_used;
    }

    int allocate(int slot_index = -1) {
        // allocate register with specific slot index
        if (slot_index >= 0) {
            ASSERT(static_cast<size_t>(slot_index) < m_reg_status.size(), slot_index, " > ", m_reg_status.size());
            ASSERT((m_reg_status[slot_index] & mark_used) == 0);
            auto reg_idx = m_reg_status[slot_index];
            m_reg_status[slot_index] |= mark_used;
            return reg_idx;
        }

        auto it = std::find_if(m_reg_status.begin(), m_reg_status.end(), [](int& v) {
            return (v & mark_used) == 0;
        });
        ASSERT(it != m_reg_status.end(), "regiter pool ", m_name, " exhausted.");
        auto reg_idx = *it;
        *it |= mark_used;
        return reg_idx;
    }

    void free(int i) {
        auto slot_id = m_reg_slot[i];
        m_reg_status[slot_id] &= (~mark_used);
    }
    void clear() {
        for (auto& s : m_reg_status)
            s &= (~mark_used);
    }

private:
    const char* m_name;
    static constexpr int mark_used = 0x80000000;
    std::vector<int> m_reg_status;  //
    std::vector<int> m_reg_slot;    //
};

struct RegExprImpl {
    // "r" register
    // "i" imm32
    // "+"/"-"/..... normal binary OP
    const char* op;
    int data = -1;
    std::unique_ptr<RegExprImpl> lhs;
    std::unique_ptr<RegExprImpl> rhs;

#ifdef __aarch64__
    // for lowering to ARM instruction with shifted register as rhs
    //  ADD  Xd, Xn, Xm{, shift #amount}
    //  AND  Xd, Xn, Xm{, shift #amount}
    //  EON  Xd, Xn, Xm{, shift #amount}
    //  EOR  Xd, Xn, Xm{, shift #amount}
    //  ORN  Xd, Xn, Xm{, shift #amount}
    //  ORR  Xd, Xn, Xm{, shift #amount}
    //  SUB  Xd, Xn, Xm{, shift #amount}
    //  TST  Xn, Xm{, shift #amount}
    enum class SHIFT_TYPE { NONE = 0, LSL, LSR, ASR, ROR };
    SHIFT_TYPE shift_type = SHIFT_TYPE::NONE;
    uint8_t shift_amount = 0;
#endif

    template <typename T>
    T as_r64() {
        ASSERT(!is_op("i"));
        return T(data);
    }
    int as_imm32() {
        ASSERT(is_op("i"));
        return data;
    }

    bool is_leaf() const {
        return (!lhs) && (!rhs);
    }
    bool is_reg() const {
        return is_op("r");
    }
    bool is_imm() const {
        return is_op("i");
    }
    bool is_cmp() const {
        return is_op(">") || is_op(">=") || is_op("<") || is_op("<=") || is_op("==") || is_op("!=");
    }
    bool is_logical_op() const {
        return is_cmp() || is_op("&&") || is_op("||") || is_op("!");
    }
    bool is_op(const char* name = nullptr) const {
        // all nodes other than leaf is op
        if (name == nullptr)
            return !is_leaf();

        // check op type
        if (op[1] == 0)
            return op[0] == name[0] && op[1] == name[1];
        else if (op[2] == 0)
            return op[0] == name[0] && op[1] == name[1] && op[2] == name[2];
        else if (op[3] == 0)
            return op[0] == name[0] && op[1] == name[1] && op[2] == name[2] && op[3] == name[3];
        return false;
    }

    bool is_swapped = false;
    void try_swap_lhs_rhs() {
        if (is_op("+") || is_op("*") || is_op("&") || is_op("|")) {
            std::swap(lhs, rhs);
            is_swapped = true;
        } else if (is_cmp()) {
            std::swap(lhs, rhs);
            is_swapped = true;
            if (is_op(">"))
                op = "<";
            else if (is_op(">="))
                op = "<=";
            else if (is_op("<"))
                op = ">";
            else if (is_op("<="))
                op = ">=";
        } else {
            // no swap
            is_swapped = false;
        }
    }

    std::string to_string() const {
        if (is_leaf()) {
            if (std::string(op) == "i")
                return std::to_string(data);
            return std::string(op) + std::to_string(data);
        }
        return std::string("t") + std::to_string(data) + " = " + lhs->to_string() + op + rhs->to_string();
    }

    std::string name() const {
        if (is_leaf()) {
            if (is_imm()) {
                return std::to_string(data);
            } else {
                return std::string(op) + std::to_string(data);
            }
        }
        return data >= 0 ? std::string("r") + std::to_string(data)
                         : "@" + std::to_string(reinterpret_cast<uintptr_t>(this));
    }

    void show_rpn() const {
        std::cout << "\033[32m::::" << " orignal expression " << "::::\033[0m" << std::endl;
        std::cout << "infix expression: ";
        _show_rpn(this, true);
        std::cout << std::endl;
        std::cout << "suffix expression: ";
        _show_rpn(this, false);
        std::cout << std::endl;
    }
    void _show_rpn(const RegExprImpl* pimpl, bool infix) const {
        if (!pimpl)
            return;
        if (pimpl->is_leaf()) {
            std::cout << pimpl->name();
            return;
        }
        if (infix) {
            if (!pimpl->rhs) {
                std::cout << "(" << pimpl->op;
                _show_rpn(pimpl->lhs.get(), infix);
                std::cout << ")";
            } else {
                std::cout << "(";
                _show_rpn(pimpl->lhs.get(), infix);
                std::cout << pimpl->op;
                _show_rpn(pimpl->rhs.get(), infix);
                std::cout << ")";
            }
        } else {
            std::cout << "(";
            _show_rpn(pimpl->lhs.get(), infix);
            std::cout << ",";
            _show_rpn(pimpl->rhs.get(), infix);
            std::cout << ")" << pimpl->op;
        }
    }

    RegExprImpl(const char* op, int data) : op(op), data(data) {}
    RegExprImpl(const char* op, std::unique_ptr<RegExprImpl>& _lhs) : op(op), lhs(std::move(_lhs)) {}
    RegExprImpl(const char* op, std::unique_ptr<RegExprImpl>& _lhs, std::unique_ptr<RegExprImpl>& _rhs)
        : op(op),
          lhs(std::move(_lhs)),
          rhs(std::move(_rhs)) {
        // regularize operand order to allow best reuse temp register (or make rhs imm)
        if (!rhs->is_leaf())
            try_swap_lhs_rhs();
        else if (lhs->is_imm())
            try_swap_lhs_rhs();
    }

    // for_each_op all op
    bool for_each_op(const std::function<bool(RegExprImpl* node)>& callback, RegExprImpl* root = nullptr) {
        if (root == nullptr)
            root = this;

        if (root->is_leaf())
            return true;

        if (root->lhs && !root->lhs->is_leaf()) {
            if (!for_each_op(callback, root->lhs.get()))
                return false;  // early terminate
        }
        if (root->rhs && !root->rhs->is_leaf()) {
            if (!for_each_op(callback, root->rhs.get()))
                return false;  // early terminate
        }
        return callback(root);
    }

    void const_folding() {
        for_each_op([&](RegExprImpl* p) {
            if (p->is_op("-") && p->rhs->is_imm()) {
                p->op = "+";
                p->rhs->data = -(p->rhs->data);
            }
            return true;
        });

        for_each_op([&](RegExprImpl* p) {
            if (p->rhs && !p->rhs->is_imm())
                return true;

            if (p->is_op("+")) {
                if (p->lhs->is_op("+") && p->lhs->rhs->is_imm()) {
                    p->rhs->data += p->lhs->rhs->as_imm32();
                    p->lhs = std::move(p->lhs->lhs);
                }
            }
            if (p->is_op("*")) {
                if (p->lhs->is_op("*") && p->lhs->rhs->is_imm()) {
                    p->rhs->data *= p->lhs->rhs->as_imm32();
                    p->lhs = std::move(p->lhs->lhs);
                }
            }
            return true;
        });
    }

    bool replace_scratch_with_dst(int assign_dst_reg_idx) {
        bool dst_register_assigned_inplace = false;
        // try to replace last scratch register with assign destination register
        auto assign_dst_reg_scratch_sn = data;
        // find the appearance of last access
        int last_access_exec_id = -1;
        int op_exec_id = 0;
        for_each_op([&](RegExprImpl* p) {
            op_exec_id++;
            if (p->lhs->is_reg() && p->lhs->data == assign_dst_reg_idx) {
                last_access_exec_id = op_exec_id;
            }
            if (p->rhs && p->rhs->is_reg() && p->rhs->data == assign_dst_reg_idx) {
                last_access_exec_id = op_exec_id;
            }
            return true;
        });
        // replace assign dst scratch with real assign dest reg
        op_exec_id = 0;
        bool replaced = false;
        for_each_op([&](RegExprImpl* p) {
            op_exec_id++;
            if (op_exec_id >= last_access_exec_id && p->data == assign_dst_reg_scratch_sn) {
                // the scratch reg has longer life-cycle, cannot replace
                if (p->lhs->data == assign_dst_reg_scratch_sn)
                    return false;
                p->data = assign_dst_reg_idx;
                replaced = true;
            }
            return true;
        });
        if (replaced) {
            dst_register_assigned_inplace = true;
            // remove useless mov
            for_each_op([&](RegExprImpl* p) {
                if (p->lhs->is_op(" ") && p->lhs->lhs->is_reg() && p->lhs->lhs->data == p->lhs->data) {
                    p->lhs = std::move(p->lhs->lhs);
                }
                return true;
            });
        }
        return dst_register_assigned_inplace;
    }
};

// all jit-based/performance-aware function should be a functor/callable because:
//   - it needs to hold reference to kernel (to save build time & resources)
//   - it needs to do other compile time preparation work and hold the relevant
//     runtime-data-struct (to make runtime faster)
// to optimze compile-time-workload itself, the functor instance itself should be
// cached with compile-time parameter as the key.
//
// because it's a functor, which supposed to have no states, so cache-factory should
// always return shared_ptr to constant object, so it won't behave differently when being
// called by different caller, and this also ensure it's multi-threading safe since it
// won't modify it's content.
//
template <typename... TTypes>
class tuple_hasher {
private:
    typedef std::tuple<TTypes...> Tuple;
    template <int N>
    size_t hash(Tuple& value) const {
        return 0;
    }
    template <int N, typename THead, typename... TTail>
    size_t hash(Tuple& value) const {
        constexpr int Index = N - sizeof...(TTail) - 1;
        return std::hash<THead>()(std::get<Index>(value)) ^ hash<N, TTail...>(value);
    }

public:
    size_t operator()(Tuple value) const {
        auto hv = hash<sizeof...(TTypes), TTypes...>(value);
        return hv;
    }
};

// create const object with internal cache with constructor-args as the key
// this helps reduces construction time overhead, and perfectly suitable
// for caching functor/callable.
template <class T, typename... CArgs>
std::shared_ptr<const T> make_cacheable(CArgs... cargs) {
    std::shared_ptr<const T> sptr;
    auto key = std::make_tuple(cargs...);
    static std::unordered_map<decltype(key), std::weak_ptr<const T>, tuple_hasher<CArgs...>> cache;
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        auto& wptr = it->second;
        sptr = wptr.lock();
        if (!sptr) {
            sptr = std::make_shared<T>(cargs...);
            // ECOUT("make_cacheable re-constructed: ", typeid(T).name(), "(", cargs..., ")");
            wptr = sptr;
        }
    } else {
        sptr = std::make_shared<T>(cargs...);
        // ECOUT("make_cacheable constructed: ", typeid(T).name(), "(", cargs..., ")");
        cache.emplace(std::make_pair(key, std::weak_ptr<const T>(sptr)));
    }
    return sptr;
}

template<int nbits, typename... Sizes>
int32_t get_imm8 (Sizes ... args){
    int v[] = { static_cast<int>(args)... };
    constexpr auto nargs = sizeof...(args);
    int32_t imm = 0;
    int ulimit = (1 << nbits);
    static_assert(nargs * nbits <= 8);
    for (int i = 0; i < nargs; i++) {
        ASSERT(v[i] >= 0 && v[i] <= ulimit);
        imm |= (v[i] << (i*nbits));
    }
    return imm;
}

}  // namespace intel_cpu
}  // namespace ov