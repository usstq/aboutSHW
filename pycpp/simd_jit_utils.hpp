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

template <typename... TS>
void _write_all(std::ostream& os, TS&&... args) {
    int dummy[sizeof...(TS)] = {(os << std::forward<TS>(args), 0)...};
    (void)dummy;
}
#define OPENVINO_ASSERT(cond, ...)                                                      \
    if (!(cond)) {                                                                      \
        std::stringstream ss;                                                           \
        _write_all(ss, __FILE__, ":", __LINE__, " ", #cond, " failed:", ##__VA_ARGS__); \
        std::cout << "\033[31m" << ss.str() << "\033[0m" << std::endl;                  \
        throw std::runtime_error(ss.str());                                             \
    }

namespace ov {
namespace intel_cpu {

static const int SIMDJIT_DEBUG = std::getenv("SIMDJIT_DEBUG") ? std::atoi(std::getenv("SIMDJIT_DEBUG")) : 0;

class reg_pool {
public:
    reg_pool(const char* name, size_t max_num_regs = 0) : m_name(name) {
        if (max_num_regs > 0) {
            add_range(0, max_num_regs - 1);
        }
    }

    void add_range(const std::vector<int>& reg_indices) {
        for(auto& reg_index : reg_indices) {
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

        for(int i = reg_index_from; i <= reg_index_to; i++) {
            m_reg_slot[i] = m_reg_status.size();
            m_reg_status.push_back(i);
        }
    }

    int allocate(int slot_index = -1) {
        // allocate register with specific slot index
        if (slot_index >= 0) {
            OPENVINO_ASSERT(static_cast<size_t>(slot_index) < m_reg_status.size());
            OPENVINO_ASSERT((m_reg_status[slot_index] & mark_used) == 0);
            auto reg_idx = m_reg_status[slot_index];
            m_reg_status[slot_index] |= mark_used;
            return reg_idx;
        }

        auto it = std::find_if(m_reg_status.begin(), m_reg_status.end(), [](int& v) {
            return (v & mark_used) == 0;
        });
        OPENVINO_ASSERT(it != m_reg_status.end(), "regiter pool ", m_name, " exhausted.");
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
    std::vector<int> m_reg_slot;   //
};

struct RegExprImpl {
    // "r" register
    // "i" imm32
    // "+"/"-"/..... normal binary OP
    const char* op;
    int data = -1;
    std::unique_ptr<RegExprImpl> lhs;
    std::unique_ptr<RegExprImpl> rhs;

    template <typename T>
    T as_r64() {
        OPENVINO_ASSERT(!is_op("i"));
        return T(data);
    }
    int as_imm32() {
        OPENVINO_ASSERT(is_op("i"));
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
        return false;
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
        std::cout << "\033[32m::::"
                  << " orignal expression "
                  << "::::\033[0m" << std::endl;
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
          rhs(std::move(_rhs)) {}

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
};

}  // namespace intel_cpu
}  // namespace ov