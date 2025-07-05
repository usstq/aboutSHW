
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <memory>
#include <openvino/core/dimension.hpp>
#include <openvino/core/except.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/op.hpp>
#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension/op.hpp>
#include <openvino/frontend/node_context.hpp>
// #include "openvino/frontend/pytorch/extension/conversion.hpp"
// #include "openvino/frontend/pytorch/extension/op.hpp"
// #include <openvino/frontend/onnx/extension/op.hpp>
#include <string>
#include <unordered_map>
#include <vector>

std::vector<std::string> split_string_single_delimiter(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> ret;
    size_t pos = 0, pos_next;
    std::string token;
    while ((pos_next = s.find(delimiter, pos)) != std::string::npos) {
        token = s.substr(pos, pos_next - pos);
        ret.push_back(token);
        pos = pos_next + 1;
    }
    // return whole string if no delimiter if found
    token = s.substr(pos, pos_next);
    ret.push_back(token);
    return ret;
}

std::vector<std::string> split_string_multiple_delimiter(const std::string& s, const std::string& delimiter) {
    std::stringstream stream(s);
    std::string line;
    std::vector<std::string> ret;
    while(std::getline(stream, line)) 
    {
        std::size_t prev = 0, pos;
        while ((pos = line.find_first_of(delimiter, prev)) != std::string::npos)
        {
            if (pos > prev)
                ret.push_back(line.substr(prev, pos-prev));
            prev = pos+1;
        }
        if (prev < line.length())
            ret.push_back(line.substr(prev, std::string::npos));
    }
    return ret;
}

// Function to trim leading whitespace
void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// Function to trim trailing whitespace
void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// Function to trim both leading and trailing whitespace
void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

class Stub : public ov::op::Op {
    std::string m_mode;
    int m_axis;

public:
    OPENVINO_OP("Stub");
    // OPENVINO_FRAMEWORK_MAP(onnx, "Stub", {}, {});
    // OPENVINO_FRAMEWORK_MAP(pytorch, "Stub", {}, {});
    Stub() = default;
    Stub(const ov::OutputVector& args) : Op({args}) {
        constructor_validate_and_infer_types();
    }

    auto extract_from_node(ov::Node* node, std::string error) {
        while (node->get_type_name() == std::string("Convert")) {
            node = node->get_input_node_ptr(0);
        }
        auto node_const = dynamic_cast<ov::op::v0::Constant*>(node);
        OPENVINO_ASSERT(node_const, error);
        std::vector<char> info(node_const->get_byte_size() + 1, 0);
        std::memcpy(info.data(), node_const->get_data_ptr(), node_const->get_byte_size());
        std::string info_str(static_cast<const char*>(info.data()));
        trim(info_str);
        std::unordered_map<std::string, std::string> result;
        auto tokens = split_string_multiple_delimiter(info_str, " \n");
        OPENVINO_ASSERT(tokens.size() > 0, "at least one token, format is 'key1:value1 key2:value2', current: ", info_str);
        for (auto token : tokens) {
            auto kv = split_string_single_delimiter(token, ":");
            OPENVINO_ASSERT(kv.size() == 2, "token format should be key:value, current: ", token);
            result[kv[0]] = kv[1];
        }
        return result;
    }

    void validate_and_infer_types() override {
        auto input_num = get_input_size();
        OPENVINO_ASSERT(input_num >= 2, "At least 2 inputs: [-2] data input, [-1] attributes");
        auto attr_node = get_input_node_ptr(input_num - 1);
        // expected:
        //  'out_dt:f32 out_shape:[-1,1,1]'
        //  'out_dt:0 out_shape:0' data type from input 0, shape from input 0
        auto shape_info = extract_from_node(attr_node, "The last input for attributes must be constant");
        OPENVINO_ASSERT(shape_info.count("out_dt"), "output info must have 'out_dt' key, example: 'out_dt:f32'/'out_dt:0'");
        OPENVINO_ASSERT(shape_info.count("out_shape"), "output info must have 'out_shape' key, example: 'out_shape:[-1,1,1]'/'out_shape:0'");
        OPENVINO_ASSERT(shape_info.count("type"), "attribute must have 'type' key for real algorithm");
        auto dt_str = shape_info["out_dt"];
        auto shape_str = shape_info["out_shape"];

        ov::element::Type dt;
        if (std::isdigit(dt_str[0])) {
            auto dt_from_input = std::stoi(dt_str);
            OPENVINO_ASSERT(dt_from_input < input_num - 1 && dt_from_input >= 0, "dt is set from input ", dt_from_input, " but max input is ", input_num - 1);
            dt = get_input_element_type(dt_from_input);
        } else {
            if (dt_str == "f32") {
                dt = ov::element::f32;
            } else if (dt_str == "f16") {
                dt = ov::element::f16;
            } else if (dt_str == "bf16") {
                dt = ov::element::bf16;
            } else if (dt_str == "u8") {
                dt = ov::element::u8;
            } else if (dt_str == "i8") {
                dt = ov::element::i8;
            } else if (dt_str == "u4") {
                dt = ov::element::u4;
            } else if (dt_str == "i4") {
                dt = ov::element::i4;
            } else if (dt_str == "u32") {
                dt = ov::element::u32;
            } else if (dt_str == "i32") {
                dt = ov::element::i32;
            } else {
                OPENVINO_ASSERT(false, "unsupport dt type, current ", dt_str);
            }
        }
        ov::PartialShape ps;
        if (std::isdigit(shape_str[0])) {
            auto shape_from_input = std::stoi(shape_str);
            OPENVINO_ASSERT(shape_from_input < input_num - 1 && shape_from_input >= 0, "shape is set from input ", shape_from_input, " but max input is ", input_num - 1);
            ps = get_input_partial_shape(shape_from_input);
        } else {
            auto len = shape_str.size();
            OPENVINO_ASSERT(len > 2 && shape_str[0] == '[' && shape_str[len - 1] == ']', "shape length format should be '[1]', current: ", shape_str);
            shape_str = shape_str.substr(1, len - 2);
            auto dim_str = split_string_single_delimiter(shape_str, ",");
            std::vector<ov::Dimension> dims;
            std::transform(dim_str.begin(), dim_str.end(), std::back_inserter(dims), [&] (const std::string& s) {
                return std::stoi(s);
            });
            ps = dims;
        }
        for (auto&& kv : shape_info) {
            m_attributes[kv.first] = kv.second;
        }

        set_output_type(0, dt, ps);
    }
    
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<Stub>(new_args);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        for (auto&& kv : m_attributes) {
            visitor.on_attribute(kv.first, kv.second);
        }
        return true;
    }
#if 0
    bool has_evaluate() const {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
        const auto& in = inputs[0];
        auto& out = outputs[0];
        if (out.data() == in.data())  // Nothing to do
            return true;
        out.set_shape(in.get_shape());
        memcpy(out.data(), in.data(), in.get_byte_size());
        return true;
    }
#endif
    std::unordered_map<std::string, std::string> m_attributes;
};

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<Stub>>(),
        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<Stub>>()
    }));