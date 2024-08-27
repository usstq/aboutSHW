
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import sys, os
import argparse

def pshape2str(partial_shape):
    dlist = []
    for dim in partial_shape:
        if dim.is_static:
            dlist.append("{}".format(dim.get_length()))
        else:
            if (dim.get_min_length() == 0 and dim.get_max_length() == -1):
                dlist.append("-1")
            else:
                dlist.append("({},{})".format(dim.get_min_length(), dim.get_max_length()))
    return "[{}]".format(",".join(dlist))

def stringify(value):
    if isinstance(value, list) or isinstance(value, tuple):
        slv = [stringify(v) for v in value]
        return f"[{','.join(slv)}]"

    if isinstance(value, str):
        return f"'{value}'"
    
    if isinstance(value, dict):
        str_kwargs = []
        for k,v in value.items():
            str_kwargs.append(f"{k}={stringify(v)}")
        return ','.join(str_kwargs)

    return f"{str(value)}"


def camel2snake(str):
    return ''.join(['_'+i.lower() if i.isupper() else i for i in str]).lstrip('_')

def type2opname(type):
    if (type == "MatMul"):
        return "matmul"
    return camel2snake(type)

def shape2str(shape):
    return str(shape).replace("{","[").replace("}","]")

fmt_dict = {}
def openvino_op(names):
    def decorator(func):
        if isinstance(names, list):
            for n in names:
                fmt_dict[n] = func
        else:
            fmt_dict[names] = func
    return decorator

def outputs_shape_str(n):
    ret = []
    for i in range(n.get_output_size()):
        ret.append(f"{n.get_output_partial_shape(i)}")
    return ",".join(ret)

@openvino_op("Parameter")
def gen(n, inputs, varname, **kwargs):
    return f"{varname} = opset.parameter({pshape2str(n.get_partial_shape())}, Type.{n.get_element_type().get_type_name()}, name = '{n.get_friendly_name()}')"

@openvino_op("Gather")
def gen(n, inputs, varname, **kwargs):
    return f"{varname} = opset.gather({inputs[0]}, indices={inputs[1]}, axis={inputs[2]}) # " + outputs_shape_str(n)

@openvino_op("Constant")
def gen(n, inputs, varname, **kwargs):
    t_type = n.get_element_type().get_type_name()
    t_shape = shape2str(n.get_output_shape(0))
    vec = n.get_vector()
    if len(vec) <= 8:
        if t_type == "boolean":
            svec = "[{}]".format(",".join([str(bool(v)) for v in vec]))
        else:
            svec = "[{}]".format(",".join([str(v) for v in vec]))
    else:
        svec = "ConstDict['{}']".format(n.get_friendly_name())
    return f"{varname} = op.Constant(Type.{t_type}, Shape({t_shape}), {svec})"

@openvino_op("VariadicSplit")
def gen(n, inputs, varname, **kwargs):
    return f"{varname} = opset.variadic_split({inputs[0]}, axis = {inputs[1]}, split_lengths = {inputs[2]}) # " + outputs_shape_str(n)

@openvino_op("Transpose")
def gen(n, inputs, varname, **kwargs):
    return f"{varname} = opset.transpose({inputs[0]}, input_order = {inputs[1]}) # " + outputs_shape_str(n)

@openvino_op("ShapeOf")
def gen(n, inputs, varname, **kwargs):
    return f"{varname} = opset.shape_of({inputs[0]}, {stringify(kwargs['attr'])}) # " + outputs_shape_str(n)

@openvino_op("Concat")
def gen(n, inputs, varname, **kwargs):
    axis = kwargs['attr']["axis"]
    return f"{varname} = opset.concat([{','.join(inputs)}], axis = {axis}, name = '{n.get_friendly_name()}') # " + outputs_shape_str(n)

@openvino_op("Reshape")
def gen(n, inputs, varname, **kwargs):
    special_zero = kwargs['attr']["special_zero"]
    return f"{varname} = opset.reshape({inputs[0]}, output_shape = {inputs[1]}, special_zero = {special_zero}) # " + outputs_shape_str(n)

@openvino_op("Squeeze")
def gen(n, inputs, varname, **kwargs):
    axes = ""
    if len(inputs) > 1:
        axes = f", axes = {inputs[1]}"
    return f"{varname} = opset.squeeze({inputs[0]} {axes}) # " + outputs_shape_str(n)

@openvino_op("Unsqueeze")
def gen(n, inputs, varname, **kwargs):
    return f"{varname} = opset.unsqueeze({inputs[0]}, axes = {inputs[1]}) # " + outputs_shape_str(n)

@openvino_op("Result")
def gen(n, inputs, varname, **kwargs):
    return f"{varname} = opset.result({inputs[0]}, name='{n.get_friendly_name()}') # " + outputs_shape_str(n)

@openvino_op("Relu")
def gen(n, inputs, varname, **kwargs):
    return f"{varname} = opset.relu({inputs[0]}) # " + outputs_shape_str(n)

@openvino_op("Broadcast")
def gen(n, inputs, varname, **kwargs):
    #print(inputs)
    #print(kwargs['attr'])
    if 'mode' in kwargs['attr']:
        mode = kwargs['attr']['mode']
    elif 'broadcast_spec' in kwargs['attr']:
        mode = kwargs['attr']['broadcast_spec']
    else:
        assert(False)

    return f"{varname} = opset.broadcast({','.join(inputs)}, broadcast_spec='{mode}', name='{n.get_friendly_name()}') # " + outputs_shape_str(n)

@openvino_op("StridedSlice")
def gen(n, inputs, varname, **kwargs):
    #print(inputs)
    #print(kwargs['attr'])
    begin_mask = kwargs['attr']['begin_mask']
    end_mask = kwargs['attr']['end_mask']
    new_axis_mask = kwargs['attr']['new_axis_mask']
    shrink_axis_mask = kwargs['attr']['shrink_axis_mask']
    ellipsis_mask = kwargs['attr']['ellipsis_mask']
    return f"{varname} = opset.strided_slice({','.join(inputs)}, begin_mask={begin_mask}, end_mask={end_mask}, new_axis_mask={new_axis_mask}, shrink_axis_mask={shrink_axis_mask}, ellipsis_mask={ellipsis_mask}) # " + outputs_shape_str(n)

@openvino_op("ConvolutionBackpropData")
def gen(n, inputs, varname, **kwargs):
    #print(inputs)
    #print(kwargs['attr'])
    strides = kwargs['attr']['strides']
    dilations = kwargs['attr']['dilations']
    pads_begin = kwargs['attr']['pads_begin']
    pads_end = kwargs['attr']['pads_end']
    auto_pad = kwargs['attr']['auto_pad']
    output_padding = kwargs['attr']['output_padding']
    return f"{varname} = opset.convolution_backprop_data({','.join(inputs)}, strides={strides}, pads_begin={pads_begin}, pads_end={pads_end}, dilations={dilations}, auto_pad='{auto_pad}',output_padding={output_padding}, name = '{n.get_friendly_name()}') # " + outputs_shape_str(n)

@openvino_op("Range")
def gen(n, inputs, varname, **kwargs):
    #print(inputs)
    #print(kwargs['attr'])
    output_type = kwargs['attr']['output_type']
    return f"{varname} = opset.range({','.join(inputs)}, name = '{n.get_friendly_name()}') # " + outputs_shape_str(n)

@openvino_op("MVN")
def gen(n, inputs, varname, **kwargs):
    #print(inputs)
    #print(kwargs['attr'])
    eps = kwargs['attr']['eps']
    normalize_variance = kwargs['attr']['normalize_variance']
    eps_mode = kwargs['attr']['eps_mode']
    return f"{varname} = opset.mvn({','.join(inputs)}, normalize_variance={normalize_variance}, eps={eps}, eps_mode='{eps_mode}', name = '{n.get_friendly_name()}') # " + outputs_shape_str(n)

@openvino_op("Convolution")
def gen(n, inputs, varname, **kwargs):
    #print(inputs)
    #print(kwargs['attr'])
    strides = kwargs['attr']['strides']
    dilations = kwargs['attr']['dilations']
    pads_begin = kwargs['attr']['pads_begin']
    pads_end = kwargs['attr']['pads_end']
    auto_pad = kwargs['attr']['auto_pad']
    return f"{varname} = opset.convolution({','.join(inputs)}, strides={strides}, pads_begin={pads_begin}, pads_end={pads_end}, dilations={dilations}, auto_pad='{auto_pad}',name = '{n.get_friendly_name()}') # " + outputs_shape_str(n)

@openvino_op("LSTMCell")
def gen(n, inputs, varname, **kwargs):
    prefix = kwargs['prefix'] + "    "
    file = kwargs['file']
    attr = kwargs['attr']
    print(f"{prefix}{varname} = opset.lstm_cell(X = {inputs[0]},", file=file)
    print(f"{prefix}                            initial_hidden_state = {inputs[1]},",file=file)
    print(f"{prefix}                            initial_cell_state = {inputs[2]},",file=file)
    print(f"{prefix}                            W = {inputs[3]},",file=file)
    print(f"{prefix}                            R = {inputs[4]},",file=file)
    print(f"{prefix}                            B = {inputs[5]},",file=file)
    print(f"{prefix}                            hidden_size = {attr['hidden_size']},",file=file)
    print(f"{prefix}                            activations = {attr['activations']},",file=file)
    print(f"{prefix}                            activations_alpha = {attr['activations_alpha']},",file=file)
    print(f"{prefix}                            activations_beta = {attr['activations_beta']},",file=file)
    print(f"{prefix}                            clip = {attr['clip']},",file=file)
    print(f"{prefix}                            name = '{n.get_friendly_name()}')",file=file)
    print(f"{prefix}                            # {outputs_shape_str(n)} ")
    return None

@openvino_op(["TensorIterator", "Loop"])
def gen(n, inputs, varname, **kwargs):
    prefix = kwargs['prefix'] + "    "
    file = kwargs['file']
    attr = kwargs['attr']

    func_name = translate(n.get_function(), prefix=prefix)

    body_name = f"{n.get_function().get_name()}"
    body_params = f"{body_name}_params"
    body_results = f"{body_name}_results"

    print(f"{prefix}{body_name} = {func_name}(ConstDict)", file=file)
    print(f"{prefix}{body_params} = {body_name}.get_parameters()", file=file)
    print(f"{prefix}{body_results} = {body_name}.get_results()", file=file)
    

    if (n.get_type_name() == "Loop"):
        print(f"{prefix}{varname} = opset.loop({inputs[0]},{inputs[1]})", file=file)
        print(f"{prefix}{varname}.set_function({body_name})", file=file)
        
        special_body_ports = n.get_special_body_ports()
        if len(special_body_ports) == 2:
            current_iter_body_port, next_cond_body_port = special_body_ports
            print(f"{prefix}{varname}.set_special_body_ports([{current_iter_body_port}, {next_cond_body_port}])", file=file)
    else:
        print(f"{prefix}{varname} = opset.tensor_iterator()", file=file)
        print(f"{prefix}{varname}.set_body({body_name})", file=file)

    # current_iter will be write into body's input port 0
    # next_cond will be read from body's output port 0
    
    #print(f"{prefix}{varname}.set_sliced_input(X_i, X.output(0), 0, 2, 2, 39, 1)")
    #print(f"{prefix}{varname}.set_sliced_input(Y_i, Y.output(0), 0, 2, 2, -1, 1)")
    #print(f"{prefix}{varname}.set_invariant_input(M_body, M.output(0))")

    def outname(name):
        if not ".output(" in name:
            return name + ".output(0)"
        return name

    for idesc in n.get_input_descriptions():
        if isinstance(idesc,  op_util.SliceInputDescription):
            print(f"{prefix}{varname}.set_sliced_input({body_params}[{idesc.body_parameter_index}], value={outname(inputs[idesc.input_index])},start={idesc.start},stride={idesc.stride},part_size={idesc.part_size},end={idesc.end},axis={idesc.axis})", file=file)
        if isinstance(idesc,  op_util.MergedInputDescription):
            print(f"{prefix}{varname}.set_merged_input({body_params}[{idesc.body_parameter_index}], initial_value={outname(inputs[idesc.input_index])}, successive_value={body_results}[{idesc.body_value_index}].output(0))", file=file)
        if isinstance(idesc,  op_util.InvariantInputDescription):
            print(f"{prefix}{varname}.set_invariant_input({body_params}[{idesc.body_parameter_index}], value={outname(inputs[idesc.input_index])})", file=file)

    for idx,odesc in enumerate(n.get_output_descriptions()):
        if isinstance(odesc,  op_util.ConcatOutputDescription):
            print(f"{prefix}{varname}.get_concatenated_slices({body_results}[{odesc.body_value_index}].output(0), start={odesc.start}, stride={odesc.stride}, part_size={odesc.part_size}, end={odesc.end}, axis={odesc.axis})")
        if isinstance(odesc,  op_util.BodyOutputDescription):
            print(f"{prefix}{varname}.get_iter_value({body_results}[{odesc.body_value_index}].output(0), iteration={odesc.iteration})")
        assert(odesc.output_index == idx)

    print(f"{prefix} # output shapes: {outputs_shape_str(n)}")
    return None

@openvino_op("LSTMSequence")
def gen(n, inputs, varname, **kwargs):
    return f"{varname} = opset.lstm_sequence({','.join(inputs)}, {stringify(kwargs['attr'])}) # " + outputs_shape_str(n)

def getSimpleIntegers(n):
    itype = n.get_element_type().get_type_name()
    try:
        # print(n.get_friendly_name(), n.get_element_type().type_name)
        vec = n.get_vector()
        if len(vec) > 8:
            return None
        rank = len(n.get_output_shape(0))
        if rank > 1:
            return None
        if n.get_element_type().is_real:
            return None

        if rank == 0:
            return f"const_{itype}({vec[0]})"

        return "const_{}([{}])".format(itype, ",".join([str(v) for v in vec]))
    except:
        return f"const_{itype}(..., name='{n.get_friendly_name()}')"


def translate(model, prefix="", file = sys.stdout):
    name = model.get_name()
    func_name = f"model_{name}"
    print(f"{prefix}def {func_name}(arg):", file=file)

    print(f"{prefix}    if isinstance(arg, Model):")
    print(f"{prefix}        ConstDict = {'{}'}")
    print(f"{prefix}        collect_const(arg, ConstDict)")
    print(f"{prefix}    else:")
    print(f"{prefix}        ConstDict = arg")

    out2name = {}
    nameid = 0
    for n in model.get_ordered_ops():
        type = n.get_type_name()
        friendly_name = n.get_friendly_name()
        
        attr = {k:v for k,v in n.get_attributes().items()}
        rtinfo = ["{}={}".format(k, v) for k,v in n.get_rt_info().items()]

        varname = n.get_name()

        # for simple interger that can be use integer literal, we just record the
        # name as literal
        if type == "Constant":
            simple_ints = getSimpleIntegers(n)
            if simple_ints:
                out2name[n.output(0)] = simple_ints
                continue
        # varname = "node{}".format(nameid)
        # nameid += 1

        num_out = len(n.outputs())
        for k, out in enumerate(n.outputs()):
            out2name[out] = varname if num_out == 1 else "{}.output({})".format(varname, k)

        inputs = []
        for i in n.inputs():
            inputs.append(out2name[i.get_source_output()])
            
        if type in fmt_dict:
            src = fmt_dict[type](n, inputs, varname, attr=attr, prefix=prefix, file=file)
        else:
            # wildcard solution
            src = f"{varname} = opset.{type2opname(type)}({','.join(inputs)}, {stringify(attr)}, name = '{n.get_friendly_name()}') #wildcard " +  outputs_shape_str(n)
            #src = f"{prefix}      {varname} = # {type} {inputs} {attr} {rtinfo} {friendly_name}"

        if src:
            line = f"{prefix}    {src}"
            print(line, file=file)
    
    params = []
    for n in model.get_parameters():
        params.append(out2name[n.output(0)])
    
    results = []
    for n in model.get_results():
        #oname = out2name[n.input(0).get_source_output()]
        #if not ".output(" in oname:
        #    oname = oname + ".output(0)"
        results.append(out2name[n.output(0)])

    print(f"{prefix}    return Model([{','.join(results)}], [{','.join(params)}], '{name}')", file=file)
    return func_name


headers='''
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import sys, os

def const(data, itype):
    if isinstance(data, list):
        shape = [len(data)]
    else:
        data = [data]
        shape = []
    return op.Constant(itype, Shape(shape), data)

def const_i64(data):
    return const(data, Type.i64)

def const_i32(data):
    return const(data, Type.i32)

def collect_const(model, consts):
    for n in model.get_ordered_ops():
        if n.get_type_name() == "Constant":
            # print(n.get_friendly_name())
            consts[n.get_friendly_name()] = list(n.get_vector())

        if hasattr(n, "get_function"):
            collect_const(n.get_function(), consts)
    return

def show_io(m):
    print("Inputs of the model:")
    for port, _input in enumerate(m.inputs):
        print("\t[{}] {}".format(port, _input))
    print("Outputs of the model:")
    for port, _output in enumerate(m.outputs):
        print("\t[{}] {}".format(port, _output))

'''

tail='''


if __name__ == "__main__":
    org_model_path = "{}"
    core = Core()

    # example of introducing custome (OP) extension
    # import os
    # os.environ["add_RnntUpdate_opset8"] = "1"
    # core.add_extension("/home/dev/tingqian/ov-rnnt/rnnt_ov_extension/build/librnnt_ov_extension.so")

    model = core.read_model(org_model_path)
    print("====", org_model_path)
    show_io(model)

    model2 = {}(model)

    print("====", "new model")
    show_io(model2)

    serialize(model2, "{}")
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("org_model_path")
    args = parser.parse_args()

    core = Core()
    #core.add_extension("/home/dev/tingqian/ov-rnnt/rnnt_ov_extension/build/librnnt_ov_extension.so")

    model = core.read_model(args.org_model_path)

    print(f"")
    print(f"# auto generated by ir2py from {args.org_model_path}")

    print(headers)
    func_name = translate(model)

    old_base, filename = os.path.split(args.org_model_path)
    new_base = os.path.join(old_base, "hacked")
    if not os.path.exists(new_base):
        os.makedirs(new_base)

    print(tail.format(args.org_model_path, func_name, os.path.join(new_base, filename)))