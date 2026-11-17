"""Export the NumPy MLP to ONNX without the onnx package: the ModelProto is written directly in protobuf wire format.

Graph: obs -> Sub(mu) -> Div(sd) -> [MatMul -> Add -> Tanh] x layers -> action. Batch dimension is dynamic.
Field numbers follow onnx.proto (ModelProto, GraphProto, NodeProto, TensorProto, ValueInfoProto, TypeProto).
"""
import struct

import numpy as np

IR_VERSION = 8
OPSET = 17
FLOAT = 1


def _varint(n):
    out = bytearray()
    n &= (1 << 64) - 1
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field, wire):
    return _varint((field << 3) | wire)


def _int(field, v):
    return _key(field, 0) + _varint(v)


def _bytes(field, b):
    if isinstance(b, str):
        b = b.encode()
    return _key(field, 2) + _varint(len(b)) + b


def _tensor(name, arr):
    arr = np.ascontiguousarray(arr, dtype="<f4")
    return b"".join(_int(1, d) for d in arr.shape) + _int(2, FLOAT) + _bytes(8, name) + _bytes(9, arr.tobytes())


def _value_info(name, dims):
    shape = b"".join(_bytes(1, _bytes(2, d) if isinstance(d, str) else _int(1, d)) for d in dims)
    tensor_type = _int(1, FLOAT) + _bytes(2, shape)
    return _bytes(1, name) + _bytes(2, _bytes(1, tensor_type))


def _node(op, inputs, outputs, name):
    return b"".join(_bytes(1, i) for i in inputs) + b"".join(_bytes(2, o) for o in outputs) + _bytes(3, name) + _bytes(4, op)


def export(mlp, path, producer="robot-policy-serving"):
    nodes, inits = [], [_tensor("mu", mlp.mu), _tensor("sd", mlp.sd)]
    nodes.append(_node("Sub", ["obs", "mu"], ["x_c"], "center"))
    nodes.append(_node("Div", ["x_c", "sd"], ["h0"], "scale"))
    h = "h0"
    for i, (W, b) in enumerate(zip(mlp.W, mlp.b)):
        inits += [_tensor(f"W{i}", W), _tensor(f"b{i}", b)]
        out = "action" if i == len(mlp.W) - 1 else f"h{i + 1}"
        nodes.append(_node("MatMul", [h, f"W{i}"], [f"mm{i}"], f"matmul{i}"))
        nodes.append(_node("Add", [f"mm{i}", f"b{i}"], [f"z{i}"], f"add{i}"))
        nodes.append(_node("Tanh", [f"z{i}"], [out], f"tanh{i}"))
        h = out
    graph = (b"".join(_bytes(1, n) for n in nodes) + _bytes(2, "policy")
             + b"".join(_bytes(5, t) for t in inits)
             + _bytes(11, _value_info("obs", ["batch", mlp.sizes[0]]))
             + _bytes(12, _value_info("action", ["batch", mlp.sizes[-1]])))
    model = (_int(1, IR_VERSION) + _bytes(2, producer) + _bytes(7, graph)
             + _bytes(8, _bytes(1, "") + _int(2, OPSET)))
    with open(path, "wb") as f:
        f.write(model)
    return path
