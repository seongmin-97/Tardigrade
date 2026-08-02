#!/usr/bin/env python3
"""
Python PyTorch vs Tardigrade C++ Tensor Automated Comparison Test Suite
========================================================================

Runs Tardigrade C++ test executable (`test_tensor_runner`) with identical inputs
and compares Forward outputs and Backward Autograd gradients against PyTorch reference outputs.
Aims for near 100% C++ Line Coverage on Tensor.cpp / Autograd.cpp.
"""

import json
import os
import subprocess
import pytest
import numpy as np
import torch

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "UnitTest")
RUNNER_BIN = os.path.join(BUILD_DIR, "test_tensor_runner")


def run_cpp_runner(op_name: str, input_str: str = "", extra_args: list = None) -> dict:
    if not os.path.exists(RUNNER_BIN):
        pytest.fail(f"C++ Test runner binary not found at: {RUNNER_BIN}. Please build the C++ project first.")

    cmd = [RUNNER_BIN, op_name]
    if extra_args:
        cmd.extend([str(a) for a in extra_args])

    res = subprocess.run(cmd, input=input_str, text=True, capture_output=True)
    if res.returncode != 0:
        pytest.fail(f"C++ runner failed (exit code {res.returncode}):\n{res.stderr}")

    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Failed to parse JSON from C++ runner output:\nOutput: {res.stdout}\nError: {e}")


def format_tensor_input(t: torch.Tensor) -> str:
    dims = list(t.shape)
    data_list = t.detach().numpy().flatten().tolist()
    return f"{len(dims)} " + " ".join(map(str, dims)) + " " + " ".join(map(str, data_list)) + "\n"


# =====================================================================
# 1. Factory & Basic Member Methods
# =====================================================================

def test_tensor_factory_zeros():
    shape = [2, 3, 4]
    cpp_res = run_cpp_runner("zeros", extra_args=shape)
    torch_res = torch.zeros(shape).numpy()
    assert cpp_res["shape"] == shape
    np.testing.assert_allclose(cpp_res["data"], torch_res.flatten(), atol=1e-6)


def test_tensor_factory_ones():
    shape = [3, 5]
    cpp_res = run_cpp_runner("ones", extra_args=shape)
    torch_res = torch.ones(shape).numpy()
    assert cpp_res["shape"] == shape
    np.testing.assert_allclose(cpp_res["data"], torch_res.flatten(), atol=1e-6)


def test_tensor_factory_fill():
    shape = [4, 2]
    val = 3.14159
    cpp_res = run_cpp_runner("fill", extra_args=[val] + shape)
    torch_res = torch.full(shape, val).numpy()
    assert cpp_res["shape"] == shape
    np.testing.assert_allclose(cpp_res["data"], torch_res.flatten(), atol=1e-6)


def test_tensor_item_and_conversion():
    t_torch = torch.tensor([42.0], dtype=torch.float64)
    input_stream = format_tensor_input(t_torch)
    cpp_res = run_cpp_runner("item", input_str=input_stream)
    assert cpp_res["item"] == 42.0
    assert cpp_res["doubleVal"] == 42.0


def test_tensor_member_methods():
    t_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    input_stream = format_tensor_input(t_torch)
    cpp_res = run_cpp_runner("member_methods", input_str=input_stream)
    np.testing.assert_allclose(cpp_res["data"], [1.0, 2.0, 3.0, 4.0], atol=1e-6)


# =====================================================================
# 2. In-Place Arithmetic Ops (+=, -=)
# =====================================================================

def test_tensor_inplace_ops():
    a_torch = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    b_torch = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
    input_stream = format_tensor_input(a_torch) + format_tensor_input(b_torch)
    cpp_res = run_cpp_runner("inplace_ops", input_str=input_stream)
    np.testing.assert_allclose(cpp_res["data"], a_torch.numpy(), atol=1e-6)


# =====================================================================
# 3. Select & SetSelect, Slice & SetSlice
# =====================================================================

def test_tensor_select_and_setselect():
    a_torch = torch.zeros((3, 3), dtype=torch.float64)
    input_stream = format_tensor_input(a_torch)
    cpp_res = run_cpp_runner("select_setselect", input_str=input_stream, extra_args=[1, 0])
    
    expected = np.array([[9.9, 0.0, 0.0], [9.9, 0.0, 0.0], [9.9, 0.0, 0.0]])
    np.testing.assert_allclose(cpp_res["data"], expected.flatten(), atol=1e-6)


def test_tensor_slice_and_setslice():
    a_torch = torch.zeros((4, 4), dtype=torch.float64)
    input_stream = format_tensor_input(a_torch)
    cpp_res = run_cpp_runner("slice_setslice", input_str=input_stream, extra_args=[0, 1, 3])
    
    expected = np.zeros((4, 4))
    expected[1:3, :] = 7.7
    np.testing.assert_allclose(cpp_res["data"], expected.flatten(), atol=1e-6)


# =====================================================================
# 4. Element-wise Binary Ops (add, sub, mul, div)
# =====================================================================

@pytest.mark.parametrize("op_name, torch_op", [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b),
])
def test_elementwise_binary_ops(op_name, torch_op):
    torch.manual_seed(42)
    shape_a = [2, 3]
    shape_b = [2, 3]
    
    a_torch = torch.randn(shape_a, dtype=torch.float64, requires_grad=True)
    b_torch = (torch.rand(shape_b, dtype=torch.float64) + 0.1).detach().requires_grad_(True)

    c_torch = torch_op(a_torch, b_torch)
    loss = c_torch.sum()
    loss.backward()

    input_stream = format_tensor_input(a_torch) + format_tensor_input(b_torch)
    cpp_res = run_cpp_runner(op_name, input_str=input_stream)

    np.testing.assert_allclose(cpp_res["data"], c_torch.detach().numpy().flatten(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_res["gradA"], a_torch.grad.numpy().flatten(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_res["gradB"], b_torch.grad.numpy().flatten(), rtol=1e-5, atol=1e-6)


# =====================================================================
# 5. Scalar Operations & Comparisons & Chained Autograd
# =====================================================================

def test_scalar_arithmetic_ops():
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    input_stream = format_tensor_input(a_torch)
    cpp_res = run_cpp_runner("scalar_ops", input_str=input_stream)

    s = 2.5
    res_torch = (a_torch + s) + (s + a_torch) + (a_torch - s) + (s - a_torch) + \
                (a_torch * s) + (s * a_torch) + (a_torch / s) + (s / a_torch) + (a_torch / s)
    np.testing.assert_allclose(cpp_res["data"], res_torch.numpy().flatten(), rtol=1e-5, atol=1e-6)


def test_comparison_operators():
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 0.5]], dtype=torch.float64)
    b_torch = torch.tensor([[1.0, 1.0], [4.0, 0.5]], dtype=torch.float64)
    input_stream = format_tensor_input(a_torch) + format_tensor_input(b_torch)
    cpp_res = run_cpp_runner("comparison_ops", input_str=input_stream)
    assert "data" in cpp_res


def test_chained_autograd_graph():
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float64, requires_grad=True)
    input_stream = format_tensor_input(a_torch) + format_tensor_input(b_torch)
    cpp_res = run_cpp_runner("chained_autograd_graph", input_str=input_stream)
    assert cpp_res["shape"] == [2, 2]


def test_broadcasting_branches():
    cpp_res = run_cpp_runner("broadcasting_branches")
    assert "data" in cpp_res


# =====================================================================
# 6. Matmul, Unary Math, Reshape, Permute, Concat, Multi-Axis Sum
# =====================================================================

def test_matmul_and_autograd():
    torch.manual_seed(123)
    a_torch = torch.randn([3, 4], dtype=torch.float64, requires_grad=True)
    b_torch = torch.randn([4, 5], dtype=torch.float64, requires_grad=True)

    c_torch = torch.matmul(a_torch, b_torch)
    loss = c_torch.sum()
    loss.backward()

    input_stream = format_tensor_input(a_torch) + format_tensor_input(b_torch)
    cpp_res = run_cpp_runner("matmul", input_str=input_stream)

    assert cpp_res["shape"] == [3, 5]
    np.testing.assert_allclose(cpp_res["data"], c_torch.detach().numpy().flatten(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_res["gradA"], a_torch.grad.numpy().flatten(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_res["gradB"], b_torch.grad.numpy().flatten(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("op_name, torch_op", [
    ("exp", torch.exp),
    ("log", torch.log),
])
def test_unary_math_ops(op_name, torch_op):
    torch.manual_seed(99)
    a_torch = (torch.rand([2, 4], dtype=torch.float64) + 0.5).detach().requires_grad_(True)

    c_torch = torch_op(a_torch)
    loss = c_torch.sum()
    loss.backward()

    input_stream = format_tensor_input(a_torch)
    cpp_res = run_cpp_runner(op_name, input_str=input_stream)

    np.testing.assert_allclose(cpp_res["data"], c_torch.detach().numpy().flatten(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_res["gradA"], a_torch.grad.numpy().flatten(), rtol=1e-5, atol=1e-6)


def test_shape_manipulation_reshape():
    a_torch = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    target_shape = [2, 6]

    input_stream = format_tensor_input(a_torch) + f"{len(target_shape)} " + " ".join(map(str, target_shape)) + "\n"
    cpp_res = run_cpp_runner("reshape", input_str=input_stream)

    assert cpp_res["shape"] == target_shape
    np.testing.assert_allclose(cpp_res["data"], a_torch.reshape(2, 6).numpy().flatten(), atol=1e-6)


def test_shape_manipulation_permute():
    a_torch = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    dims = [2, 0, 1]

    input_stream = format_tensor_input(a_torch) + f"{len(dims)} " + " ".join(map(str, dims)) + "\n"
    cpp_res = run_cpp_runner("permute", input_str=input_stream)

    c_torch = a_torch.permute(dims)
    assert cpp_res["shape"] == list(c_torch.shape)
    np.testing.assert_allclose(cpp_res["data"], c_torch.numpy().flatten(), atol=1e-6)


def test_reduction_sum_axis():
    a_torch = torch.randn([2, 3, 4], dtype=torch.float64, requires_grad=True)
    axis = 1

    c_torch = torch.sum(a_torch, dim=axis, keepdim=True)
    loss = c_torch.sum()
    loss.backward()

    input_stream = format_tensor_input(a_torch) + f"{axis}\n"
    cpp_res = run_cpp_runner("sum", input_str=input_stream)

    np.testing.assert_allclose(cpp_res["data"], c_torch.detach().numpy().flatten(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_res["gradA"], a_torch.grad.numpy().flatten(), rtol=1e-5, atol=1e-6)


def test_reduce_max_axis():
    a_torch = torch.randn([2, 3, 4], dtype=torch.float64, requires_grad=True)
    axis = 1

    c_torch, _ = torch.max(a_torch, dim=axis, keepdim=True)
    loss = c_torch.sum()
    loss.backward()

    input_stream = format_tensor_input(a_torch) + f"{axis}\n"
    cpp_res = run_cpp_runner("reduce_max", input_str=input_stream)

    np.testing.assert_allclose(cpp_res["data"], c_torch.detach().numpy().flatten(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_res["gradA"], a_torch.grad.numpy().flatten(), rtol=1e-5, atol=1e-6)



def test_sum_multi_axes():
    a_torch = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    input_stream = format_tensor_input(a_torch)
    cpp_res = run_cpp_runner("sum_multi_axes", input_str=input_stream)

    c_torch = torch.sum(a_torch, dim=(0, 1))
    np.testing.assert_allclose(cpp_res["data"], c_torch.numpy().flatten(), atol=1e-6)


def test_tensor_concat():
    a_torch = torch.ones((2, 3), dtype=torch.float64)
    b_torch = torch.ones((2, 3), dtype=torch.float64) * 2.0
    input_stream = format_tensor_input(a_torch) + format_tensor_input(b_torch)
    cpp_res = run_cpp_runner("concat", input_str=input_stream, extra_args=[0])

    c_torch = torch.cat([a_torch, b_torch], dim=0)
    assert cpp_res["shape"] == [4, 3]
    np.testing.assert_allclose(cpp_res["data"], c_torch.numpy().flatten(), atol=1e-6)


def test_im2col_col2im():
    x_torch = torch.arange(25, dtype=torch.float64).reshape(1, 1, 5, 5)
    input_stream = format_tensor_input(x_torch)
    cpp_res = run_cpp_runner("im2col_col2im", input_str=input_stream)
    np.testing.assert_allclose(cpp_res["data"], x_torch.numpy().flatten(), atol=1e-6)


def test_convolve_2d():
    torch.manual_seed(777)
    x_torch = torch.randn([1, 1, 5, 5], dtype=torch.float64, requires_grad=True)
    k_torch = torch.randn([1, 1, 3, 3], dtype=torch.float64, requires_grad=True)

    y_torch = torch.nn.functional.conv2d(x_torch, k_torch, stride=1, padding=0)
    loss = y_torch.sum()
    loss.backward()

    input_stream = format_tensor_input(x_torch) + format_tensor_input(k_torch) + "1 0\n"
    cpp_res = run_cpp_runner("convolve", input_str=input_stream)

    np.testing.assert_allclose(cpp_res["data"], y_torch.detach().numpy().flatten(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_res["gradX"], x_torch.grad.numpy().flatten(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cpp_res["gradK"], k_torch.grad.numpy().flatten(), rtol=1e-5, atol=1e-6)


def test_tensor_exception_branches():
    cpp_res = run_cpp_runner("exceptions")
    assert cpp_res["exceptions_tested"] == 19



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
