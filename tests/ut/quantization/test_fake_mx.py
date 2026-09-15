#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest
import torch

from vllm_ascend.quantization.fake_mx import (
    fake_mx_quantize,
    learned_hadamard_transform,
    randomized_hadamard_transform,
)


def test_fake_mxfp4_matches_amct_e2m1_rounding_and_preserves_dtype():
    values = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 0.0],
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 0.0],
        dtype=torch.float32,
    )

    actual = fake_mx_quantize(values, "mxfp4", group_size=values.numel())

    assert actual.dtype == values.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fake_mxfp4_applies_per_group_amct_shared_exponent():
    values = torch.tensor([[12.0, 10.0, 1.0, -0.75, 0.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)

    actual = fake_mx_quantize(values, "mxfp4", group_size=4)

    expected = torch.tensor([[12.0, 12.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)
    assert actual.shape == values.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fake_mxfp8_matches_amct_half_away_rounding_without_fp8_dtype():
    values = torch.tensor([1.0, 1.0625, 1.125, 1.1875, 1.25, 1.875, 0.0, 0.0], dtype=torch.float16)

    actual = fake_mx_quantize(values, "mxfp8", group_size=values.numel())

    expected = torch.tensor([1.0, 1.125, 1.125, 1.25, 1.25, 1.875, 0.0, 0.0], dtype=torch.float16)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fake_mxfp8_uses_element_range_when_selecting_shared_scale():
    values = torch.tensor([448.0, 0.25] + [0.0] * 30, dtype=torch.float32)

    actual = fake_mx_quantize(values, "mxfp8", group_size=32)

    assert actual[0] == 448.0
    assert actual[1] == 0.25


def test_fake_mx_rejects_non_divisible_last_dimension_like_amct_unflatten():
    with pytest.raises(ValueError, match="divisible"):
        fake_mx_quantize(torch.ones(5), "mxfp4", group_size=4)


def test_fake_mx_supports_per_block_clip_ratio_tensor():
    values = torch.tensor([[6.0, 5.0, 3.0, 2.0, 6.0, 5.0, 3.0, 2.0]])
    clip_ratio = torch.tensor([[1.0, 0.5]])

    actual = fake_mx_quantize(values, "mxfp4", group_size=4, clip_ratio=clip_ratio)

    torch.testing.assert_close(actual[..., :4], torch.tensor([[6.0, 6.0, 3.0, 2.0]]))
    torch.testing.assert_close(actual[..., 4:], torch.tensor([[3.0, 3.0, 3.0, 2.0]]))


def test_randomized_hadamard_transform_applies_normalized_hd_by_group():
    values = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0]])
    signs = torch.ones(8, dtype=torch.int8)

    actual = randomized_hadamard_transform(values, signs, group_size=4)

    expected = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.linalg.vector_norm(actual), torch.linalg.vector_norm(values))


def test_learned_hadamard_transform_matches_amct_q_block_contract():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    transform = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    actual = learned_hadamard_transform(values, transform)

    expected = values.reshape(-1, 2).matmul(transform).reshape_as(values)
    torch.testing.assert_close(actual, expected)


def test_learned_hadamard_transform_rejects_misaligned_dimension():
    with pytest.raises(ValueError, match="divisible by matrix_size"):
        learned_hadamard_transform(torch.ones(1, 3), torch.eye(2))


@pytest.mark.parametrize("group_size", (0, 3))
def test_randomized_hadamard_transform_rejects_non_power_of_two_group(group_size):
    with pytest.raises(ValueError, match="power of two"):
        randomized_hadamard_transform(torch.ones(1, 4), torch.ones(4), group_size)


def test_fake_mxfp4_matches_amct_half_away_from_zero_and_shared_exponent():
    values = torch.tensor([[0.25, -0.25, 1.25, -1.25, 7.0, 0.0, 0.0, 0.0]])

    actual = fake_mx_quantize(values, "mxfp4", group_size=4)

    expected = torch.tensor([[0.25, -0.25, 1.5, -1.5, 6.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("tensor", "mx_format", "group_size", "error"),
    [
        (torch.ones(2, dtype=torch.int32), "mxfp4", 32, TypeError),
        (torch.ones(2), "invalid", 32, ValueError),
        (torch.ones(2), "mxfp8", 0, ValueError),
        (torch.ones(2), "mxfp8", 32, ValueError),
    ],
)
def test_fake_mx_rejects_invalid_inputs(tensor, mx_format, group_size, error):
    with pytest.raises(error):
        clip_ratio = 0.0 if mx_format == "mxfp8" and group_size == 32 else 1.0
        fake_mx_quantize(tensor, mx_format, group_size, clip_ratio=clip_ratio)
