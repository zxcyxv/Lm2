"""Fused Triton kernels for complex diagonal affine prefix scans."""
from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _complex_affine_combine(
    left_a_real,
    left_a_imag,
    left_b_real,
    left_b_imag,
    right_a_real,
    right_a_imag,
    right_b_real,
    right_b_imag,
):
    """Return ``right o left`` for scalar complex affine maps."""
    composed_a_real = (
        right_a_real * left_a_real - right_a_imag * left_a_imag
    )
    composed_a_imag = (
        right_a_real * left_a_imag + right_a_imag * left_a_real
    )
    rotated_b_real = (
        right_a_real * left_b_real - right_a_imag * left_b_imag
    )
    rotated_b_imag = (
        right_a_real * left_b_imag + right_a_imag * left_b_real
    )
    return (
        composed_a_real,
        composed_a_imag,
        right_b_real + rotated_b_real,
        right_b_imag + rotated_b_imag,
    )


@triton.jit
def _complex_affine_forward_kernel(
    multiplier,
    increment,
    output,
    rows: tl.constexpr,
    horizons: tl.constexpr,
    channels: tl.constexpr,
    values: tl.constexpr,
    block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    horizon = tl.arange(0, horizons)
    row_mask = row < rows
    value = row % values
    quotient = row // values
    channel = quotient % channels
    batch = quotient // channels

    multiplier_index = (
        ((batch[:, None] * horizons + horizon[None, :]) * channels
         + channel[:, None])
        * 2
    )
    increment_index = (
        (((batch[:, None] * horizons + horizon[None, :]) * channels
          + channel[:, None])
         * values + value[:, None])
        * 2
    )
    mask = row_mask[:, None]
    a_real = tl.load(multiplier + multiplier_index, mask=mask, other=0.0)
    a_imag = tl.load(
        multiplier + multiplier_index + 1, mask=mask, other=0.0
    )
    b_real = tl.load(increment + increment_index, mask=mask, other=0.0)
    b_imag = tl.load(
        increment + increment_index + 1, mask=mask, other=0.0
    )
    _, _, state_real, state_imag = tl.associative_scan(
        (a_real, a_imag, b_real, b_imag),
        axis=1,
        combine_fn=_complex_affine_combine,
    )
    tl.store(output + increment_index, state_real, mask=mask)
    tl.store(output + increment_index + 1, state_imag, mask=mask)


@triton.jit
def _complex_affine_adjoint_kernel(
    multiplier,
    state_gradient,
    adjoint,
    rows: tl.constexpr,
    horizons: tl.constexpr,
    channels: tl.constexpr,
    values: tl.constexpr,
    block_rows: tl.constexpr,
):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    horizon = tl.arange(0, horizons)
    row_mask = row < rows
    value = row % values
    quotient = row // values
    channel = quotient % channels
    batch = quotient // channels

    next_horizon = horizon + 1
    multiplier_index = (
        ((batch[:, None] * horizons + next_horizon[None, :]) * channels
         + channel[:, None])
        * 2
    )
    increment_index = (
        (((batch[:, None] * horizons + horizon[None, :]) * channels
          + channel[:, None])
         * values + value[:, None])
        * 2
    )
    multiplier_mask = row_mask[:, None] & (next_horizon[None, :] < horizons)
    mask = row_mask[:, None]
    reverse_a_real = tl.load(
        multiplier + multiplier_index,
        mask=multiplier_mask,
        other=0.0,
    )
    reverse_a_imag = -tl.load(
        multiplier + multiplier_index + 1,
        mask=multiplier_mask,
        other=0.0,
    )
    gradient_real = tl.load(
        state_gradient + increment_index, mask=mask, other=0.0
    )
    gradient_imag = tl.load(
        state_gradient + increment_index + 1, mask=mask, other=0.0
    )
    _, _, adjoint_real, adjoint_imag = tl.associative_scan(
        (
            reverse_a_real,
            reverse_a_imag,
            gradient_real,
            gradient_imag,
        ),
        axis=1,
        combine_fn=_complex_affine_combine,
        reverse=True,
    )
    tl.store(adjoint + increment_index, adjoint_real, mask=mask)
    tl.store(adjoint + increment_index + 1, adjoint_imag, mask=mask)


@triton.jit
def _complex_affine_multiplier_gradient_kernel(
    states,
    adjoint,
    multiplier_gradient,
    rows: tl.constexpr,
    horizons: tl.constexpr,
    channels: tl.constexpr,
    values: tl.constexpr,
    block_values: tl.constexpr,
):
    row = tl.program_id(0)
    value = tl.arange(0, block_values)
    channel = row % channels
    quotient = row // channels
    horizon = quotient % horizons
    batch = quotient // horizons
    value_mask = value < values

    adjoint_index = (
        (((batch * horizons + horizon) * channels + channel) * values + value)
        * 2
    )
    previous_horizon = horizon - 1
    previous_index = (
        (((batch * horizons + previous_horizon) * channels + channel)
         * values + value)
        * 2
    )
    adjoint_real = tl.load(
        adjoint + adjoint_index, mask=value_mask, other=0.0
    )
    adjoint_imag = tl.load(
        adjoint + adjoint_index + 1, mask=value_mask, other=0.0
    )
    previous_mask = value_mask & (horizon > 0)
    previous_real = tl.load(
        states + previous_index, mask=previous_mask, other=0.0
    )
    previous_imag = tl.load(
        states + previous_index + 1, mask=previous_mask, other=0.0
    )
    contribution_real = (
        adjoint_real * previous_real + adjoint_imag * previous_imag
    )
    contribution_imag = (
        adjoint_imag * previous_real - adjoint_real * previous_imag
    )
    output_index = ((batch * horizons + horizon) * channels + channel) * 2
    tl.store(multiplier_gradient + output_index, tl.sum(contribution_real))
    tl.store(
        multiplier_gradient + output_index + 1,
        tl.sum(contribution_imag),
    )


@triton.jit
def _constant_rotation_forward_value_kernel(
    step_angle,
    increment,
    root,
    output,
    batches: tl.constexpr,
    horizons: tl.constexpr,
    channels: tl.constexpr,
    values: tl.constexpr,
    block_values: tl.constexpr,
    angle_sign: tl.constexpr,
    has_root: tl.constexpr,
):
    """Fuse frame conversion, inclusive sum, and inverse conversion."""
    program = tl.program_id(0)
    value = tl.arange(0, block_values)[:, None]
    horizon = tl.arange(0, horizons)[None, :]
    channel = program % channels
    batch = program // channels
    value_mask = value < values

    angle = (
        tl.load(step_angle + channel).to(tl.float32) * angle_sign
    )
    cumulative_angle = (horizon + 1).to(tl.float32) * angle
    cosine = tl.cos(cumulative_angle)
    sine = tl.sin(cumulative_angle)
    increment_index = (
        (((batch * horizons + horizon) * channels + channel) * values
         + value)
        * 2
    )
    increment_real = tl.load(
        increment + increment_index,
        mask=value_mask,
        other=0.0,
    )
    increment_imag = tl.load(
        increment + increment_index + 1,
        mask=value_mask,
        other=0.0,
    )

    frame_real = cosine * increment_real + sine * increment_imag
    frame_imag = cosine * increment_imag - sine * increment_real
    frame_prefix_real = tl.cumsum(frame_real, axis=1)
    frame_prefix_imag = tl.cumsum(frame_imag, axis=1)
    if has_root:
        root_index = ((batch * channels + channel) * values + value) * 2
        root_real = tl.load(
            root + root_index,
            mask=value_mask,
            other=0.0,
        )
        root_imag = tl.load(
            root + root_index + 1,
            mask=value_mask,
            other=0.0,
        )
        frame_prefix_real += root_real
        frame_prefix_imag += root_imag

    state_real = cosine * frame_prefix_real - sine * frame_prefix_imag
    state_imag = sine * frame_prefix_real + cosine * frame_prefix_imag
    tl.store(
        output + increment_index,
        state_real,
        mask=value_mask,
    )
    tl.store(
        output + increment_index + 1,
        state_imag,
        mask=value_mask,
    )


@triton.jit
def _constant_rotation_forward_channel_kernel(
    step_angle,
    increment,
    root,
    output,
    batches: tl.constexpr,
    horizons: tl.constexpr,
    channels: tl.constexpr,
    block_channels: tl.constexpr,
    angle_sign: tl.constexpr,
    has_root: tl.constexpr,
):
    """The value-one specialization tiles channels into each program."""
    channel_blocks = tl.cdiv(channels, block_channels)
    program = tl.program_id(0)
    channel_vector = (
        (program % channel_blocks) * block_channels
        + tl.arange(0, block_channels)
    )
    channel = channel_vector[:, None]
    batch = program // channel_blocks
    horizon = tl.arange(0, horizons)[None, :]
    channel_mask = channel < channels

    angle = tl.load(
        step_angle + channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32) * angle_sign
    cumulative_angle = (horizon + 1).to(tl.float32) * angle
    cosine = tl.cos(cumulative_angle)
    sine = tl.sin(cumulative_angle)
    increment_index = (
        ((batch * horizons + horizon) * channels + channel) * 2
    )
    increment_real = tl.load(
        increment + increment_index,
        mask=channel_mask,
        other=0.0,
    )
    increment_imag = tl.load(
        increment + increment_index + 1,
        mask=channel_mask,
        other=0.0,
    )

    frame_real = cosine * increment_real + sine * increment_imag
    frame_imag = cosine * increment_imag - sine * increment_real
    frame_prefix_real = tl.cumsum(frame_real, axis=1)
    frame_prefix_imag = tl.cumsum(frame_imag, axis=1)
    if has_root:
        root_index = (batch * channels + channel) * 2
        root_real = tl.load(
            root + root_index,
            mask=channel_mask,
            other=0.0,
        )
        root_imag = tl.load(
            root + root_index + 1,
            mask=channel_mask,
            other=0.0,
        )
        frame_prefix_real += root_real
        frame_prefix_imag += root_imag

    state_real = cosine * frame_prefix_real - sine * frame_prefix_imag
    state_imag = sine * frame_prefix_real + cosine * frame_prefix_imag
    tl.store(
        output + increment_index,
        state_real,
        mask=channel_mask,
    )
    tl.store(
        output + increment_index + 1,
        state_imag,
        mask=channel_mask,
    )


@triton.jit
def _constant_rotation_backward_value_kernel(
    step_angle,
    increment,
    states,
    state_gradient,
    step_angle_gradient,
    increment_gradient,
    root_gradient,
    batches: tl.constexpr,
    horizons: tl.constexpr,
    channels: tl.constexpr,
    values: tl.constexpr,
    block_values: tl.constexpr,
    angle_sign: tl.constexpr,
    has_root: tl.constexpr,
):
    program = tl.program_id(0)
    value = tl.arange(0, block_values)[:, None]
    horizon = tl.arange(0, horizons)[None, :]
    channel = program % channels
    batch = program // channels
    value_mask = value < values

    angle = (
        tl.load(step_angle + channel).to(tl.float32) * angle_sign
    )
    cumulative_angle = (horizon + 1).to(tl.float32) * angle
    cosine = tl.cos(cumulative_angle)
    sine = tl.sin(cumulative_angle)
    index = (
        (((batch * horizons + horizon) * channels + channel) * values
         + value)
        * 2
    )
    gradient_real = tl.load(
        state_gradient + index,
        mask=value_mask,
        other=0.0,
    )
    gradient_imag = tl.load(
        state_gradient + index + 1,
        mask=value_mask,
        other=0.0,
    )

    frame_gradient_real = cosine * gradient_real + sine * gradient_imag
    frame_gradient_imag = cosine * gradient_imag - sine * gradient_real
    frame_suffix_real = tl.cumsum(
        frame_gradient_real,
        axis=1,
        reverse=True,
    )
    frame_suffix_imag = tl.cumsum(
        frame_gradient_imag,
        axis=1,
        reverse=True,
    )
    adjoint_real = cosine * frame_suffix_real - sine * frame_suffix_imag
    adjoint_imag = sine * frame_suffix_real + cosine * frame_suffix_imag
    tl.store(
        increment_gradient + index,
        adjoint_real,
        mask=value_mask,
    )
    tl.store(
        increment_gradient + index + 1,
        adjoint_imag,
        mask=value_mask,
    )

    state_real = tl.load(states + index, mask=value_mask, other=0.0)
    state_imag = tl.load(states + index + 1, mask=value_mask, other=0.0)
    increment_real = tl.load(increment + index, mask=value_mask, other=0.0)
    increment_imag = tl.load(
        increment + index + 1,
        mask=value_mask,
        other=0.0,
    )
    transported_real = state_real - increment_real
    transported_imag = state_imag - increment_imag
    angle_contribution = (
        adjoint_imag * transported_real
        - adjoint_real * transported_imag
    )
    by_horizon = tl.sum(angle_contribution, axis=0)
    angle_contribution_sum = tl.sum(by_horizon, axis=0) * angle_sign
    tl.atomic_add(step_angle_gradient + channel, angle_contribution_sum)

    if has_root:
        root_index = ((batch * channels + channel) * values + value) * 2
        tl.store(
            root_gradient + root_index,
            frame_suffix_real[:, 0],
            mask=value[:, 0] < values,
        )
        tl.store(
            root_gradient + root_index + 1,
            frame_suffix_imag[:, 0],
            mask=value[:, 0] < values,
        )


@triton.jit
def _constant_rotation_backward_channel_kernel(
    step_angle,
    increment,
    states,
    state_gradient,
    step_angle_gradient,
    increment_gradient,
    root_gradient,
    batches: tl.constexpr,
    horizons: tl.constexpr,
    channels: tl.constexpr,
    block_channels: tl.constexpr,
    angle_sign: tl.constexpr,
    has_root: tl.constexpr,
):
    channel_blocks = tl.cdiv(channels, block_channels)
    program = tl.program_id(0)
    channel_vector = (
        (program % channel_blocks) * block_channels
        + tl.arange(0, block_channels)
    )
    channel = channel_vector[:, None]
    batch = program // channel_blocks
    horizon = tl.arange(0, horizons)[None, :]
    channel_mask = channel < channels

    angle = tl.load(
        step_angle + channel,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32) * angle_sign
    cumulative_angle = (horizon + 1).to(tl.float32) * angle
    cosine = tl.cos(cumulative_angle)
    sine = tl.sin(cumulative_angle)
    index = ((batch * horizons + horizon) * channels + channel) * 2
    gradient_real = tl.load(
        state_gradient + index,
        mask=channel_mask,
        other=0.0,
    )
    gradient_imag = tl.load(
        state_gradient + index + 1,
        mask=channel_mask,
        other=0.0,
    )

    frame_gradient_real = cosine * gradient_real + sine * gradient_imag
    frame_gradient_imag = cosine * gradient_imag - sine * gradient_real
    frame_suffix_real = tl.cumsum(
        frame_gradient_real,
        axis=1,
        reverse=True,
    )
    frame_suffix_imag = tl.cumsum(
        frame_gradient_imag,
        axis=1,
        reverse=True,
    )
    adjoint_real = cosine * frame_suffix_real - sine * frame_suffix_imag
    adjoint_imag = sine * frame_suffix_real + cosine * frame_suffix_imag
    tl.store(
        increment_gradient + index,
        adjoint_real,
        mask=channel_mask,
    )
    tl.store(
        increment_gradient + index + 1,
        adjoint_imag,
        mask=channel_mask,
    )

    state_real = tl.load(states + index, mask=channel_mask, other=0.0)
    state_imag = tl.load(states + index + 1, mask=channel_mask, other=0.0)
    increment_real = tl.load(increment + index, mask=channel_mask, other=0.0)
    increment_imag = tl.load(
        increment + index + 1,
        mask=channel_mask,
        other=0.0,
    )
    transported_real = state_real - increment_real
    transported_imag = state_imag - increment_imag
    angle_contribution = (
        adjoint_imag * transported_real
        - adjoint_real * transported_imag
    )
    angle_contribution_sum = (
        tl.sum(angle_contribution, axis=1) * angle_sign
    )
    tl.atomic_add(
        step_angle_gradient + channel_vector,
        angle_contribution_sum,
        mask=channel_vector < channels,
    )

    if has_root:
        root_index = (batch * channels + channel_vector) * 2
        root_gradient_real = tl.sum(
            frame_suffix_real * (horizon == 0),
            axis=1,
        )
        root_gradient_imag = tl.sum(
            frame_suffix_imag * (horizon == 0),
            axis=1,
        )
        tl.store(
            root_gradient + root_index,
            root_gradient_real,
            mask=channel_vector < channels,
        )
        tl.store(
            root_gradient + root_index + 1,
            root_gradient_imag,
            mask=channel_vector < channels,
        )


def _launch_forward(
    multipliers: torch.Tensor,
    increments: torch.Tensor,
) -> torch.Tensor:
    if not multipliers.is_cuda or not increments.is_cuda:
        raise ValueError("Triton affine scan requires CUDA tensors")
    if multipliers.dtype != torch.complex64 or increments.dtype != torch.complex64:
        raise ValueError("Triton affine scan requires complex64 tensors")
    if multipliers.shape != increments.shape[:-1]:
        raise ValueError("Triton affine scan shapes are incompatible")
    multipliers = multipliers.contiguous()
    increments = increments.contiguous()
    leading = increments.shape[:-3]
    horizons, channels, values = increments.shape[-3:]
    batches = math_prod(leading)
    rows = batches * channels * values
    output = torch.empty_like(increments)
    block_rows = 32
    _complex_affine_forward_kernel[(triton.cdiv(rows, block_rows),)](
        torch.view_as_real(multipliers),
        torch.view_as_real(increments),
        torch.view_as_real(output),
        rows=rows,
        horizons=horizons,
        channels=channels,
        values=values,
        block_rows=block_rows,
        num_warps=8,
    )
    return output


def math_prod(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


class _TritonComplexAffineScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, multipliers, increments):
        contiguous_multipliers = multipliers.contiguous()
        contiguous_increments = increments.contiguous()
        states = _launch_forward(
            contiguous_multipliers,
            contiguous_increments,
        )
        ctx.save_for_backward(contiguous_multipliers, states)
        return states

    @staticmethod
    def backward(ctx, state_gradients):
        multipliers, states = ctx.saved_tensors
        state_gradients = state_gradients.resolve_conj().contiguous()
        leading = states.shape[:-3]
        horizons, channels, values = states.shape[-3:]
        batches = math_prod(leading)
        rows = batches * channels * values
        block_rows = 32
        adjoint = torch.empty_like(state_gradients)
        _complex_affine_adjoint_kernel[(triton.cdiv(rows, block_rows),)](
            torch.view_as_real(multipliers),
            torch.view_as_real(state_gradients),
            torch.view_as_real(adjoint),
            rows=rows,
            horizons=horizons,
            channels=channels,
            values=values,
            block_rows=block_rows,
            num_warps=8,
        )
        multiplier_gradient = torch.empty_like(multipliers)
        block_values = triton.next_power_of_2(values)
        _complex_affine_multiplier_gradient_kernel[
            (batches * horizons * channels,)
        ](
            torch.view_as_real(states),
            torch.view_as_real(adjoint),
            torch.view_as_real(multiplier_gradient),
            rows=batches * horizons * channels,
            horizons=horizons,
            channels=channels,
            values=values,
            block_values=block_values,
            num_warps=1,
        )
        return multiplier_gradient, adjoint


def _validate_constant_rotation_inputs(
    step_angles: torch.Tensor,
    increments: torch.Tensor,
    root: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not step_angles.is_cuda or not increments.is_cuda:
        raise ValueError("Triton rotating-frame scan requires CUDA tensors")
    if step_angles.dtype != torch.float32:
        raise ValueError("Triton rotating-frame angles must be float32")
    if increments.dtype != torch.complex64:
        raise ValueError("Triton rotating-frame increments must be complex64")
    if increments.ndim < 3:
        raise ValueError("rotating-frame increments need H/C/V dimensions")
    horizons, channels, values = increments.shape[-3:]
    if horizons < 1 or horizons & (horizons - 1):
        raise ValueError("Triton rotating-frame horizon must be a power of two")
    if step_angles.shape != (channels,):
        raise ValueError("rotating-frame angles must have shape [channels]")
    if root is not None:
        if not root.is_cuda or root.dtype != torch.complex64:
            raise ValueError("rotating-frame root must be CUDA complex64")
        expected_root = (*increments.shape[:-3], channels, values)
        if root.shape != expected_root:
            raise ValueError(
                f"rotating-frame root must have shape {expected_root}"
            )
    return (
        step_angles.contiguous(),
        increments.contiguous(),
        None if root is None else root.contiguous(),
    )


def _launch_constant_rotation_forward(
    step_angles: torch.Tensor,
    increments: torch.Tensor,
    root: torch.Tensor | None,
    angle_sign: int,
) -> torch.Tensor:
    leading = increments.shape[:-3]
    horizons, channels, values = increments.shape[-3:]
    batches = math_prod(leading)
    output = torch.empty_like(increments)
    root_storage = (
        increments.new_empty((0,)) if root is None else root
    )
    if values == 1:
        block_channels = 32
        grid = (batches * triton.cdiv(channels, block_channels),)
        _constant_rotation_forward_channel_kernel[grid](
            step_angles,
            torch.view_as_real(increments),
            torch.view_as_real(root_storage),
            torch.view_as_real(output),
            batches=batches,
            horizons=horizons,
            channels=channels,
            block_channels=block_channels,
            angle_sign=angle_sign,
            has_root=root is not None,
            num_warps=8,
        )
    else:
        block_values = triton.next_power_of_2(values)
        _constant_rotation_forward_value_kernel[(batches * channels,)](
            step_angles,
            torch.view_as_real(increments),
            torch.view_as_real(root_storage),
            torch.view_as_real(output),
            batches=batches,
            horizons=horizons,
            channels=channels,
            values=values,
            block_values=block_values,
            angle_sign=angle_sign,
            has_root=root is not None,
            num_warps=8,
        )
    return output


def _launch_constant_rotation_backward(
    step_angles: torch.Tensor,
    increments: torch.Tensor,
    states: torch.Tensor,
    state_gradients: torch.Tensor,
    *,
    angle_sign: int,
    has_root: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    leading = increments.shape[:-3]
    horizons, channels, values = increments.shape[-3:]
    batches = math_prod(leading)
    step_angle_gradient = torch.zeros_like(step_angles)
    increment_gradient = torch.empty_like(increments)
    root_gradient = (
        torch.empty(
            *leading,
            channels,
            values,
            device=increments.device,
            dtype=increments.dtype,
        )
        if has_root
        else None
    )
    root_gradient_storage = (
        increments.new_empty((0,))
        if root_gradient is None
        else root_gradient
    )
    if values == 1:
        block_channels = 32
        grid = (batches * triton.cdiv(channels, block_channels),)
        _constant_rotation_backward_channel_kernel[grid](
            step_angles,
            torch.view_as_real(increments),
            torch.view_as_real(states),
            torch.view_as_real(state_gradients),
            step_angle_gradient,
            torch.view_as_real(increment_gradient),
            torch.view_as_real(root_gradient_storage),
            batches=batches,
            horizons=horizons,
            channels=channels,
            block_channels=block_channels,
            angle_sign=angle_sign,
            has_root=has_root,
            num_warps=8,
        )
    else:
        block_values = triton.next_power_of_2(values)
        _constant_rotation_backward_value_kernel[(batches * channels,)](
            step_angles,
            torch.view_as_real(increments),
            torch.view_as_real(states),
            torch.view_as_real(state_gradients),
            step_angle_gradient,
            torch.view_as_real(increment_gradient),
            torch.view_as_real(root_gradient_storage),
            batches=batches,
            horizons=horizons,
            channels=channels,
            values=values,
            block_values=block_values,
            angle_sign=angle_sign,
            has_root=has_root,
            num_warps=8,
        )
    return step_angle_gradient, increment_gradient, root_gradient


class _TritonConstantRotationScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, step_angles, increments, root, angle_sign):
        root_value = None if root.numel() == 0 else root
        step_angles, increments, root_value = (
            _validate_constant_rotation_inputs(
                step_angles,
                increments,
                root_value,
            )
        )
        states = _launch_constant_rotation_forward(
            step_angles,
            increments,
            root_value,
            int(angle_sign),
        )
        ctx.save_for_backward(step_angles, increments, states)
        ctx.angle_sign = int(angle_sign)
        ctx.has_root = root_value is not None
        return states

    @staticmethod
    def backward(ctx, state_gradients):
        step_angles, increments, states = ctx.saved_tensors
        state_gradients = state_gradients.resolve_conj().contiguous()
        (
            step_angle_gradient,
            increment_gradient,
            root_gradient,
        ) = _launch_constant_rotation_backward(
            step_angles,
            increments,
            states,
            state_gradients,
            angle_sign=ctx.angle_sign,
            has_root=ctx.has_root,
        )
        if root_gradient is None:
            root_gradient = increments.new_empty((0,))
        return (
            step_angle_gradient,
            increment_gradient,
            root_gradient,
            None,
        )


def triton_complex_affine_scan(
    multipliers: torch.Tensor,
    increments: torch.Tensor,
) -> torch.Tensor:
    """Return exact complex affine prefix states with a fused backward."""
    return _TritonComplexAffineScan.apply(multipliers, increments)


def triton_constant_rotation_scan(
    step_angles: torch.Tensor,
    increments: torch.Tensor,
    root: torch.Tensor | None = None,
    *,
    angle_sign: int = 1,
) -> torch.Tensor:
    """Fuse an exact constant-unitary rotating-frame prefix recurrence.

    ``increments`` has shape ``[..., horizon, channel, value]`` and ``root``,
    when supplied, has shape ``[..., channel, value]``. The returned states
    exactly realize ``x_h = exp(i * sign * angle) x_(h-1) + increment_h``.
    """
    if angle_sign not in (-1, 1):
        raise ValueError("angle_sign must be -1 or 1")
    root_storage = (
        increments.new_empty((0,)) if root is None else root
    )
    return _TritonConstantRotationScan.apply(
        step_angles,
        increments,
        root_storage,
        angle_sign,
    )


__all__ = [
    "triton_complex_affine_scan",
    "triton_constant_rotation_scan",
]
