"""Normalizing Flow Transforms for VITS Text-to-Speech.

This module implements piecewise rational quadratic spline transforms used in
normalizing flows for the VITS model. These transforms provide flexible and
invertible mappings with tractable Jacobian determinants.

The rational quadratic spline is a monotonic piecewise function defined by
bin widths, heights, and derivatives at the knot points. It supports both
forward and inverse transformations.

Key functions:
    - piecewise_rational_quadratic_transform: Main entry point for spline transforms
    - unconstrained_rational_quadratic_spline: Spline with linear tails
    - rational_quadratic_spline: Core spline implementation
    - searchsorted: Binary search for bin indices

References:
    Neural Spline Flows (Durkan et al., 2019)
    https://arxiv.org/abs/1906.04032
"""

import torch
from torch.nn import functional as F

import numpy as np


# Default minimum values to ensure numerical stability
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(inputs, 
                                           unnormalized_widths,
                                           unnormalized_heights,
                                           unnormalized_derivatives,
                                           inverse=False,
                                           tails=None, 
                                           tail_bound=1.,
                                           min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                           min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                           min_derivative=DEFAULT_MIN_DERIVATIVE):
    """Apply piecewise rational quadratic spline transform.
    
    Main entry point for spline-based normalizing flow transforms. Dispatches
    to either constrained or unconstrained spline based on tail configuration.
    
    Args:
        inputs (torch.Tensor): Input tensor to transform.
        unnormalized_widths (torch.Tensor): Unnormalized bin widths (will be softmaxed).
        unnormalized_heights (torch.Tensor): Unnormalized bin heights (will be softmaxed).
        unnormalized_derivatives (torch.Tensor): Unnormalized derivatives at knots.
        inverse (bool, optional): Whether to compute inverse transform. Defaults to False.
        tails (str, optional): Tail behavior ('linear' or None). Defaults to None.
        tail_bound (float, optional): Bound for tail region. Defaults to 1.0.
        min_bin_width (float, optional): Minimum bin width. Defaults to 1e-3.
        min_bin_height (float, optional): Minimum bin height. Defaults to 1e-3.
        min_derivative (float, optional): Minimum derivative value. Defaults to 1e-3.
        
    Returns:
        tuple: (outputs, logabsdet)
            - outputs: Transformed tensor with same shape as inputs
            - logabsdet: Log absolute determinant of Jacobian
    """
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {
            'tails': tails,
            'tail_bound': tail_bound
        }

    outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            **spline_kwargs
    )
    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    """Find bin indices for inputs using binary search.
    
    Finds the index of the bin that each input value falls into.
    A small epsilon is added to the last bin boundary to ensure
    values at the boundary are handled correctly.
    
    Args:
        bin_locations (torch.Tensor): Cumulative bin boundaries.
        inputs (torch.Tensor): Values to find bin indices for.
        eps (float, optional): Small value to add to last boundary. Defaults to 1e-6.
        
    Returns:
        torch.Tensor: Bin indices for each input value.
    """
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def unconstrained_rational_quadratic_spline(inputs,
                                            unnormalized_widths,
                                            unnormalized_heights,
                                            unnormalized_derivatives,
                                            inverse=False,
                                            tails='linear',
                                            tail_bound=1.,
                                            min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                            min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                            min_derivative=DEFAULT_MIN_DERIVATIVE):
    """Apply rational quadratic spline with linear tails outside bounds.
    
    Handles inputs outside the [-tail_bound, tail_bound] interval with
    identity mapping (linear tails), while applying the spline transform
    to inputs inside the interval.
    
    Args:
        inputs (torch.Tensor): Input tensor to transform.
        unnormalized_widths (torch.Tensor): Unnormalized bin widths.
        unnormalized_heights (torch.Tensor): Unnormalized bin heights.
        unnormalized_derivatives (torch.Tensor): Unnormalized derivatives.
        inverse (bool, optional): Compute inverse transform. Defaults to False.
        tails (str, optional): Tail type (only 'linear' supported). Defaults to 'linear'.
        tail_bound (float, optional): Bound for spline region. Defaults to 1.0.
        min_bin_width (float, optional): Minimum bin width. Defaults to 1e-3.
        min_bin_height (float, optional): Minimum bin height. Defaults to 1e-3.
        min_derivative (float, optional): Minimum derivative. Defaults to 1e-3.
        
    Returns:
        tuple: (outputs, logabsdet)
            - outputs: Transformed tensor
            - logabsdet: Log absolute Jacobian determinant
            
    Raises:
        RuntimeError: If unsupported tail type is specified.
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, logabsdet


def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE):
    """Core rational quadratic spline transform implementation.
    
    Implements a piecewise rational quadratic function that is monotonically
    increasing and differentiable. The spline is parameterized by bin widths,
    heights, and derivatives at knot points.
    
    The forward transform maps inputs from [left, right] to [bottom, top].
    The inverse transform maps from [bottom, top] back to [left, right].
    
    Args:
        inputs (torch.Tensor): Input tensor (must be within [left, right] for forward
            or [bottom, top] for inverse).
        unnormalized_widths (torch.Tensor): Unnormalized bin widths, shape [..., num_bins].
        unnormalized_heights (torch.Tensor): Unnormalized bin heights, shape [..., num_bins].
        unnormalized_derivatives (torch.Tensor): Unnormalized derivatives, shape [..., num_bins+1].
        inverse (bool, optional): Compute inverse transform. Defaults to False.
        left (float, optional): Left boundary of input domain. Defaults to 0.
        right (float, optional): Right boundary of input domain. Defaults to 1.
        bottom (float, optional): Bottom boundary of output domain. Defaults to 0.
        top (float, optional): Top boundary of output domain. Defaults to 1.
        min_bin_width (float, optional): Minimum bin width for stability. Defaults to 1e-3.
        min_bin_height (float, optional): Minimum bin height for stability. Defaults to 1e-3.
        min_derivative (float, optional): Minimum derivative for stability. Defaults to 1e-3.
        
    Returns:
        tuple: (outputs, logabsdet)
            - outputs: Transformed tensor with same shape as inputs
            - logabsdet: Log absolute Jacobian determinant (negated for inverse)
            
    Raises:
        ValueError: If inputs are outside the valid domain or if bin constraints
            cannot be satisfied with the given number of bins.
    """
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError('Input to a transform is not within its domain')

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one
                                             - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_plus_one
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet
