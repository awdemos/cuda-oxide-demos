/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-crate kernel library — defines kernels for use in other crates
//!
//! This library crate exports generic kernels that get monomorphized in the
//! consuming binary crate at compile time.

use std::ops::{Add, Mul};

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Generic scale kernel: out[i] = input[i] * factor
    #[kernel]
    pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = input[idx_raw] * factor;
        }
    }

    /// Generic add kernel: c[i] = a[i] + b[i]
    #[kernel]
    pub fn add<T: Copy + Add<Output = T>>(a: &[T], b: &[T], mut c: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}
