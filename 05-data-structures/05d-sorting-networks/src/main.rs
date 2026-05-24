/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Odd-even transposition sort: alternating compare-and-swap passes
    #[kernel]
    pub fn odd_even_sort(data: &mut DisjointSlice<u32>, n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize - 1 {
            return;
        }
        let phase = thread::blockIdx_y() as usize;
        let even = i % 2 == 0;
        let active = if phase % 2 == 0 { even } else { !even };
        if !active {
            return;
        }

        let a = data.get(thread::index_1d()).unwrap();
        let b = data.get(thread::index_1d().add(1)).unwrap();
        if a > b {
            if let Some(e) = data.get_mut(thread::index_1d()) {
                *e = b;
            }
            if let Some(e) = data.get_mut(thread::index_1d().add(1)) {
                *e = a;
            }
        }
    }
}

fn main() {
    println!("=== Odd-Even Transposition Sort ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 256;
    let mut data: Vec<u32> = (0..N as u32).map(|i| (i * 2654435761u32) ^ (i >> 3)).collect();
    let mut expected = data.clone();
    expected.sort();

    let mut data_dev = DeviceBuffer::from_host(&stream, &data).unwrap();

    for phase in 0..N {
        let cfg = LaunchConfig {
            grid_dim: ((N as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        module
            .odd_even_sort((&stream).into(), cfg, &mut data_dev, N as u32)
            .expect("sort failed");
    }

    let result = data_dev.to_host_vec(&stream).unwrap();
    let ok = result == expected;
    println!("First 8 sorted: {:?}", &result[..8]);
    println!("Last 8 sorted: {:?}", &result[N - 8..]);
    println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
}
