/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::LaunchConfig;

#[derive(serde::Serialize, serde::Deserialize)]
struct KernelConfig {
    name: String,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared: u32,
}

fn main() {
    println!("=== serde Kernel Config ===\n");

    let cfg = KernelConfig {
        name: "vecadd".into(),
        grid: (16, 1, 1),
        block: (256, 1, 1),
        shared: 0,
    };

    let json = serde_json::to_string_pretty(&cfg).expect("serialize");
    println!("Serialized config:");
    println!("{}", json);

    let decoded: KernelConfig = serde_json::from_str(&json).expect("deserialize");
    let lc = LaunchConfig {
        grid_dim: decoded.grid,
        block_dim: decoded.block,
        shared_mem_bytes: decoded.shared,
    };
    println!("\nDecoded LaunchConfig:");
    println!("  grid_dim  = {:?}", lc.grid_dim);
    println!("  block_dim = {:?}", lc.block_dim);
    println!("  shared    = {}", lc.shared_mem_bytes);
    println!("\n  ✓ PASS");
}
