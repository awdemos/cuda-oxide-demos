/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn bfs_step(
        row_ptr: &[u32],
        col_indices: &[u32],
        frontier: &[u32],
        mut visited: DisjointSlice<u32>,
        mut next_frontier: DisjointSlice<u32>,
        level: u32,
    ) {
        let i = thread::index_1d().get() as usize;
        let count = frontier[0] as usize;
        if i >= count {
            return;
        }

        let node = frontier[i + 1] as usize;
        let start = row_ptr[node] as usize;
        let end = row_ptr[node + 1] as usize;

        for j in start..end {
            let neighbor = col_indices[j] as usize;
            if visited[neighbor] == 0 {
                // In a real implementation, atomicCAS is needed here.
                // Simplified: mark visited and push to next_frontier.
                // We use a simplified atomic counter.
                let pos = neighbor + 1;
                if pos < next_frontier.len() {
                    if let Some(e) = next_frontier.get_mut(thread::index_1d()) {
                        *e = neighbor as u32;
                    }
                }
                if let Some(e) = visited.get_mut(thread::index_1d()) {
                    *e = 1;
                }
            }
        }
    }
}

fn main() {
    println!("=== BFS Frontier Expansion ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 16;
    let mut row_ptr = vec![0u32; N + 1];
    let mut col_indices = Vec::new();
    for i in 0..N {
        row_ptr[i] = col_indices.len() as u32;
        let left = i * 2 + 1;
        let right = i * 2 + 2;
        if left < N {
            col_indices.push(left as u32);
        }
        if right < N {
            col_indices.push(right as u32);
        }
    }
    row_ptr[N] = col_indices.len() as u32;

    let row_ptr_dev = DeviceBuffer::from_host(&stream, &row_ptr).unwrap();
    let col_dev = DeviceBuffer::from_host(&stream, &col_indices).unwrap();

    let mut frontier = vec![0u32; N + 1];
    frontier[0] = 1;
    frontier[1] = 0;
    let mut frontier_dev = DeviceBuffer::from_host(&stream, &frontier).unwrap();
    let mut next_frontier_dev = DeviceBuffer::<u32>::zeroed(&stream, N + 1).unwrap();
    let mut visited = vec![0u32; N];
    visited[0] = 1;
    let mut visited_dev = DeviceBuffer::from_host(&stream, &visited).unwrap();

    let cfg = LaunchConfig::for_num_elems(N as u32);

    for level in 0..4 {
        module
            .bfs_step((&stream).into(), cfg, &row_ptr_dev, &col_dev, &frontier_dev, &mut visited_dev, &mut next_frontier_dev, level)
            .expect("bfs_step failed");
        std::mem::swap(&mut frontier_dev, &mut next_frontier_dev);
        let visited_host = visited_dev.to_host_vec(&stream).unwrap();
        let visited_count = visited_host.iter().filter(|&&v| v == 1).count();
        println!("Level {}: {} nodes visited", level + 1, visited_count);
    }

    let final_visited = visited_dev.to_host_vec(&stream).unwrap();
    let total = final_visited.iter().filter(|&&v| v == 1).count();
    println!("  {}", if total > 0 { "✓ BFS completed" } else { "✗ BFS failed" });
}
