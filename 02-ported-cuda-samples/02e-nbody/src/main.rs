/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! N-Body Gravitational Simulation — classic CUDA sample ported to cuda-oxide.
//!
//! Demonstrates:
//! - O(n²) pairwise force calculation
//! - Shared-memory tiling of body data (each block loads a tile of bodies)
//! - f32 arithmetic with softening to avoid singularities
//! - Multi-timestep integration with double buffering
//! - Rough energy-conservation check
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 02-ported-cuda-samples/02e-nbody/Cargo.toml

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};

const BLOCK_SIZE: usize = 256;
const TILE_SIZE: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    /// N-body gravitational force kernel with shared-memory tiling.
    ///
    /// Each thread computes the total acceleration on one body by
    /// iterating over all other bodies in tiles loaded into shared
    /// memory. This reduces global-memory traffic by ~TILE_SIZE.
    #[kernel]
    pub fn nbody_timestep(
        n: u32,
        dt: f32,
        softening: f32,
        pos_x_in: &[f32],
        pos_y_in: &[f32],
        pos_z_in: &[f32],
        vel_x_in: &[f32],
        vel_y_in: &[f32],
        vel_z_in: &[f32],
        mass: &[f32],
        mut pos_x_out: DisjointSlice<f32>,
        mut pos_y_out: DisjointSlice<f32>,
        mut pos_z_out: DisjointSlice<f32>,
        mut vel_x_out: DisjointSlice<f32>,
        mut vel_y_out: DisjointSlice<f32>,
        mut vel_z_out: DisjointSlice<f32>,
    ) {
        // Shared-memory tiles for one block of bodies.
        static mut SH_X: SharedArray<f32, TILE_SIZE> = SharedArray::UNINIT;
        static mut SH_Y: SharedArray<f32, TILE_SIZE> = SharedArray::UNINIT;
        static mut SH_Z: SharedArray<f32, TILE_SIZE> = SharedArray::UNINIT;
        static mut SH_M: SharedArray<f32, TILE_SIZE> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bid = thread::blockIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let gid = bid * bdim + tid;
        let n_val = n as usize;

        if gid >= n_val {
            return;
        }

        let my_x = pos_x_in[gid];
        let my_y = pos_y_in[gid];
        let my_z = pos_z_in[gid];

        let mut ax = 0.0f32;
        let mut ay = 0.0f32;
        let mut az = 0.0f32;

        let num_tiles = n_val.div_ceil(TILE_SIZE);
        let mut tile = 0usize;
        while tile < num_tiles {
            let tile_start = tile * TILE_SIZE;
            let load_idx = tile_start + tid;

            // Cooperatively load one tile of body data into shared memory.
            unsafe {
                if load_idx < n_val {
                    SH_X[tid] = pos_x_in[load_idx];
                    SH_Y[tid] = pos_y_in[load_idx];
                    SH_Z[tid] = pos_z_in[load_idx];
                    SH_M[tid] = mass[load_idx];
                } else {
                    SH_X[tid] = 0.0;
                    SH_Y[tid] = 0.0;
                    SH_Z[tid] = 0.0;
                    SH_M[tid] = 0.0;
                }
            }
            thread::sync_threads();

            // Compute forces from all bodies in this tile.
            let mut j = 0usize;
            while j < TILE_SIZE {
                let j_global = tile_start + j;
                if j_global != gid {
                    unsafe {
                        let dx = SH_X[j] - my_x;
                        let dy = SH_Y[j] - my_y;
                        let dz = SH_Z[j] - my_z;
                        let dist_sqr = dx * dx + dy * dy + dz * dz + softening;
                        let inv_dist = 1.0 / dist_sqr.sqrt();
                        let inv_dist3 = inv_dist * inv_dist * inv_dist;
                        let f = SH_M[j] * inv_dist3;
                        ax += f * dx;
                        ay += f * dy;
                        az += f * dz;
                    }
                }
                j += 1;
            }
            thread::sync_threads();
            tile += 1;
        }

        // Integrate velocity and position (symplectic Euler).
        let vx = vel_x_in[gid] + ax * dt;
        let vy = vel_y_in[gid] + ay * dt;
        let vz = vel_z_in[gid] + az * dt;

        let px = pos_x_in[gid] + vx * dt;
        let py = pos_y_in[gid] + vy * dt;
        let pz = pos_z_in[gid] + vz * dt;

        unsafe {
            *vel_x_out.get_unchecked_mut(gid) = vx;
            *vel_y_out.get_unchecked_mut(gid) = vy;
            *vel_z_out.get_unchecked_mut(gid) = vz;
            *pos_x_out.get_unchecked_mut(gid) = px;
            *pos_y_out.get_unchecked_mut(gid) = py;
            *pos_z_out.get_unchecked_mut(gid) = pz;
        }
    }
}

/// Host reference O(n²) n-body step.
fn nbody_host_step(
    n: usize,
    dt: f32,
    softening: f32,
    pos_x: &[f32],
    pos_y: &[f32],
    pos_z: &[f32],
    vel_x: &[f32],
    vel_y: &[f32],
    vel_z: &[f32],
    mass: &[f32],
    px_out: &mut [f32],
    py_out: &mut [f32],
    pz_out: &mut [f32],
    vx_out: &mut [f32],
    vy_out: &mut [f32],
    vz_out: &mut [f32],
) {
    for i in 0..n {
        let mut ax = 0.0f32;
        let mut ay = 0.0f32;
        let mut az = 0.0f32;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = pos_x[j] - pos_x[i];
            let dy = pos_y[j] - pos_y[i];
            let dz = pos_z[j] - pos_z[i];
            let dist_sqr = dx * dx + dy * dy + dz * dz + softening;
            let inv_dist = 1.0 / dist_sqr.sqrt();
            let inv_dist3 = inv_dist * inv_dist * inv_dist;
            let f = mass[j] * inv_dist3;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }
        vx_out[i] = vel_x[i] + ax * dt;
        vy_out[i] = vel_y[i] + ay * dt;
        vz_out[i] = vel_z[i] + az * dt;
        px_out[i] = pos_x[i] + vx_out[i] * dt;
        py_out[i] = pos_y[i] + vy_out[i] * dt;
        pz_out[i] = pos_z[i] + vz_out[i] * dt;
    }
}

fn compute_energy(
    n: usize,
    pos_x: &[f32],
    pos_y: &[f32],
    pos_z: &[f32],
    vel_x: &[f32],
    vel_y: &[f32],
    vel_z: &[f32],
    mass: &[f32],
) -> f32 {
    let mut ke = 0.0f32;
    for i in 0..n {
        let v2 = vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i] + vel_z[i] * vel_z[i];
        ke += 0.5 * mass[i] * v2;
    }

    let mut pe = 0.0f32;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pos_x[j] - pos_x[i];
            let dy = pos_y[j] - pos_y[i];
            let dz = pos_z[j] - pos_z[i];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            pe -= mass[i] * mass[j] / dist;
        }
    }

    ke + pe
}

fn main() {
    println!("=== N-Body Gravitational Simulation ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 512;
    const DT: f32 = 0.001;
    const SOFTENING: f32 = 1e-4;
    const STEPS: usize = 10;

    println!("Bodies:     {}", N);
    println!("Timestep:   {}", DT);
    println!("Softening:  {}", SOFTENING);
    println!("Steps:      {}\n", STEPS);

    // Initialise bodies in a cube with small random velocities.
    let mut pos_x: Vec<f32> = (0..N).map(|i| ((i * 13) % 100) as f32 * 0.1).collect();
    let mut pos_y: Vec<f32> = (0..N).map(|i| ((i * 17) % 100) as f32 * 0.1).collect();
    let mut pos_z: Vec<f32> = (0..N).map(|i| ((i * 19) % 100) as f32 * 0.1).collect();
    let mut vel_x: Vec<f32> = (0..N).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();
    let mut vel_y: Vec<f32> = (0..N).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let mut vel_z: Vec<f32> = (0..N).map(|i| ((i % 3) as f32 - 1.0) * 0.01).collect();
    let mass: Vec<f32> = (0..N).map(|_| 1.0f32).collect();

    // Compute initial energy.
    let energy0 = compute_energy(N, &pos_x, &pos_y, &pos_z, &vel_x, &vel_y, &vel_z, &mass);
    println!("Initial total energy: {:.6}", energy0);

    // Allocate device buffers (double buffered).
    let mut px_a = DeviceBuffer::from_host(&stream, &pos_x).unwrap();
    let mut py_a = DeviceBuffer::from_host(&stream, &pos_y).unwrap();
    let mut pz_a = DeviceBuffer::from_host(&stream, &pos_z).unwrap();
    let mut vx_a = DeviceBuffer::from_host(&stream, &vel_x).unwrap();
    let mut vy_a = DeviceBuffer::from_host(&stream, &vel_y).unwrap();
    let mut vz_a = DeviceBuffer::from_host(&stream, &vel_z).unwrap();
    let m_dev = DeviceBuffer::from_host(&stream, &mass).unwrap();

    let mut px_b = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut py_b = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut pz_b = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut vx_b = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut vy_b = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut vz_b = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let cfg = LaunchConfig {
        grid_dim: ((N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1),
        block_dim: (BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Run timesteps, swapping buffers each step.
    for step in 0..STEPS {
        if step % 2 == 0 {
            module
                .nbody_timestep(
                    &stream, cfg, N as u32, DT, SOFTENING, &px_a, &py_a, &pz_a, &vx_a, &vy_a,
                    &vz_a, &m_dev, &mut px_b, &mut py_b, &mut pz_b, &mut vx_b, &mut vy_b,
                    &mut vz_b,
                )
                .expect("Kernel launch failed");
        } else {
            module
                .nbody_timestep(
                    &stream, cfg, N as u32, DT, SOFTENING, &px_b, &py_b, &pz_b, &vx_b, &vy_b,
                    &vz_b, &m_dev, &mut px_a, &mut py_a, &mut pz_a, &mut vx_a, &mut vy_a,
                    &mut vz_a,
                )
                .expect("Kernel launch failed");
        }
    }

    // Read final state back from whichever buffer was written last.
    let (px_final, py_final, pz_final, vx_final, vy_final, vz_final) = if STEPS % 2 == 0 {
        (
            px_a.to_host_vec(&stream).unwrap(),
            py_a.to_host_vec(&stream).unwrap(),
            pz_a.to_host_vec(&stream).unwrap(),
            vx_a.to_host_vec(&stream).unwrap(),
            vy_a.to_host_vec(&stream).unwrap(),
            vz_a.to_host_vec(&stream).unwrap(),
        )
    } else {
        (
            px_b.to_host_vec(&stream).unwrap(),
            py_b.to_host_vec(&stream).unwrap(),
            pz_b.to_host_vec(&stream).unwrap(),
            vx_b.to_host_vec(&stream).unwrap(),
            vy_b.to_host_vec(&stream).unwrap(),
            vz_b.to_host_vec(&stream).unwrap(),
        )
    };

    // Verify against host reference for the same number of steps.
    let mut hx = pos_x.clone();
    let mut hy = pos_y.clone();
    let mut hz = pos_z.clone();
    let mut hvx = vel_x.clone();
    let mut hvy = vel_y.clone();
    let mut hvz = vel_z.clone();

    let mut hx_n = vec![0.0f32; N];
    let mut hy_n = vec![0.0f32; N];
    let mut hz_n = vec![0.0f32; N];
    let mut hvx_n = vec![0.0f32; N];
    let mut hvy_n = vec![0.0f32; N];
    let mut hvz_n = vec![0.0f32; N];

    for _ in 0..STEPS {
        nbody_host_step(
            N, DT, SOFTENING, &hx, &hy, &hz, &hvx, &hvy, &hvz, &mass, &mut hx_n, &mut hy_n,
            &mut hz_n, &mut hvx_n, &mut hvy_n, &mut hvz_n,
        );
        std::mem::swap(&mut hx, &mut hx_n);
        std::mem::swap(&mut hy, &mut hy_n);
        std::mem::swap(&mut hz, &mut hz_n);
        std::mem::swap(&mut hvx, &mut hvx_n);
        std::mem::swap(&mut hvy, &mut hvy_n);
        std::mem::swap(&mut hvz, &mut hvz_n);
    }

    // Positional accuracy check.
    let mut max_pos_err = 0.0f32;
    for i in 0..N {
        let dx = (px_final[i] - hx[i]).abs();
        let dy = (py_final[i] - hy[i]).abs();
        let dz = (pz_final[i] - hz[i]).abs();
        max_pos_err = max_pos_err.max(dx).max(dy).max(dz);
    }
    println!("Max position error vs host: {:.6e}", max_pos_err);

    // Energy conservation check.
    let energy_final = compute_energy(
        N, &px_final, &py_final, &pz_final, &vx_final, &vy_final, &vz_final, &mass,
    );
    let energy_change = (energy_final - energy0).abs();
    let rel_change = energy_change / energy0.abs();

    println!("Final total energy:   {:.6}", energy_final);
    println!("Absolute energy drift: {:.6e}", energy_change);
    println!("Relative energy drift: {:.6e}", rel_change);

    let pos_ok = max_pos_err < 1e-3;
    let energy_ok = rel_change < 0.05; // 5% tolerance for f32 + few steps

    println!(
        "\n{}  Position match vs host reference",
        if pos_ok { "✓" } else { "✗" }
    );
    println!(
        "{}  Energy conservation (rel. drift < 5%)",
        if energy_ok { "✓" } else { "✗" }
    );

    if pos_ok && energy_ok {
        println!("\n✓ SUCCESS: N-body simulation verified!");
    } else {
        println!("\n✗ FAILED: Verification failed.");
        std::process::exit(1);
    }
}
