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
    pub fn ray_sphere(
        origins: &[f32],
        directions: &[f32],
        spheres: &[f32],
        mut hit_t: DisjointSlice<f32>,
        n_rays: u32,
        n_spheres: u32,
    ) {
        let ray = thread::index_1d().get() as usize;
        if ray >= n_rays as usize {
            return;
        }

        let ox = origins[ray * 3];
        let oy = origins[ray * 3 + 1];
        let oz = origins[ray * 3 + 2];
        let dx = directions[ray * 3];
        let dy = directions[ray * 3 + 1];
        let dz = directions[ray * 3 + 2];

        let mut min_t = -1.0f32;

        for s in 0..n_spheres as usize {
            let cx = spheres[s * 4];
            let cy = spheres[s * 4 + 1];
            let cz = spheres[s * 4 + 2];
            let r = spheres[s * 4 + 3];

            let ocx = ox - cx;
            let ocy = oy - cy;
            let ocz = oz - cz;

            let a = dx * dx + dy * dy + dz * dz;
            let b = 2.0 * (ocx * dx + ocy * dy + ocz * dz);
            let c = ocx * ocx + ocy * ocy + ocz * ocz - r * r;
            let disc = b * b - 4.0 * a * c;

            if disc >= 0.0 {
                let t = (-b - disc.sqrt()) / (2.0 * a);
                if t >= 0.0 && (min_t < 0.0 || t < min_t) {
                    min_t = t;
                }
            }
        }

        if let Some(e) = hit_t.get_mut(thread::index_1d()) {
            *e = min_t;
        }
    }
}

fn main() {
    println!("=== Ray-Sphere Intersection ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N_RAYS: usize = 1024;
    const N_SPHERES: usize = 8;

    let mut origins = vec![0.0f32; N_RAYS * 3];
    let mut directions = vec![0.0f32; N_RAYS * 3];
    for i in 0..N_RAYS {
        let theta = (i as f32) * 6.283185 / (N_RAYS as f32);
        let phi = (i as f32) * 3.14159 / (N_RAYS as f32);
        origins[i * 3] = (phi.sin() * theta.cos()) * 5.0;
        origins[i * 3 + 1] = phi.cos() * 5.0;
        origins[i * 3 + 2] = (phi.sin() * theta.sin()) * 5.0;
        directions[i * 3] = -origins[i * 3];
        directions[i * 3 + 1] = -origins[i * 3 + 1];
        directions[i * 3 + 2] = -origins[i * 3 + 2];
        let len = (directions[i * 3].powi(2) + directions[i * 3 + 1].powi(2) + directions[i * 3 + 2].powi(2)).sqrt();
        directions[i * 3] /= len;
        directions[i * 3 + 1] /= len;
        directions[i * 3 + 2] /= len;
    }

    let mut spheres = vec![0.0f32; N_SPHERES * 4];
    for s in 0..N_SPHERES {
        let angle = (s as f32) * 6.283185 / (N_SPHERES as f32);
        spheres[s * 4] = angle.cos() * 2.0;
        spheres[s * 4 + 1] = angle.sin() * 2.0;
        spheres[s * 4 + 2] = 0.0;
        spheres[s * 4 + 3] = 0.8;
    }

    let origins_dev = DeviceBuffer::from_host(&stream, &origins).unwrap();
    let directions_dev = DeviceBuffer::from_host(&stream, &directions).unwrap();
    let spheres_dev = DeviceBuffer::from_host(&stream, &spheres).unwrap();
    let mut hit_dev = DeviceBuffer::<f32>::zeroed(&stream, N_RAYS).unwrap();

    let cfg = LaunchConfig::for_num_elems(N_RAYS as u32);
    module
        .ray_sphere((&stream).into(), cfg, &origins_dev, &directions_dev, &spheres_dev, &mut hit_dev, N_RAYS as u32, N_SPHERES as u32)
        .expect("ray_sphere failed");

    let hits = hit_dev.to_host_vec(&stream).unwrap();
    let hit_count = hits.iter().filter(|&&t| t >= 0.0).count();
    println!("Rays: {}, Hit count: {}", N_RAYS, hit_count);
    println!("Hit_t[0..4]: {:?}", &hits[..4]);
    println!("  {}", if hit_count > 0 { "✓ PASS" } else { "✗ FAIL" });
}
