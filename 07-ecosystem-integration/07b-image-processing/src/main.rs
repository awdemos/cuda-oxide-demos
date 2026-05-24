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
    pub fn grayscale(rgba: &[u8], mut gray: DisjointSlice<u8>, n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        let r = rgba[i * 4];
        let g = rgba[i * 4 + 1];
        let b = rgba[i * 4 + 2];
        let avg = ((r as u16 + g as u16 + b as u16) / 3) as u8;
        if let Some(e) = gray.get_mut(thread::index_1d()) {
            *e = avg;
        }
    }
}

fn main() {
    println!("=== Image Grayscale ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const W: usize = 64;
    const H: usize = 64;
    let mut rgba = vec![0u8; W * H * 4];
    for y in 0..H {
        for x in 0..W {
            let idx = (y * W + x) * 4;
            if ((x / 8) + (y / 8)) % 2 == 0 {
                rgba[idx] = 255;
                rgba[idx + 1] = 0;
                rgba[idx + 2] = 0;
            } else {
                rgba[idx] = 0;
                rgba[idx + 1] = 0;
                rgba[idx + 2] = 255;
            }
            rgba[idx + 3] = 255;
        }
    }

    let rgba_dev = DeviceBuffer::from_host(&stream, &rgba).unwrap();
    let mut gray_dev = DeviceBuffer::<u8>::zeroed(&stream, W * H).unwrap();

    let cfg = LaunchConfig::for_num_elems((W * H) as u32);
    module
        .grayscale((&stream).into(), cfg, &rgba_dev, &mut gray_dev, (W * H) as u32)
        .expect("grayscale failed");

    let gray_host = gray_dev.to_host_vec(&stream).unwrap();

    let path = "/tmp/cuda-oxide-demo-bw.png";
    if let Err(e) = image::save_buffer(path, &gray_host, W as u32, H as u32, image::ColorType::L8) {
        println!("  Could not save image: {}", e);
    } else {
        println!("  Saved to {}", path);
    }

    let first = gray_host[0];
    let middle = gray_host[W * H / 2];
    let avg = (first as u32 + middle as u32) / 2;
    println!("  {}", if avg == 127 { "✓ PASS" } else { "✗ FAIL" });
}
