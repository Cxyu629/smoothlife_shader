use image::{DynamicImage, ImageResult};

use crate::*;

use self::utils::linear_from_points;

mod utils;

#[derive(Resource)]
pub struct GOLKernels {
    pub outer_kernel: Kernel,
    pub inner_kernel: Kernel,
}

pub struct Kernel {
    pub image: DynamicImage,
    pub area: f32,
}

impl Kernel {
    pub fn from_descriptor(kernel_descriptor: KernelDescriptor) -> Self {
        let kernel_size = match kernel_descriptor {
            KernelDescriptor::Circle { radius, antialias_width } => ((radius.ceil() + antialias_width.ceil()) * 2.0) as u32,
            KernelDescriptor::Ring { outer_radius, antialias_width, .. } => ((outer_radius.ceil() + antialias_width.ceil()) * 2.0) as u32,
        };

        let mut image_buffer = RgbaImage::new(kernel_size, kernel_size);

        let activation_function = |x: f32| match kernel_descriptor {
            KernelDescriptor::Circle {
                radius: r,
                antialias_width: b,
            } => {
                let functor = linear_from_points((r - b / 2.0, 1.0), (r + b / 2.0, 0.0));
                functor(x).clamp(0.0, 1.0)
            }
            KernelDescriptor::Ring {
                outer_radius: ro,
                inner_radius: ri,
                antialias_width: b,
            } => {
                let functor_left = linear_from_points((ri - b / 2.0, 0.0), (ri + b / 2.0, 1.0));
                let functor_right = linear_from_points((ro - b / 2.0, 1.0), (ro + b / 2.0, 0.0));
                functor_left(x).min(functor_right(x)).clamp(0.0, 1.0)
            }
        };

        let mut area = 0f32;

        for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
            let center = (
                (kernel_size - 1) as f32 / 2.0,
                (kernel_size - 1) as f32 / 2.0,
            );
            let distance_from_center =
                ((x as f32 - center.0) as f32).hypot((y as f32 - center.1) as f32);
            let value = activation_function(distance_from_center);

            let ivalue = (value * 255.999) as u8;
            let ir = ivalue;
            let ig = ivalue;
            let ib = ivalue;
            let ia = 255;

            area += value;

            *pixel = Rgba([ir, ig, ib, ia]);
        }

        let image = DynamicImage::ImageRgba8(image_buffer);

        Self { image, area }
    }

    #[allow(unused)]
    pub fn save_to(&self, path: &str) -> ImageResult<()> {
        self.image.save(path)
    }
}

pub enum KernelDescriptor {
    Circle {
        radius: f32,
        antialias_width: f32,
    },
    Ring {
        outer_radius: f32,
        inner_radius: f32,
        antialias_width: f32,
    },
}
