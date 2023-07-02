use std::num::NonZeroU32;

use bevy::render::renderer::RenderQueue;
use image::{DynamicImage, ImageResult};

use crate::*;

use self::utils::lerp;

mod utils;

pub struct GOLKernelPlugin;

impl Plugin for GOLKernelPlugin {
    fn build(&self, app: &mut App) {
        let kernels = GOLKernels {
            outer_kernel: Kernel::from_descriptor(KernelDescriptor::Ring {
                outer_radius: 21.0,
                inner_radius: 7.0,
                antialias_width: 1.0,
            }),
            inner_kernel: Kernel::from_descriptor(KernelDescriptor::Circle {
                radius: 7.0,
                antialias_width: 1.0,
            }),
        };

        if let Err(e) = kernels
            .outer_kernel
            .save_to("smoothlife_shader/assets/outer_kernel.png")
        {
            eprintln!("Image saving error: {}", e);
        };
        if let Err(e) = kernels
            .inner_kernel
            .save_to("smoothlife_shader/assets/inner_kernel.png")
        {
            eprintln!("Image saving error: {}", e);
        };

        let kernel_data = GOLKernelData {
            outer_kernel_area: kernels.outer_kernel.area,
            inner_kernel_area: kernels.inner_kernel.area,
        };

        app.insert_resource(kernel_data);

        // kernel_data.outer_kernel.save_to("smoothlife_shader/assets/outer_kernel.png");
        let outer_kernel = kernels.outer_kernel.image;
        let inner_kernel = kernels.inner_kernel.image;

        let outer_size = Extent3d {
            width: outer_kernel.width(),
            height: outer_kernel.height(),
            depth_or_array_layers: 1,
        };

        let inner_size = Extent3d {
            width: inner_kernel.width(),
            height: inner_kernel.height(),
            depth_or_array_layers: 1,
        };

        let render_device = app.world.get_resource_mut::<RenderDevice>().unwrap();

        let outer_texture = render_device.create_texture(&TextureDescriptor {
            label: None,
            size: outer_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let inner_texture = render_device.create_texture(&TextureDescriptor {
            label: None,
            size: inner_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let render_queue = app.world.resource::<RenderQueue>();
        render_queue.write_texture(
            ImageCopyTexture {
                texture: &outer_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &outer_kernel.to_rgba8(),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * outer_kernel.width()),
                rows_per_image: NonZeroU32::new(outer_kernel.height()),
            },
            outer_size,
        );

        render_queue.write_texture(
            ImageCopyTexture {
                texture: &inner_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &outer_kernel.to_rgba8(),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * outer_kernel.width()),
                rows_per_image: NonZeroU32::new(outer_kernel.height()),
            },
            inner_size,
        );

        let outer_texture_view = outer_texture.create_view(&TextureViewDescriptor::default());
        let inner_texture_view = inner_texture.create_view(&TextureViewDescriptor::default());

        app.insert_resource(GOLKernelTexture {
            outer_texture_view,
            inner_texture_view,
        });
    }
}

#[derive(Resource)]
pub struct GOLKernels {
    pub outer_kernel: Kernel,
    pub inner_kernel: Kernel,
}

#[derive(Resource)]
pub struct GOLKernelData {
    pub outer_kernel_area: f32,
    pub inner_kernel_area: f32,
}

#[derive(Resource)]
pub struct GOLKernelTexture {
    pub outer_texture_view: TextureView,
    pub inner_texture_view: TextureView,
}

pub struct Kernel {
    pub image: DynamicImage,
    pub area: f32,
}

impl Kernel {
    pub fn from_descriptor(kernel_descriptor: KernelDescriptor) -> Self {
        let kernel_size = match kernel_descriptor {
            KernelDescriptor::Circle {
                radius,
                antialias_width,
            } => ((radius.ceil() + antialias_width.ceil()) * 2.0) as u32,
            KernelDescriptor::Ring {
                outer_radius,
                antialias_width,
                ..
            } => ((outer_radius.ceil() + antialias_width.ceil()) * 2.0) as u32,
        };

        let mut image_buffer = RgbaImage::new(kernel_size, kernel_size);

        let activation_function = |x: f32| match kernel_descriptor {
            KernelDescriptor::Circle {
                radius: r,
                antialias_width: b,
            } => lerp(r + b / 2.0, r - b / 2.0, x),
            KernelDescriptor::Ring {
                outer_radius: ro,
                inner_radius: ri,
                antialias_width: b,
            } => lerp(ri - b / 2.0, ri + b / 2.0, x).min(lerp(ro + b / 2.0, ro - b / 2.0, x)),
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
