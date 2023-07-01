use std::num::NonZeroU32;

use bevy::render::renderer::RenderQueue;

use crate::*;
pub struct GameOfLifeComputePlugin;

impl Plugin for GameOfLifeComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugin(ExtractResourcePlugin::<GameOfLifeImage>::default());
        let render_app = app.sub_app_mut(RenderApp);

        // Dealing with kernels is so fussy smh.

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

        kernels.outer_kernel.save_to("smoothlife_shader/assets/outer_kernel.png");
        kernels.inner_kernel.save_to("smoothlife_shader/assets/inner_kernel.png");

        let render_device = render_app.world.get_resource_mut::<RenderDevice>().unwrap();

        // kernel_data.outer_kernel.save_to("smoothlife_shader/assets/outer_kernel.png");
        let outer_kernel = kernels.outer_kernel.image;
        let inner_kernel = kernels.inner_kernel.image;

        let gol_params = GOLParams {
            random_float: rand::random::<f32>(),
            outer_kernel_area: kernels.outer_kernel.area,
            inner_kernel_area: kernels.inner_kernel.area,
        };

        let params = [
            gol_params.random_float,
            gol_params.outer_kernel_area,
            gol_params.inner_kernel_area,
        ];

        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: bevy::core::cast_slice(&params),
        });

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

        let outer_texture = render_device.create_texture(&TextureDescriptor {
            label: None,
            size: outer_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let inner_texture = render_device.create_texture(&TextureDescriptor {
            label: None,
            size: inner_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let render_queue = render_app.world.resource::<RenderQueue>();
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

        render_app.insert_resource(GOLKernelTexture {
            outer_texture_view,
            inner_texture_view,
        });

        render_app.insert_resource(GOLParamsMeta { buffer });

        render_app
            .init_resource::<GameOfLifePipeline>()
            .add_system(queue_bind_group.in_set(RenderSet::Queue));

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("game_of_life", GameOfLifeNode::default());
        render_graph.add_node_edge(
            "game_of_life",
            bevy::render::main_graph::node::CAMERA_DRIVER,
        );
    }
}

#[derive(Resource, Clone, Deref, ExtractResource)]
pub struct GameOfLifeImage(pub Handle<Image>);

#[derive(Resource)]
pub struct GOLKernelTexture {
    pub outer_texture_view: TextureView,
    pub inner_texture_view: TextureView,
}

#[derive(Resource)]
pub struct GOLParamsMeta {
    pub buffer: Buffer,
}

#[derive(Resource)]
pub struct GOLParams {
    pub random_float: f32,
    pub outer_kernel_area: f32,
    pub inner_kernel_area: f32,
}

#[derive(Resource)]
pub struct GameOfLifeImageBindGroup(pub BindGroup);

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<GameOfLifePipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    game_of_life_image: Res<GameOfLifeImage>,
    kernel_textures: Res<GOLKernelTexture>,
    render_device: Res<RenderDevice>,
    params_meta: ResMut<GOLParamsMeta>,
) {
    let view = &gpu_images[&game_of_life_image.0];
    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.texture_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: params_meta.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&kernel_textures.outer_texture_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&kernel_textures.inner_texture_view),
            },
        ],
    });
    commands.insert_resource(GameOfLifeImageBindGroup(bind_group));
}

#[derive(Resource)]
pub struct GameOfLifePipeline {
    pub texture_bind_group_layout: BindGroupLayout,
    pub init_pipeline: CachedComputePipelineId,
    pub update_pipeline: CachedComputePipelineId,
}

impl FromWorld for GameOfLifePipeline {
    fn from_world(world: &mut World) -> Self {
        let texture_bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadWrite,
                                format: TextureFormat::Rgba8Unorm,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: BufferSize::new(
                                    3 * std::mem::size_of::<f32>() as u64,
                                ),
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadOnly,
                                format: TextureFormat::Rgba8Unorm,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::ReadOnly,
                                format: TextureFormat::Rgba8Unorm,
                                view_dimension: TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });
        let shader = world.resource::<AssetServer>().load("smoothlife.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        GameOfLifePipeline {
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

pub enum GameOfLifeState {
    Loading,
    Init,
    Update,
}

pub struct GameOfLifeNode {
    pub state: GameOfLifeState,
}

impl Default for GameOfLifeNode {
    fn default() -> Self {
        Self {
            state: GameOfLifeState::Loading,
        }
    }
}

impl render_graph::Node for GameOfLifeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GameOfLifePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            GameOfLifeState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    self.state = GameOfLifeState::Init;
                }
            }
            GameOfLifeState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = GameOfLifeState::Update;
                }
            }
            GameOfLifeState::Update => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let texture_bind_group = &world.resource::<GameOfLifeImageBindGroup>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GameOfLifePipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, texture_bind_group, &[]);

        // select the pipeline based on the current state
        match self.state {
            GameOfLifeState::Loading => {}
            GameOfLifeState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
            GameOfLifeState::Update => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}
