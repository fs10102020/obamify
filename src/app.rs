mod calculate;
mod gif_recorder;
mod gui;
mod morph_sim;
mod preset;

#[cfg(target_arch = "wasm32")]
pub use crate::app::calculate::worker::worker_entry;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::{
    num::NonZeroU64,
    sync::{Arc, RwLock},
};

#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::AtomicU32;

use bytemuck::{Pod, Zeroable};
use eframe::CreationContext;
use egui_wgpu::{self, wgpu};
use uuid::Uuid;
use wgpu::util::DeviceExt;

//const INVALID_ID: u32 = 0xFFFF_FFFF;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SeedPos {
    xy: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SeedColor {
    rgba: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsCommon {
    width: u32,
    height: u32,
    n_seeds: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsJfa {
    width: u32,
    height: u32,
    step: u32,
    _pad: u32,
}
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_RESOLUTION: u32 = 2048;

#[cfg(target_arch = "wasm32")]
const DEFAULT_RESOLUTION: u32 = 1024;

/// Top-level GUI mode: image transformation or free drawing.
pub enum GuiMode {
    Transform,
    Draw,
}

use crate::app::{calculate::ProgressMsg, morph_sim::Sim, preset::UnprocessedPreset};
use crate::app::{calculate::util::GenerationSettings, preset::Preset};

#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, collections::VecDeque, rc::Rc};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::closure::Closure;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_arch = "wasm32")]
use web_sys::{Worker, WorkerOptions, WorkerType, js_sys};

/// Interactive obamify application state and renderer resources.
pub struct ObamifyApp {
    //prev_frame_time: std::time::Instant,
    // UI state
    size: (u32, u32),
    seed_count: u32,

    #[cfg(not(target_arch = "wasm32"))]
    progress_tx: mpsc::SyncSender<ProgressMsg>,
    #[cfg(not(target_arch = "wasm32"))]
    progress_rx: mpsc::Receiver<ProgressMsg>,

    #[cfg(target_arch = "wasm32")]
    worker: Option<Worker>,

    #[cfg(target_arch = "wasm32")]
    inbox: Rc<RefCell<VecDeque<ProgressMsg>>>,

    #[cfg(target_arch = "wasm32")]
    pending_images: Rc<RefCell<VecDeque<gui::PendingImageResult>>>,

    gif_recorder: gif_recorder::GifRecorder,
    sim: Sim,

    // Seeds CPU copy
    seeds: Vec<SeedPos>,
    colors: Arc<RwLock<Vec<SeedColor>>>,

    pixeldata: Arc<RwLock<Vec<calculate::drawing_process::PixelData>>>,
    #[cfg(target_arch = "wasm32")]
    drawing_optimizer: Option<calculate::drawing_process::DrawingOptimizer>,

    // EGUI texture id for presenting the shaded RGBA texture
    egui_tex_id: Option<egui::TextureId>,

    // GPU resources (lifetime tied to eframe's RenderState device)
    params_common_buf: wgpu::Buffer,
    params_jfa_buf: wgpu::Buffer,

    // Textures & views
    seed_tex: wgpu::Texture, // Seed positions as texture (WebGL compatible)
    seed_tex_view: wgpu::TextureView,
    color_lookup_tex: wgpu::Texture, // Color lookup table as texture (WebGL compatible)
    color_lookup_tex_view: wgpu::TextureView,

    ids_a: wgpu::Texture,
    ids_b: wgpu::Texture,
    ids_a_view: wgpu::TextureView,
    ids_b_view: wgpu::TextureView,

    // Color (linear storage + srgb view for egui - render target)
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,

    // Pipelines
    seed_splat_pipeline: wgpu::RenderPipeline,
    jfa_pipeline: wgpu::RenderPipeline,
    shade_pipeline: wgpu::RenderPipeline,

    // Bind group layouts
    seed_bgl: wgpu::BindGroupLayout,
    jfa_bgl: wgpu::BindGroupLayout,
    shade_bgl: wgpu::BindGroupLayout,

    // Sampler for texture reads
    nearest_sampler: wgpu::Sampler,

    // Bind groups that are re-created when textures change
    seed_bg: wgpu::BindGroup,
    jfa_bg_a_to_b: wgpu::BindGroup,
    jfa_bg_b_to_a: wgpu::BindGroup,
    shade_bg: wgpu::BindGroup,
    shade_bg_b: wgpu::BindGroup,
    preview_image: Option<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>>,
    stroke_count: u32,

    frame_count: u32,

    gui: gui::GuiState,
    #[cfg(not(target_arch = "wasm32"))]
    current_drawing_id: Arc<AtomicU32>,
    current_filter_mode: wgpu::FilterMode,

    reverse: bool,
}

impl ObamifyApp {
    fn apply_sim_init(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        seed_count: u32,
        seeds: Vec<SeedPos>,
        colors: Vec<SeedColor>,
        sim: Sim,
    ) {
        self.seed_count = seed_count;
        self.seeds = seeds;
        self.sim = sim;

        // Update seed texture (WebGL compatible)
        let (seed_tex, seed_tex_view) =
            Self::make_seed_texture(device, queue, &self.seeds, self.seed_count);
        self.seed_tex = seed_tex;
        self.seed_tex_view = seed_tex_view;

        let params_common = ParamsCommon {
            width: self.size.0,
            height: self.size.1,
            n_seeds: self.seed_count,
            _pad: 0,
        };
        self.params_common_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_common"),
            contents: bytemuck::bytes_of(&params_common),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Update color lookup texture (WebGL compatible)
        let (color_lookup_tex, color_lookup_tex_view) =
            Self::make_color_lookup_texture(device, queue, &colors, self.seed_count);
        self.color_lookup_tex = color_lookup_tex;
        self.color_lookup_tex_view = color_lookup_tex_view;

        *self.colors.write().unwrap_or_else(|e| e.into_inner()) = colors;
        *self.pixeldata.write().unwrap_or_else(|e| e.into_inner()) =
            calculate::drawing_process::PixelData::init_canvas();
        #[cfg(target_arch = "wasm32")]
        {
            self.drawing_optimizer = None;
        }

        self.rebuild_bind_groups(device);
    }

    /// Replaces the active simulation with a saved preset.
    pub fn change_sim(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: Preset,
        change_index: usize,
    ) {
        let (seed_count, mut seeds, colors, mut sim) = morph_sim::init_image(self.size.0, source);
        sim.prepare_play(&mut seeds, self.reverse);
        self.apply_sim_init(device, queue, seed_count, seeds, colors, sim);
        self.gui.current_preset = change_index;
    }

    /// Replaces the active simulation with a drawing canvas source.
    pub fn canvas_sim(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &UnprocessedPreset,
    ) {
        let (seed_count, seeds, colors, sim) = morph_sim::init_canvas(self.size.0, source.clone());
        self.apply_sim_init(device, queue, seed_count, seeds, colors, sim);
    }

    /// Creates an application instance from an `eframe` creation context.
    ///
    /// # Panics
    ///
    /// Panics if the app was not started with the `wgpu` renderer, or if the
    /// built-in preset/image assets are invalid.
    pub fn new(cc: &CreationContext<'_>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be built with the 'wgpu' feature and Renderer::Wgpu")
            .clone();
        let device = &rs.device;
        let size = (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION);

        // get all folders in ../presets
        let presets: Vec<Preset> = if let Some(storage) = cc.storage {
            eframe::get_value::<Vec<Preset>>(storage, "presets")
                .filter(|p| validate_presets(p))
                .unwrap_or_else(get_presets)
        } else {
            get_presets()
        };

        let has_obamified_once = if let Some(storage) = cc.storage {
            eframe::get_value::<bool>(storage, "has_obamified_once").unwrap_or(false)
        } else {
            false
        };

        #[cfg(target_arch = "wasm32")]
        let random_preset = (js_sys::Math::random() * (presets.len() as f64)) as usize;

        #[cfg(not(target_arch = "wasm32"))]
        let random_preset = frand::Rand::with_seed(
            std::time::SystemTime::now().elapsed().unwrap().as_nanos() as u64,
        )
        .gen_range(0..presets.len() as u64) as usize;

        let (seed_count, seeds, colors, sim) =
            morph_sim::init_image(size.0, presets[random_preset].clone());

        // Create textures for WebGL compatibility (no storage buffers in shaders)
        let (seed_tex, seed_tex_view) =
            Self::make_seed_texture(device, &rs.queue, &seeds, seed_count);
        let (color_lookup_tex, color_lookup_tex_view) =
            Self::make_color_lookup_texture(device, &rs.queue, &colors, seed_count);

        let params_common = ParamsCommon {
            width: size.0,
            height: size.1,
            n_seeds: seed_count,
            _pad: 0,
        };
        let params_common_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_common"),
            contents: bytemuck::bytes_of(&params_common),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_jfa = ParamsJfa {
            width: size.0,
            height: size.1,
            step: 1,
            _pad: 0,
        };
        let params_jfa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_jfa"),
            contents: bytemuck::bytes_of(&params_jfa),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === Textures ===
        let (ids_a, ids_a_view) = Self::make_ids_texture(device, size, Some("ids_a"));
        let (ids_b, ids_b_view) = Self::make_ids_texture(device, size, Some("ids_b"));
        let (color_tex, color_view) = Self::make_color_texture(device, size, Some("color"));

        // === Pipelines ===
        let seed_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_seed_splat"),
            entries: &[
                // seed positions texture (WebGL compatible)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // params common
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ParamsCommon>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });

        let jfa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_jfa"),
            entries: &[
                // seed positions texture (WebGL compatible)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // src ids texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // src ids sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // params_jfa
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<ParamsJfa>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let shade_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_shade"),
            entries: &[
                // ids texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // ids sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // seed positions texture (WebGL compatible)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // colors texture (WebGL compatible)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // params common
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<ParamsCommon>() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });

        // Sampler for texture reads
        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Shader modules
        let seed_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("seed_splat.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/seed.wgsl").into()),
        });
        let jfa_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jfa.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/jfa.wgsl").into()),
        });
        let shade_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shade.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shade.wgsl").into()),
        });

        // Pipelines
        let seed_splat_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("seed_splat_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pl_seed"),
                    bind_group_layouts: &[&seed_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &seed_sm,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &seed_sm,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let jfa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jfa_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pl_jfa"),
                    bind_group_layouts: &[&jfa_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &jfa_sm,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &jfa_sm,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let shade_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shade_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pl_shade"),
                    bind_group_layouts: &[&shade_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &shade_sm,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shade_sm,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Bind groups
        let seed_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_seed_splat"),
            layout: &seed_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_common_buf.as_entire_binding(),
                },
            ],
        });

        let jfa_bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_a_to_b"),
            layout: &jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_jfa_buf.as_entire_binding(),
                },
            ],
        });

        let jfa_bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_b_to_a"),
            layout: &jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_jfa_buf.as_entire_binding(),
                },
            ],
        });

        let shade_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_shade"),
            layout: &shade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ids_a_view), // will point to the final ids
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&color_lookup_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_common_buf.as_entire_binding(),
                },
            ],
        });

        let shade_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_shade_b"),
            layout: &shade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&color_lookup_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_common_buf.as_entire_binding(),
                },
            ],
        });

        #[cfg(not(target_arch = "wasm32"))]
        let (progress_tx, progress_rx) = mpsc::sync_channel::<ProgressMsg>(1);

        Self {
            size,
            seed_count,

            seeds,
            colors: Arc::new(RwLock::new(colors)),
            pixeldata: Arc::new(RwLock::new(
                calculate::drawing_process::PixelData::init_canvas(),
            )),
            #[cfg(target_arch = "wasm32")]
            drawing_optimizer: None,
            egui_tex_id: None,
            sim,
            params_common_buf,
            params_jfa_buf,
            seed_tex,
            seed_tex_view,
            color_lookup_tex,
            color_lookup_tex_view,
            ids_a,
            ids_b,
            ids_a_view,
            ids_b_view,
            color_tex,
            color_view,
            seed_splat_pipeline,
            jfa_pipeline,
            shade_pipeline,
            seed_bgl,
            jfa_bgl,
            shade_bgl,
            nearest_sampler,
            seed_bg,
            jfa_bg_a_to_b,
            jfa_bg_b_to_a,
            shade_bg,
            shade_bg_b,
            //prev_frame_time: std::time::Instant::now(),
            #[cfg(not(target_arch = "wasm32"))]
            progress_tx,
            #[cfg(not(target_arch = "wasm32"))]
            progress_rx,
            gif_recorder: gif_recorder::GifRecorder::new(),
            preview_image: None,
            stroke_count: 0,
            gui: gui::GuiState::default(presets, random_preset, has_obamified_once),
            frame_count: 0,
            #[cfg(not(target_arch = "wasm32"))]
            current_drawing_id: Arc::new(AtomicU32::new(0)),
            #[cfg(target_arch = "wasm32")]
            worker: None,
            #[cfg(target_arch = "wasm32")]
            inbox: Rc::new(RefCell::new(VecDeque::new())),
            #[cfg(target_arch = "wasm32")]
            pending_images: Rc::new(RefCell::new(VecDeque::new())),
            current_filter_mode: wgpu::FilterMode::Linear,

            reverse: false,
        }
    }

    /// Returns the latest background processing message, if one is available.
    pub fn get_latest_msg(&mut self) -> Option<ProgressMsg> {
        #[cfg(target_arch = "wasm32")]
        {
            self.inbox.borrow_mut().pop_front()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            match self.progress_rx.try_recv() {
                Ok(msg) => Some(msg),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    eprintln!("progress channel disconnected");
                    None
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    /// Advances the WASM drawing optimizer within a per-frame budget.
    pub fn advance_drawing_optimizer(&mut self) {
        let Some(mut optimizer) = self.drawing_optimizer.take() else {
            return;
        };
        let colors = self.colors.read().unwrap_or_else(|e| e.into_inner());
        let pixel_data = self.pixeldata.read().unwrap_or_else(|e| e.into_inner());
        let swap_budget = ((self.seed_count as usize) / 4).clamp(1024, 8192);
        if let Some(assignments) = optimizer.step(&colors, &pixel_data, swap_budget) {
            self.sim.set_assignments(assignments, self.size.0);
        }
        self.drawing_optimizer = Some(optimizer);
    }

    #[cfg(target_arch = "wasm32")]
    fn ensure_worker(&mut self, _ctx: &egui::Context) {
        if self.worker.is_some() {
            return;
        }
        let inbox = Rc::clone(&self.inbox);

        let worker = {
            let wasm_script_src = js_sys::Reflect::get(
                &js_sys::global(),
                &JsValue::from_str("__wbindgen_script_src"),
            )
            .ok()
            .and_then(|v| v.as_string())
            .and_then(|url| url.rsplit('/').next().map(|s| format!("./{}", s)))
            .unwrap_or_else(|| {
                // Fallback: parse from Error stack trace to find obamify-{hash}.js
                let error = js_sys::Error::new("stack trace");
                if let Ok(stack_val) = js_sys::Reflect::get(&error, &JsValue::from_str("stack")) {
                    if let Some(stack) = stack_val.as_string() {
                        // Look for obamify-{hash}.js in the stack
                        if let Some(start) = stack.find("obamify-") {
                            let rest = &stack[start..];
                            if let Some(end) = rest.find(".js") {
                                let filename = &rest[..end + 3];
                                return format!("./{}", filename);
                            }
                        }
                    }
                }

                String::from("./obamify.js")
            });

            // Use worker.js and pass the script name as a query parameter
            let worker_url = format!("./worker.js?script={}", wasm_script_src);

            let opts = WorkerOptions::new();
            opts.set_type(WorkerType::Module);
            let w = match Worker::new_with_options(&worker_url, &opts) {
                Ok(w) => w,
                Err(e) => {
                    inbox.borrow_mut().push_back(ProgressMsg::Error(format!(
                        "failed to create worker: {e:?}"
                    )));
                    return;
                }
            };

            // ---- onerror: may be ErrorEvent OR a generic Event/JsValue ----
            let error_inbox = Rc::clone(&inbox);
            let onerror = Closure::wrap(Box::new(move |e: JsValue| {
                let mut message = String::from("worker error");
                if let Some(err) = e.dyn_ref::<web_sys::ErrorEvent>() {
                    // Safe: has .message()
                    web_sys::console::error_2(&"worker error:".into(), &err.message().into());
                    message = format!("worker error: {}", err.message());
                    // (Optional) filenames/lineno may be empty on module workers:
                    // web_sys::console::error_3(&"at".into(), &err.filename().into(), &err.lineno().into());
                } else if let Some(ev) = e.dyn_ref::<web_sys::Event>() {
                    // No message property
                    let ty = ev.type_();
                    web_sys::console::error_2(
                        &"worker error (generic Event):".into(),
                        &ty.clone().into(),
                    );
                    message = format!("worker error: {ty}");
                } else {
                    // Something else (could even be undefined/null)
                    web_sys::console::error_1(&JsValue::from_str(&format!(
                        "worker error (unknown): {:?}",
                        js_sys::JSON::stringify(&e).ok()
                    )));
                }
                error_inbox
                    .borrow_mut()
                    .push_back(ProgressMsg::Error(message));
            }) as Box<dyn FnMut(JsValue)>);
            // set_onerror takes a Function; unchecked_ref is fine here
            w.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            onerror.forget();

            // ---- onmessageerror: data failed to deserialize ----
            let message_error_inbox = Rc::clone(&inbox);
            let onmsgerr = Closure::wrap(Box::new(move |e: JsValue| {
                let message = if let Some(me) = e.dyn_ref::<web_sys::MessageEvent>() {
                    web_sys::console::error_2(&"worker messageerror; data:".into(), &me.data());
                    String::from("worker message error")
                } else {
                    web_sys::console::error_1(&"worker messageerror (unknown payload)".into());
                    String::from("worker message error: unknown payload")
                };
                message_error_inbox
                    .borrow_mut()
                    .push_back(ProgressMsg::Error(message));
            }) as Box<dyn FnMut(JsValue)>);
            w.set_onmessageerror(Some(onmsgerr.as_ref().unchecked_ref()));
            onmsgerr.forget();

            w
        };

        //web_sys::console::log_1(&"worker created".into());

        // Receive progress messages
        {
            let message_inbox = Rc::clone(&inbox);
            let onmessage = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
                if let Ok(msg) = serde_wasm_bindgen::from_value::<ProgressMsg>(e.data()) {
                    message_inbox.borrow_mut().push_back(msg);
                }
            }) as Box<dyn FnMut(_)>);
            worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
            onmessage.forget();
        }

        self.worker = Some(worker);
    }

    #[cfg(target_arch = "wasm32")]
    fn start_job(&mut self, src: UnprocessedPreset, settings: GenerationSettings) {
        self.inbox.borrow_mut().clear();
        if let Some(w) = &self.worker {
            let req = calculate::worker::WorkerReq {
                source: src,
                settings,
            };
            let v = match serde_wasm_bindgen::to_value(&req) {
                Ok(v) => v,
                Err(e) => {
                    self.inbox
                        .borrow_mut()
                        .push_back(ProgressMsg::Error(format!("serialization error: {e}")));
                    return;
                }
            };
            match w.post_message(&v) {
                Ok(()) => {}
                Err(e) => {
                    self.inbox
                        .borrow_mut()
                        .push_back(ProgressMsg::Error(format!("postMessage failed: {e:?}")));
                }
            }
        }
    }

    fn make_ids_texture(
        device: &wgpu::Device,
        size: (u32, u32),
        label: Option<&str>,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ids_view"),
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            ..Default::default()
        });
        (tex, view)
    }

    fn make_color_texture(
        device: &wgpu::Device,
        size: (u32, u32),
        label: Option<&str>,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn make_seed_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        seeds: &[SeedPos],
        max_seeds: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        // Pack seeds into a 2D texture to respect WebGL texture size limits (typically 2048-4096)
        // Use a square-ish layout: width = 1024, height = ceil(max_seeds / 1024)
        const TEX_WIDTH: u32 = 1024;
        let tex_height = max_seeds.div_ceil(TEX_WIDTH);

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("seed_positions"),
            size: wgpu::Extent3d {
                width: TEX_WIDTH,
                height: tex_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float, // Store x,y as 2 floats
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload seed data to texture (packed in 2D)
        let mut data = vec![0.0f32; (TEX_WIDTH * tex_height * 2) as usize];
        for (i, seed) in seeds.iter().enumerate() {
            data[i * 2] = seed.xy[0];
            data[i * 2 + 1] = seed.xy[1];
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEX_WIDTH * 8), // 2 floats * 4 bytes per pixel
                rows_per_image: Some(tex_height),
            },
            wgpu::Extent3d {
                width: TEX_WIDTH,
                height: tex_height,
                depth_or_array_layers: 1,
            },
        );

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn update_seed_texture_data(&self, queue: &wgpu::Queue, seeds: &[SeedPos]) {
        // Update seed texture data without recreating the texture
        const TEX_WIDTH: u32 = 1024;
        let tex_height = self.seed_count.div_ceil(TEX_WIDTH);

        let mut data = vec![0.0f32; (TEX_WIDTH * tex_height * 2) as usize];
        for (i, seed) in seeds.iter().enumerate() {
            data[i * 2] = seed.xy[0];
            data[i * 2 + 1] = seed.xy[1];
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.seed_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEX_WIDTH * 8), // 2 floats * 4 bytes per pixel
                rows_per_image: Some(tex_height),
            },
            wgpu::Extent3d {
                width: TEX_WIDTH,
                height: tex_height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn make_color_lookup_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        colors: &[SeedColor],
        max_seeds: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        // Pack colors into a 2D texture to respect WebGL texture size limits
        const TEX_WIDTH: u32 = 1024;
        let tex_height = max_seeds.div_ceil(TEX_WIDTH);

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color_lookup"),
            size: wgpu::Extent3d {
                width: TEX_WIDTH,
                height: tex_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float, // Store RGBA as 4 floats
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload color data to texture (packed in 2D)
        let mut data = vec![0.0f32; (TEX_WIDTH * tex_height * 4) as usize];
        for (i, color) in colors.iter().enumerate() {
            data[i * 4] = color.rgba[0];
            data[i * 4 + 1] = color.rgba[1];
            data[i * 4 + 2] = color.rgba[2];
            data[i * 4 + 3] = color.rgba[3];
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEX_WIDTH * 16), // 4 floats * 4 bytes per pixel
                rows_per_image: Some(tex_height),
            },
            wgpu::Extent3d {
                width: TEX_WIDTH,
                height: tex_height,
                depth_or_array_layers: 1,
            },
        );

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn ensure_registered_texture(
        &mut self,
        rs: &egui_wgpu::RenderState,
        filter_mode: wgpu::FilterMode,
    ) {
        if self.egui_tex_id.is_none() || self.current_filter_mode != filter_mode {
            let id = rs.renderer.write().register_native_texture(
                &rs.device,
                &self.color_view,
                filter_mode,
            );
            self.egui_tex_id = Some(id);
            self.current_filter_mode = filter_mode;
        }
    }

    fn rebuild_bind_groups(&mut self, device: &wgpu::Device) {
        // Rebuild any BGs that reference texture views
        self.seed_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_seed_splat"),
            layout: &self.seed_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_common_buf.as_entire_binding(),
                },
            ],
        });
        self.jfa_bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_a_to_b"),
            layout: &self.jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_jfa_buf.as_entire_binding(),
                },
            ],
        });
        self.jfa_bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_jfa_b_to_a"),
            layout: &self.jfa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_jfa_buf.as_entire_binding(),
                },
            ],
        });
        self.shade_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_shade"),
            layout: &self.shade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ids_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.color_lookup_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_common_buf.as_entire_binding(),
                },
            ],
        });
        self.shade_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_shade_b"),
            layout: &self.shade_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ids_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.nearest_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.seed_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.color_lookup_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_common_buf.as_entire_binding(),
                },
            ],
        });
    }

    fn resize_textures(&mut self, device: &wgpu::Device, new_size: (u32, u32)) {
        self.size = new_size;
        // Recreate textures
        let (ids_a, ids_a_view) = Self::make_ids_texture(device, self.size, Some("ids_a"));
        let (ids_b, ids_b_view) = Self::make_ids_texture(device, self.size, Some("ids_b"));
        let (color_tex, color_view) = Self::make_color_texture(device, self.size, Some("color"));
        self.ids_a = ids_a;
        self.ids_a_view = ids_a_view;
        self.ids_b = ids_b;
        self.ids_b_view = ids_b_view;
        self.color_tex = color_tex;
        self.color_view = color_view;

        // Update params_common
        let params_common = ParamsCommon {
            width: self.size.0,
            height: self.size.1,
            n_seeds: self.seed_count,
            _pad: 0,
        };
        self.params_common_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_common"),
            contents: bytemuck::bytes_of(&params_common),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let params_jfa = ParamsJfa {
            width: self.size.0,
            height: self.size.1,
            step: 1,
            _pad: 0,
        };

        self.params_jfa_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_jfa"),
            contents: bytemuck::bytes_of(&params_jfa),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Force re-registering the egui texture
        self.egui_tex_id = None;
    }

    fn run_gpu(&mut self, rs: &egui_wgpu::RenderState) {
        let device = &rs.device;

        // Prepare commands
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voronoi_jfa_encoder"),
        });

        // 1) Clear ID texture A (where we'll splat seeds)
        {
            let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_ids_a"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ids_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        // 2) Seed splat into A
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("seed_splat"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ids_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.seed_splat_pipeline);
            rpass.set_bind_group(0, &self.seed_bg, &[]);
            rpass.draw(0..self.seed_count, 0..1);
        }

        // 3) JFA passes, ping-pong A<->B

        let max_dim = self.size.0.max(self.size.1);
        let mut step = 1u32;
        while step < max_dim {
            step <<= 1;
        }
        step >>= 1;

        let mut flip = false;
        let mut is_first_jfa_pass = true;
        while step >= 1 {
            let pj = ParamsJfa {
                width: self.size.0,
                height: self.size.1,
                step,
                _pad: 0,
            };
            let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_jfa_staging"),
                contents: bytemuck::bytes_of(&pj),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            encoder.copy_buffer_to_buffer(
                &staging,
                0,
                &self.params_jfa_buf,
                0,
                std::mem::size_of::<ParamsJfa>() as u64,
            );
            {
                // On first pass writing to B, clear it. After that, always load previous content.
                let load_op = if is_first_jfa_pass && !flip {
                    wgpu::LoadOp::Clear(wgpu::Color::WHITE)
                } else {
                    wgpu::LoadOp::Load
                };

                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("jfa_step"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: if !flip {
                            &self.ids_b_view
                        } else {
                            &self.ids_a_view
                        },
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: load_op,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                rpass.set_pipeline(&self.jfa_pipeline);
                rpass.set_bind_group(
                    0,
                    if !flip {
                        &self.jfa_bg_a_to_b
                    } else {
                        &self.jfa_bg_b_to_a
                    },
                    &[],
                );
                rpass.draw(0..4, 0..1);
            }
            is_first_jfa_pass = false;
            flip = !flip;
            step >>= 1;
        }

        // 4) Shade to color (the final IDs are in A if flip is true, else in B).
        // Our shade BG was built with ids_a_view at binding 0. If the last write ended in B,
        // we use the pre-built shade_bg_b bind group for this dispatch.
        let shade_with_b = flip;
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shade"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.shade_pipeline);
            rpass.set_bind_group(
                0,
                if shade_with_b {
                    &self.shade_bg_b
                } else {
                    &self.shade_bg
                },
                &[],
            );
            rpass.draw(0..4, 0..1);
        }

        // Submit
        rs.queue.submit(std::iter::once(encoder.finish()));
    }

    fn stop_recording_gif(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.gif_recorder.stop();
        self.gui.animate = false;
        self.resize_textures(device, (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION));
        self.reset_sim(device, queue);
    }

    fn reset_sim(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.change_sim(
            device,
            queue,
            self.gui.presets[self.gui.current_preset].clone(),
            self.gui.current_preset,
        );
    }

    fn draw(
        &mut self,
        last_mouse_pos: Option<(f32, f32)>,
        mousepos: (f32, f32),
        queue: &wgpu::Queue,
    ) {
        let stroke_id = if last_mouse_pos.is_some() {
            self.stroke_count
        } else {
            self.stroke_count += 1;
            self.stroke_count
        };
        for (i, seedpos) in self.seeds.iter().enumerate() {
            let sx = seedpos.xy[0];
            let sy = seedpos.xy[1];

            let last_mouse_pos = if let Some(a) = last_mouse_pos {
                a
            } else {
                mousepos
            };

            let dist = point_to_line_dist(
                sx,
                sy,
                last_mouse_pos.0,
                last_mouse_pos.1,
                mousepos.0,
                mousepos.1,
            );
            let thickness = if self.gui.drawing_color == [0.0, 0.0, 0.0, DRAWING_ALPHA] {
                30.0
            } else {
                50.0
            };
            let transition = 10.0;
            if dist < thickness + transition {
                let color = self.gui.drawing_color;
                let alpha =
                    ((thickness + transition - dist) / transition).clamp(0.0, 1.0) * color[3];
                let blend = |c1: f32, c2: f32, a: f32| (1.0 - a) * c1 + a * c2;
                let mut colors = self.colors.write().unwrap_or_else(|e| e.into_inner());
                (*colors)[i].rgba[0] = blend((*colors)[i].rgba[0], color[0], alpha);
                (*colors)[i].rgba[1] = blend((*colors)[i].rgba[1], color[1], alpha);
                (*colors)[i].rgba[2] = blend((*colors)[i].rgba[2], color[2], alpha);

                self.sim.cells[i].set_age(0);
                self.sim.cells[i].set_dst_force(0.05 + (stroke_id as f32 * 0.004).sqrt());
                self.sim.cells[i].set_stroke_id(stroke_id);
                self.pixeldata.write().unwrap_or_else(|e| e.into_inner())[i] =
                    calculate::drawing_process::PixelData { stroke_id };

                //self.colors[i].rgba = [0.0, 0.0, 0.0, 1.0];
            }
        }

        // Update the color lookup texture with modified colors
        const TEX_WIDTH: u32 = 1024;
        let tex_height = self.seed_count.div_ceil(TEX_WIDTH);

        let colors = self.colors.read().unwrap_or_else(|e| e.into_inner());
        let mut data = vec![0.0f32; (TEX_WIDTH * tex_height * 4) as usize];
        for (i, color) in colors.iter().enumerate() {
            data[i * 4] = color.rgba[0];
            data[i * 4 + 1] = color.rgba[1];
            data[i * 4 + 2] = color.rgba[2];
            data[i * 4 + 3] = color.rgba[3];
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color_lookup_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEX_WIDTH * 16), // 4 floats * 4 bytes per pixel
                rows_per_image: Some(tex_height),
            },
            wgpu::Extent3d {
                width: TEX_WIDTH,
                height: tex_height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn handle_drawing(
        &mut self,
        ctx: &egui::Context,
        queue: &wgpu::Queue,
        ui: &mut egui::Ui,
        aspect: f32,
    ) {
        // get mouse position over image
        if let Some(pos) = ui.ctx().pointer_interact_pos() {
            let rect = ui.min_rect();

            if rect.contains(pos) {
                let min_y = rect.min.y;
                let min_x = rect.min.x - (rect.height() * aspect - rect.width()) / 2.0;

                let uv = (pos - egui::pos2(min_x, min_y)) / rect.height();
                let img_x = uv.x * self.size.0 as f32;
                let img_y = uv.y * self.size.1 as f32;

                if img_x > 0.0
                    && img_y > 0.0
                    && img_x < self.size.0 as f32
                    && img_y < self.size.1 as f32
                    && ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary))
                {
                    self.draw(self.gui.last_mouse_pos, (img_x, img_y), queue);
                    self.gui.last_mouse_pos = Some((img_x, img_y));
                } else {
                    self.gui.last_mouse_pos = None;
                }
            } else {
                self.gui.last_mouse_pos = None;
            }
        } else {
            self.gui.last_mouse_pos = None;
        }
    }

    fn init_canvas(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let blank = image::load_from_memory(include_bytes!("./app/calculate/blank.png"))
            .unwrap()
            .to_rgb8();

        let settings = GenerationSettings::default(Uuid::new_v4(), "canvas".to_string());
        let source = UnprocessedPreset {
            name: "canvas".to_string(),
            width: blank.width(),
            height: blank.height(),
            source_img: blank.into_raw(),
        };
        self.canvas_sim(device, queue, &source);
        self.gui.animate = true;

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.current_drawing_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            std::thread::spawn({
                let tx = self.progress_tx.clone();
                let colors = Arc::clone(&self.colors);
                let pixel_data = Arc::clone(&self.pixeldata);
                let current_id = self.current_drawing_id.clone();
                let my_id = current_id.load(std::sync::atomic::Ordering::SeqCst);
                let source = source.clone();
                move || {
                    let result = calculate::drawing_process::drawing_process_genetic(
                        source,
                        settings,
                        tx.clone(),
                        colors,
                        pixel_data,
                        my_id,
                        current_id,
                    );
                    match result {
                        Ok(()) => {}
                        Err(err) => {
                            let _ = tx.send(ProgressMsg::Error(err.to_string()));
                        }
                    }
                }
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            let colors = self
                .colors
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            match calculate::drawing_process::DrawingOptimizer::new(source, settings, &colors) {
                Ok(optimizer) => self.drawing_optimizer = Some(optimizer),
                Err(err) => self
                    .inbox
                    .borrow_mut()
                    .push_back(ProgressMsg::Error(err.to_string())),
            }
        }
    }
}

const DRAWING_ALPHA: f32 = 0.5;
fn point_to_line_dist(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    if dx == 0.0 && dy == 0.0 {
        (px - x0).hypot(py - y0)
    } else {
        let t = (((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
        (px - (x0 + t * dx)).hypot(py - (y0 + t * dy))
    }
}

fn validate_presets(presets: &[Preset]) -> bool {
    if presets.is_empty() {
        return false;
    }
    for p in presets {
        let expected_bytes = p.inner.width as usize * p.inner.height as usize * 3;
        if p.inner.source_img.len() != expected_bytes {
            return false;
        }
        if p.inner.width != p.inner.height {
            return false;
        }
        let n = (p.inner.width * p.inner.height) as usize;
        if p.assignments.len() != n {
            return false;
        }
        let mut seen = vec![false; n];
        for &a in &p.assignments {
            if a >= n {
                return false;
            }
            if std::mem::replace(&mut seen[a], true) {
                return false;
            }
        }
    }
    true
}

macro_rules! include_presets {
    ($($name:literal),*) => {
        fn get_presets() -> Vec<Preset> {
            vec![
                $({
                    let img = image::load_from_memory(include_bytes!(concat!(
                        "../presets/",
                        $name,
                        "/source.png"
                    )))
                    .expect("preset source.png must be valid")
                    .to_rgb8();
                    Preset {
                        inner: UnprocessedPreset {
                            name: $name.to_owned(),
                            width: img.width(),
                            height: img.height(),
                            source_img: img.into_raw(),
                        },
                        assignments: include_str!(concat!("../presets/", $name, "/assignments.json"))
                            .to_string()
                            .strip_prefix('[')
                            .expect("assignments.json must start with '['")
                            .strip_suffix(']')
                            .expect("assignments.json must end with ']'")
                            .split(',')
                            .map(|s| s.parse().expect("assignments.json must contain valid integers"))
                            .collect::<Vec<usize>>(),
                    }
                }),*
            ]
        }
    };
}

include_presets! { "wisetree", "blackhole", "cat", "cat2", "colorful" }
