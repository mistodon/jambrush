#[cfg(all(target_os = "macos", not(feature = "opengl")))]
extern crate gfx_backend_metal as backend;

#[cfg(all(windows, not(feature = "opengl"), not(feature = "dx11")))]
extern crate gfx_backend_dx12 as backend;

#[cfg(all(unix, not(target_os = "macos"), not(feature = "opengl")))]
extern crate gfx_backend_vulkan as backend;

#[cfg(feature = "opengl")]
extern crate gfx_backend_gl as backend;

#[cfg(feature = "dx11")]
extern crate gfx_backend_dx11 as backend;

mod gfxutils;

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use image::{DynamicImage, GenericImage, RgbaImage};
use rusttype::{gpu_cache::Cache as RTCache, Font as RTFont, PositionedGlyph};
use texture_packer::TexturePacker;
use winit::{EventsLoop, WindowBuilder};

use crate::gfxutils::*;

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const MAX_SPRITE_COUNT: usize = 100000; // TODO: Make dynamic
const MAX_DEPTH: f32 = 10000.0; // TODO: Is this the best we can do? Also we don't even _use_ depth.

fn srgb_to_linear(color: [f32; 4]) -> [f32; 4] {
    const FACTOR: f32 = 2.2;
    let [r, g, b, a] = color;
    [r.powf(FACTOR), g.powf(FACTOR), b.powf(FACTOR), a]
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Vertex {
    pub tint: [f32; 4],
    pub uv: [f32; 2],
    pub offset: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct SpritePushConstants {
    pub transform: [[f32; 4]; 4],
    pub tint: [f32; 4],
    pub uv_origin: [f32; 2],
    pub uv_scale: [f32; 2],
}

#[derive(Debug, Clone)]
struct Glyph {
    pub glyph: PositionedGlyph<'static>,
    pub tint: [f32; 4],
    pub font_id: usize,
}

#[derive(Debug, Clone)]
pub struct Sprite {
    id: usize,
    sub_uv_offset: [f32; 2],
    sub_uv_scale: [f32; 2],
    size: [f32; 2],
}

impl Sprite {
    pub fn size(&self) -> [f32; 2] {
        self.size
    }
}

#[derive(Debug)]
pub struct SpriteSheet {
    id: usize,
    width: usize,
    height: usize,
    size: [f32; 2],
}

impl SpriteSheet {
    pub fn new(sprite: &Sprite, size_in_sprites: [usize; 2]) -> Self {
        let [width, height] = size_in_sprites;
        SpriteSheet {
            id: sprite.id,
            width,
            height,
            size: sprite.size,
        }
    }

    pub fn from_size(sprite: &Sprite, sprite_size: [usize; 2]) -> Self {
        let [sx, sy] = sprite_size;
        let [w, h] = sprite.size;
        let width = w as usize / sx;
        let height = h as usize / sy;
        SpriteSheet {
            id: sprite.id,
            width,
            height,
            size: sprite.size,
        }
    }

    pub fn sprite(&self, coord: [usize; 2]) -> Sprite {
        let [x, y] = coord;
        debug_assert!(
            x < self.width && y < self.height,
            "Sprite {:?} is out of the {:?} range allowed by this SpriteSheet.",
            [x, y],
            [self.width, self.height]
        );
        Sprite {
            id: self.id,
            sub_uv_scale: [1.0 / self.width as f32, 1.0 / self.height as f32],
            sub_uv_offset: [x as f32, y as f32],
            size: [
                self.size[0] / self.width as f32,
                self.size[1] / self.height as f32,
            ],
        }
    }
}

#[derive(Debug)]
pub struct Font {
    id: usize,
}

#[derive(Debug, Default)]
struct DropAlarm(bool);

impl Drop for DropAlarm {
    fn drop(&mut self) {
        if !self.0 {
            panic!("JamBrushSystem dropped without calling `destroy()`");
        }
    }
}

// TODO: all the unwraps...

#[derive(Debug, Default, Clone)]
pub struct JamBrushConfig {
    pub canvas_size: Option<[u32; 2]>,
    pub max_texture_atlas_size: Option<u32>,
    pub logging: bool,
    pub debugging: bool,
    pub debug_texture_atlas: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capture {
    TextureAtlas,
    Canvas,
    // TODO: Window,
}

fn make_transform(pos: [f32; 2], scale: [f32; 2], resolution: [f32; 2]) -> [[f32; 4]; 4] {
    let [sx, sy] = resolution;
    let [w, h] = scale;
    let [x, y] = pos;
    let dx = -1.0 + 2.0 * (x / sx as f32);
    let dy = -1.0 + 2.0 * (y / sy as f32);

    [
        [(w / sx as f32) * 2.0, 0.0, 0.0, 0.0],
        [0.0, (h / sy as f32) * 2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [dx, dy, 0.0, 1.0],
    ]
}

struct InitInfo {
    atlas_size: u32,
}

struct CpuSideCache {
    sprite_textures: Vec<RgbaImage>,
    sprite_regions: Vec<([f32; 2], [f32; 2], [f32; 2])>,
    atlas_image: RgbaImage,
    sprite_vertex_data: Vec<Vertex>,
    fonts: Vec<RTFont<'static>>,
    glyph_cache: RTCache<'static>,
}

struct GpuResources {
    #[cfg(not(feature = "opengl"))]
    _instance: backend::Instance,

    surface: backend::Surface,
    surface_color_format: Format,
    adapter: gfx_hal::Adapter<backend::Backend>,
    device: TDevice,
    queue_group: gfx_hal::QueueGroup<backend::Backend, Graphics>,
    command_pool: gfx_hal::CommandPool<backend::Backend, Graphics>,
    render_pass: TRenderPass,
    set_layout: TDescriptorSetLayout,
    pipeline_layout: TPipelineLayout,
    pipeline: TGraphicsPipeline,
    desc_pool: TDescriptorPool,
    sprites_desc_set: TDescriptorSet,
    blit_desc_set: TDescriptorSet,
    texture_semaphore: TSemaphore,
    scene_semaphore: TSemaphore,
    frame_semaphore: TSemaphore,
    present_semaphore: TSemaphore,
    rtt_image: TImage,
    rtt_memory: TMemory,
    rtt_view: TImageView,
    rtt_sampler: TSampler,
    rtt_framebuffer: TFramebuffer,
    texture_fence: TFence,
    atlas_texture: TImage,
    atlas_memory: TMemory,
    atlas_view: TImageView,
    atlas_sampler: TSampler,
    vertex_buffers: Vec<TBuffer>,
    vertex_memories: Vec<TMemory>,
    swapchain: Option<(
        backend::Swapchain,
        Extent,
        Vec<TImage>,
        Vec<TImageView>,
        Vec<TFramebuffer>,
    )>,
    swapchain_invalidated: bool,
}

impl GpuResources {
    fn destroy(self) {
        let GpuResources {
            device,
            command_pool,
            render_pass,
            set_layout,
            pipeline_layout,
            pipeline,
            desc_pool,
            texture_semaphore,
            scene_semaphore,
            frame_semaphore,
            present_semaphore,
            rtt_image,
            rtt_memory,
            rtt_view,
            rtt_sampler,
            rtt_framebuffer,
            texture_fence,
            atlas_texture,
            atlas_memory,
            atlas_view,
            atlas_sampler,
            vertex_buffers,
            vertex_memories,
            ..
        } = self;

        unsafe {
            device.destroy_sampler(atlas_sampler);
            device.destroy_image_view(atlas_view);
            device.free_memory(atlas_memory);
            device.destroy_image(atlas_texture);
            device.destroy_fence(texture_fence);
            device.destroy_framebuffer(rtt_framebuffer);
            device.destroy_sampler(rtt_sampler);
            device.destroy_image_view(rtt_view);
            device.free_memory(rtt_memory);
            device.destroy_image(rtt_image);
            for buffer in vertex_buffers {
                device.destroy_buffer(buffer);
            }
            for memory in vertex_memories {
                device.free_memory(memory);
            }
            device.destroy_semaphore(present_semaphore);
            device.destroy_semaphore(frame_semaphore);
            device.destroy_semaphore(scene_semaphore);
            device.destroy_semaphore(texture_semaphore);
            device.destroy_descriptor_pool(desc_pool);
            device.destroy_graphics_pipeline(pipeline);
            device.destroy_pipeline_layout(pipeline_layout);
            device.destroy_descriptor_set_layout(set_layout);
            device.destroy_command_pool(command_pool.into_raw());
            device.destroy_render_pass(render_pass);
        }
    }
}

struct WindowData {
    #[cfg(not(feature = "opengl"))]
    window: winit::Window,

    resolution: [u32; 2],
    dpi_factor: f64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RenderStats {
    pub frame_time: Duration,
    pub frame_render_time: Duration,
    pub present_failed: bool,
}

// TODO: Lots. Think about resolution/rebuilding RTT texture
pub struct JamBrushSystem {
    drop_alarm: DropAlarm,

    cpu_cache: CpuSideCache,
    gpu: Option<GpuResources>,
    window_data: WindowData,

    logging: bool,
    debugging: bool,
    recording: Option<PathBuf>,
    recorded_frames: usize,
    sprite_atlas_outdated: bool,
    render_stats: RenderStats,
    time_at_last_frame: Instant,
}

impl JamBrushSystem {
    fn initialize_window(
        window_builder: WindowBuilder,
        events_loop: &EventsLoop,
        canvas_size: Option<[u32; 2]>,
        logging: bool,
        max_texture_atlas_size: Option<u32>,
    ) -> (InitInfo, WindowData, GpuResources) {
        if logging {
            println!("Initializing window and render context");
        }

        #[cfg(not(feature = "opengl"))]
        let (window, inner_size, dpi_factor, _instance, surface, adapter) = {
            let window = window_builder.build(events_loop).unwrap();
            let inner_size = window.get_inner_size().unwrap();
            let dpi_factor = window.get_hidpi_factor();
            let _instance = backend::Instance::create("JamBrush", 1);
            let surface = _instance.create_surface(&window);
            let adapter = _instance.enumerate_adapters().remove(0);
            (window, inner_size, dpi_factor, _instance, surface, adapter)
        };

        #[cfg(feature = "opengl")]
        let (inner_size, dpi_factor, adapter, surface) = {
            // TODO: We probably shouldn't just make a new window...
            let window = {
                let builder = backend::config_context(
                    backend::glutin::ContextBuilder::new(),
                    Format::Rgba8Srgb,
                    None,
                )
                .with_vsync(true);

                backend::glutin::WindowedContext::new_windowed(
                    window_builder,
                    builder,
                    &events_loop,
                )
                .unwrap()
            };

            let inner_size = window.get_inner_size().unwrap();
            let dpi_factor = window.get_hidpi_factor();
            let surface = backend::Surface::from_window(window);
            let adapter = surface.enumerate_adapters().remove(0);
            (inner_size, dpi_factor, adapter, surface)
        };

        let resolution = canvas_size.unwrap_or_else(|| {
            let (window_w, window_h) = inner_size.into();
            [window_w, window_h]
        });

        let (device, queue_group) = adapter
            .open_with::<_, Graphics>(1, |family| surface.supports_queue_family(family))
            .unwrap();

        let command_pool = unsafe {
            device
                .create_command_pool_typed(&queue_group, CommandPoolCreateFlags::empty())
                .unwrap()
        };

        let (_caps, formats, _) = surface.compatibility(&adapter.physical_device);

        let surface_color_format = {
            match formats {
                Some(choices) => choices
                    .into_iter()
                    .find(|format| format.base_format().1 == ChannelType::Srgb)
                    .unwrap(),
                None => Format::Rgba8Srgb,
            }
        };

        let render_pass = unsafe {
            let color_attachment = Attachment {
                format: Some(surface_color_format),
                samples: 1,
                ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::Present,
            };

            let subpass = SubpassDesc {
                colors: &[(0, Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let dependency = SubpassDependency {
                passes: SubpassRef::External..SubpassRef::Pass(0),
                stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT
                    ..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                accesses: Access::empty()
                    ..(Access::COLOR_ATTACHMENT_READ | Access::COLOR_ATTACHMENT_WRITE),
            };

            device
                .create_render_pass(&[color_attachment], &[subpass], &[dependency])
                .unwrap()
        };

        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &[
                        DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: DescriptorType::SampledImage,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: DescriptorType::Sampler,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                    ],
                    &[],
                )
                .unwrap()
        };

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(vec![&set_layout], &[])
                .unwrap()
        };

        let vertex_shader_module = unsafe {
            let spirv = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/compiled/sprite.vert.spv"
            ));
            device.create_shader_module(spirv).unwrap()
        };

        let fragment_shader_module = unsafe {
            let spirv = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/compiled/sprite.frag.spv"
            ));
            device.create_shader_module(spirv).unwrap()
        };

        let pipeline = unsafe {
            let vs_entry = EntryPoint::<backend::Backend> {
                entry: "main",
                module: &vertex_shader_module,
                specialization: Default::default(),
            };

            let fs_entry = EntryPoint::<backend::Backend> {
                entry: "main",
                module: &fragment_shader_module,
                specialization: Default::default(),
            };

            let shader_entries = GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let subpass = Subpass {
                index: 0,
                main_pass: &render_pass,
            };

            let mut pipeline_desc = GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                Rasterizer::FILL,
                &pipeline_layout,
                subpass,
            );

            pipeline_desc
                .blender
                .targets
                .push(ColorBlendDesc(ColorMask::ALL, BlendState::ALPHA));

            pipeline_desc.vertex_buffers.push(VertexBufferDesc {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                rate: VertexInputRate::Vertex,
            });

            pipeline_desc.attributes.push(AttributeDesc {
                location: 0,
                binding: 0,
                element: Element {
                    format: Format::Rgba32Sfloat,
                    offset: 0,
                },
            });

            pipeline_desc.attributes.push(AttributeDesc {
                location: 1,
                binding: 0,
                element: Element {
                    format: Format::Rg32Sfloat,
                    offset: 16,
                },
            });

            pipeline_desc.attributes.push(AttributeDesc {
                location: 2,
                binding: 0,
                element: Element {
                    format: Format::Rgb32Sfloat,
                    offset: 24,
                },
            });

            device
                .create_graphics_pipeline(&pipeline_desc, None)
                .expect("create_graphics_pipeline failed")
        };

        let mut desc_pool = unsafe {
            device
                .create_descriptor_pool(
                    2,
                    &[
                        DescriptorRangeDesc {
                            ty: DescriptorType::SampledImage,
                            count: 2,
                        },
                        DescriptorRangeDesc {
                            ty: DescriptorType::Sampler,
                            count: 2,
                        },
                    ],
                    DescriptorPoolCreateFlags::empty(),
                )
                .unwrap()
        };

        let sprites_desc_set = unsafe { desc_pool.allocate_set(&set_layout).unwrap() };
        let blit_desc_set = unsafe { desc_pool.allocate_set(&set_layout).unwrap() };

        let texture_semaphore = device.create_semaphore().unwrap();
        let scene_semaphore = device.create_semaphore().unwrap();
        let frame_semaphore = device.create_semaphore().unwrap();
        let present_semaphore = device.create_semaphore().unwrap();

        let memory_types = adapter.physical_device.memory_properties().memory_types;

        const QY: f32 = {
            #[cfg(feature = "opengl")]
            {
                0.0
            }
            #[cfg(not(feature = "opengl"))]
            {
                1.0
            }
        };
        let (quad_buffer, quad_memory) = unsafe {
            utils::create_buffer(
                &device,
                &memory_types,
                Properties::CPU_VISIBLE,
                buffer::Usage::VERTEX,
                &[
                    Vertex {
                        offset: [-1.0, -1.0, 0.0],
                        uv: [0.0, 1.0 - QY],
                        tint: WHITE,
                    },
                    Vertex {
                        offset: [-1.0, 1.0, 0.0],
                        uv: [0.0, QY],
                        tint: WHITE,
                    },
                    Vertex {
                        offset: [1.0, -1.0, 0.0],
                        uv: [1.0, 1.0 - QY],
                        tint: WHITE,
                    },
                    Vertex {
                        offset: [-1.0, 1.0, 0.0],
                        uv: [0.0, QY],
                        tint: WHITE,
                    },
                    Vertex {
                        offset: [1.0, 1.0, 0.0],
                        uv: [1.0, QY],
                        tint: WHITE,
                    },
                    Vertex {
                        offset: [1.0, -1.0, 0.0],
                        uv: [1.0, 1.0 - QY],
                        tint: WHITE,
                    },
                ],
            )
        };

        let (sprites_buffer, sprites_memory) = unsafe {
            utils::empty_buffer::<Vertex>(
                &device,
                &memory_types,
                Properties::CPU_VISIBLE,
                buffer::Usage::VERTEX,
                MAX_SPRITE_COUNT * 6,
            )
        };

        let vertex_buffers = vec![quad_buffer, sprites_buffer];
        let vertex_memories = vec![quad_memory, sprites_memory];

        if logging {
            println!("  Canvas size: {} x {}", resolution[0], resolution[1]);
        }

        let (rtt_image, rtt_memory, rtt_view, rtt_sampler, rtt_framebuffer) = unsafe {
            let [width, height] = resolution;
            let extent = Extent {
                width,
                height,
                depth: 1,
            };

            let (rtt_image, rtt_memory, rtt_view) = utils::create_image(
                &device,
                &memory_types,
                extent.width,
                extent.height,
                Format::Rgba8Srgb,
                img::Usage::SAMPLED,
                Aspects::COLOR,
            );

            let rtt_sampler = device
                .create_sampler(img::SamplerInfo::new(Filter::Nearest, WrapMode::Clamp))
                .unwrap();

            let rtt_framebuffer = device
                .create_framebuffer(&render_pass, vec![&rtt_view], extent)
                .unwrap();

            (
                rtt_image,
                rtt_memory,
                rtt_view,
                rtt_sampler,
                rtt_framebuffer,
            )
        };

        let limits = adapter.physical_device.limits();
        let atlas_size_limit = limits.max_image_2d_size;
        let atlas_size = max_texture_atlas_size
            .unwrap_or(atlas_size_limit)
            .min(atlas_size_limit);

        if logging {
            println!(
                "  Texture atlas dimensions: {} x {}",
                atlas_size, atlas_size
            );
        }

        let texture_fence = device.create_fence(false).unwrap();

        let (atlas_texture, atlas_memory, atlas_view, atlas_sampler) = unsafe {
            let (texture_image, texture_memory, texture_view) = utils::create_image(
                &device,
                &memory_types,
                atlas_size,
                atlas_size,
                Format::Rgba8Srgb,
                img::Usage::TRANSFER_DST | img::Usage::SAMPLED,
                Aspects::COLOR,
            );

            let texture_sampler = device
                .create_sampler(img::SamplerInfo::new(Filter::Nearest, WrapMode::Clamp))
                .unwrap();

            // TODO: Maybe allow pre-loading sprites here?

            (texture_image, texture_memory, texture_view, texture_sampler)
        };

        unsafe {
            device.write_descriptor_sets(vec![
                DescriptorSetWrite {
                    set: &blit_desc_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Image(&rtt_view, Layout::Undefined)),
                },
                DescriptorSetWrite {
                    set: &blit_desc_set,
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Sampler(&rtt_sampler)),
                },
            ]);

            device.write_descriptor_sets(vec![
                DescriptorSetWrite {
                    set: &sprites_desc_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Image(&atlas_view, Layout::Undefined)),
                },
                DescriptorSetWrite {
                    set: &sprites_desc_set,
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Sampler(&atlas_sampler)),
                },
            ]);
        }

        let swapchain = None;

        (
            InitInfo { atlas_size },
            WindowData {
                #[cfg(not(feature = "opengl"))]
                window,

                resolution,
                dpi_factor,
            },
            GpuResources {
                #[cfg(not(feature = "opengl"))]
                _instance,

                surface,
                surface_color_format,
                adapter,
                device,
                queue_group,
                command_pool,
                render_pass,
                set_layout,
                pipeline_layout,
                pipeline,
                desc_pool,
                sprites_desc_set,
                blit_desc_set,
                texture_semaphore,
                scene_semaphore,
                frame_semaphore,
                present_semaphore,
                rtt_image,
                rtt_memory,
                rtt_view,
                rtt_sampler,
                rtt_framebuffer,
                texture_fence,
                atlas_texture,
                atlas_memory,
                atlas_view,
                atlas_sampler,
                vertex_buffers,
                vertex_memories,
                swapchain,
                swapchain_invalidated: true,
            },
        )
    }

    pub fn new(
        window_builder: WindowBuilder,
        events_loop: &EventsLoop,
        config: &JamBrushConfig,
    ) -> Self {
        if config.logging {
            println!("Constructing JamBrushSystem:");
        }

        let (init_info, window_data, gpu) = Self::initialize_window(
            window_builder,
            events_loop,
            config.canvas_size,
            config.logging,
            config.max_texture_atlas_size,
        );

        let mut atlas_image =
            DynamicImage::new_rgba8(init_info.atlas_size, init_info.atlas_size).to_rgba();

        if config.debug_texture_atlas {
            for (x, y, pixel) in atlas_image.enumerate_pixels_mut() {
                let r = x & 255;
                let g = x >> 8;
                let b = y & 255;
                let a = y >> 8;
                *pixel = image::Rgba([r as u8, g as u8, b as u8, a as u8]);
            }
        }

        let glyph_cache = RTCache::builder()
            .dimensions(init_info.atlas_size, init_info.atlas_size / 2)
            .position_tolerance(0.1)
            .scale_tolerance(0.1)
            .pad_glyphs(true)
            .multithread(true)
            .build();

        JamBrushSystem {
            drop_alarm: DropAlarm(false),

            cpu_cache: CpuSideCache {
                sprite_textures: vec![],
                sprite_regions: vec![],
                atlas_image,
                sprite_vertex_data: Vec::with_capacity(MAX_SPRITE_COUNT * 6),
                fonts: vec![],
                glyph_cache,
            },

            gpu: Some(gpu),
            window_data,
            logging: config.logging,
            debugging: config.debugging,
            recording: None,
            recorded_frames: 0,
            sprite_atlas_outdated: false,
            render_stats: RenderStats::default(),
            time_at_last_frame: Instant::now(),
        }
    }

    pub fn rebuild_window(&mut self, window_builder: WindowBuilder, events_loop: &EventsLoop) {
        let old_gpu = self.gpu.take().unwrap();
        old_gpu.destroy();

        let (_, window_data, gpu) = Self::initialize_window(
            window_builder,
            events_loop,
            Some(self.window_data.resolution),
            self.logging,
            Some(self.cpu_cache.atlas_image.width()),
        );

        self.window_data = window_data;
        self.gpu = Some(gpu);
        self.upload_atlas_texture();
    }

    pub fn destroy(mut self) {
        self.drop_alarm.0 = true;

        if self.gpu.as_mut().unwrap().swapchain.is_some() {
            self.destroy_swapchain();
        }

        let JamBrushSystem { gpu, .. } = self;

        if let Some(gpu) = gpu {
            gpu.destroy();
        }
    }

    fn log<S: AsRef<str>>(&self, message: S) {
        if self.logging {
            println!("JamBrush: {}", message.as_ref())
        }
    }

    pub fn start_rendering(
        &mut self,
        canvas_clear_color: [f32; 4],
        border_clear_color: Option<[f32; 4]>,
    ) -> Renderer {
        if self.sprite_atlas_outdated {
            self.update_sprite_atlas();
        }

        Renderer::new(
            self,
            canvas_clear_color,
            border_clear_color.unwrap_or(canvas_clear_color),
        )
    }

    pub fn window(&self) -> &winit::Window {
        #[cfg(not(feature = "opengl"))]
        {
            &self.window_data.window
        }

        #[cfg(feature = "opengl")]
        {
            unimplemented!("Cannot retrieve Window with OpenGL backing just yet")
        }
    }

    pub fn window_resized(&mut self, _resolution: (u32, u32)) {
        // TODO: Only invalidate if changed
        let gpu = self.gpu.as_mut().unwrap();
        gpu.swapchain_invalidated = true;

        #[cfg(feature = "opengl")]
        {
            gpu.surface.get_window().resize(_resolution.into());
        }
        self.log("Swapchain invalidated: window resized");
    }

    pub fn dpi_factor_changed(&mut self, dpi_factor: f64) {
        if self.window_data.dpi_factor != dpi_factor {
            self.window_data.dpi_factor = dpi_factor;
            let gpu = self.gpu.as_mut().unwrap();
            gpu.swapchain_invalidated = true;
            self.log("Swapchain invalidated: DPI factor changed");
        }
    }

    pub fn load_sprite_file<P: AsRef<Path>>(&mut self, path: P) -> Sprite {
        let image_bytes = std::fs::read(path.as_ref()).unwrap();

        self.load_sprite(&image_bytes)
    }

    pub fn load_sprite_rgba(&mut self, size: [u32; 2], rgba_bytes: &[u8]) -> Sprite {
        let sprite_index = self.cpu_cache.sprite_textures.len();
        {
            let sprite_img: RgbaImage =
                RgbaImage::from_raw(size[0], size[1], rgba_bytes.to_owned())
                    .expect("Failed to create image from bytes");
            self.cpu_cache.sprite_textures.push(sprite_img);
        }

        self.sprite_atlas_outdated = true;

        self.log(format!(
            "Sprite {} ({} x {}) loaded and queued for upload",
            sprite_index, size[0], size[1]
        ));

        Sprite {
            id: sprite_index,
            sub_uv_scale: [1.0, 1.0],
            sub_uv_offset: [0.0, 0.0],
            size: [size[0] as f32, size[1] as f32],
        }
    }

    pub fn load_sprite(&mut self, image_bytes: &[u8]) -> Sprite {
        let image = image::load_from_memory(&image_bytes).unwrap().to_rgba();
        let (w, h) = image.dimensions();

        self.load_sprite_rgba([w, h], &image)
    }

    pub fn load_font_file<P: AsRef<Path>>(&mut self, path: P) -> Font {
        let font_bytes = std::fs::read(path.as_ref()).unwrap();
        let font = RTFont::from_bytes(font_bytes).unwrap();
        let font_index = self.cpu_cache.fonts.len();
        self.cpu_cache.fonts.push(font);

        Font { id: font_index }
    }

    pub fn load_font(&mut self, font_bytes: &[u8]) -> Font {
        let font = RTFont::from_bytes(font_bytes.to_owned()).unwrap();
        let font_index = self.cpu_cache.fonts.len();
        self.cpu_cache.fonts.push(font);

        Font { id: font_index }
    }

    fn update_sprite_atlas(&mut self) {
        use texture_packer::TexturePackerConfig;

        self.log("Updating sprite atlas...");

        let (aw, ah) = self.cpu_cache.atlas_image.dimensions();
        let atlas_config = TexturePackerConfig {
            max_width: aw,
            max_height: ah / 2,
            allow_rotation: false,
            trim: false,
            texture_outlines: self.debugging,
            ..Default::default()
        };

        {
            let mut atlas_packer = TexturePacker::new_skyline(atlas_config);

            for (index, texture) in self.cpu_cache.sprite_textures.iter().enumerate() {
                // TODO: ugh, string keys?
                atlas_packer.pack_ref(index.to_string(), texture);
            }

            self.cpu_cache.sprite_regions.clear();
            for (i, texture) in self.cpu_cache.sprite_textures.iter().enumerate() {
                let frame = atlas_packer
                    .get_frame(&i.to_string())
                    .expect("Failed to get frame in atlas for sprite")
                    .frame;
                let x = frame.x as f32 / aw as f32;
                let y = frame.y as f32 / ah as f32;
                let w = frame.w as f32 / aw as f32;
                let h = frame.h as f32 / ah as f32;
                let (pw, ph) = texture.dimensions();
                self.cpu_cache
                    .sprite_regions
                    .push(([x, y], [w, h], [pw as f32, ph as f32]));
                self.cpu_cache
                    .atlas_image
                    .copy_from(texture, frame.x, frame.y);
            }
        }

        self.upload_atlas_texture();
        self.sprite_atlas_outdated = false;

        self.log("Updated sprite atlas.");
    }

    fn upload_atlas_texture(&mut self) {
        let gpu = self.gpu.as_mut().unwrap();
        unsafe {
            utils::upload_image_data(
                &gpu.device,
                &gpu.adapter.physical_device,
                &mut gpu.command_pool,
                &mut gpu.queue_group.queues[0],
                &gpu.texture_fence,
                &self.cpu_cache.atlas_image,
                &gpu.atlas_texture,
            );
        }
    }

    fn update_swapchain(&mut self) {
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.swapchain_invalidated && gpu.swapchain.is_some() {
            self.destroy_swapchain();
        }

        let gpu = self.gpu.as_mut().unwrap();
        if gpu.swapchain.is_none() {
            self.create_swapchain();
        }
    }

    fn destroy_swapchain(&mut self) {
        self.log("Destroying swapchain");
        let gpu = self.gpu.as_mut().unwrap();

        let (swapchain, _extent, _frame_images, frame_views, framebuffers) =
            gpu.swapchain.take().unwrap();

        unsafe {
            gpu.device.wait_idle().unwrap();
            gpu.command_pool.reset();

            for framebuffer in framebuffers {
                gpu.device.destroy_framebuffer(framebuffer);
            }

            for image_view in frame_views {
                gpu.device.destroy_image_view(image_view);
            }

            gpu.device.destroy_swapchain(swapchain);
        }
    }

    fn create_swapchain(&mut self) {
        self.log("Creating swapchain");
        let gpu = self.gpu.as_mut().unwrap();

        gpu.swapchain_invalidated = false;
        let (caps, _, _) = gpu.surface.compatibility(&gpu.adapter.physical_device);

        let swap_config = SwapchainConfig::from_caps(
            &caps,
            gpu.surface_color_format,
            Extent2D {
                width: self.window_data.resolution[0],
                height: self.window_data.resolution[1],
            },
        );

        self.log(format!("  Present mode: {:?}", swap_config.present_mode));

        self.log(format!("  DPI factor: {}", self.window_data.dpi_factor));

        self.log(format!(
            "  Swapchain dimensions: {} x {}",
            swap_config.extent.width, swap_config.extent.height
        ));

        unsafe {
            let gpu = self.gpu.as_mut().unwrap();
            let extent = swap_config.extent.to_extent();
            let (swapchain, backbuffer) = gpu
                .device
                .create_swapchain(&mut gpu.surface, swap_config, None)
                .unwrap();

            let (frame_images, frame_views, framebuffers) = {
                let color_range = SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                };

                let image_views = backbuffer
                    .iter()
                    .map(|image| {
                        gpu.device
                            .create_image_view(
                                image,
                                ViewKind::D2,
                                gpu.surface_color_format,
                                Swizzle::NO,
                                color_range.clone(),
                            )
                            .unwrap()
                    })
                    .collect::<Vec<_>>();

                let fbos = image_views
                    .iter()
                    .map(|image_view| {
                        gpu.device
                            .create_framebuffer(&gpu.render_pass, vec![image_view], extent)
                            .unwrap()
                    })
                    .collect();

                (backbuffer, image_views, fbos)
            };

            gpu.swapchain = Some((swapchain, extent, frame_images, frame_views, framebuffers));
        }
    }

    pub fn capture_image(&mut self, capture_type: Capture) -> ([u32; 2], Vec<u8>) {
        let gpu = self.gpu.as_mut().unwrap();
        unsafe {
            gpu.device.wait_idle().unwrap();
            gpu.device.reset_fence(&gpu.texture_fence).unwrap();

            let swizzle = false;
            let image = match capture_type {
                Capture::Canvas => &gpu.rtt_image, // TODO: Clear up image/texture confusion
                Capture::TextureAtlas => &gpu.atlas_texture,
            };

            let footprint = gpu.device.get_image_subresource_footprint(
                image,
                Subresource {
                    aspects: Aspects::COLOR,
                    level: 0,
                    layer: 0,
                },
            );
            let memory_size = footprint.slice.end - footprint.slice.start;
            let memory_types = gpu.adapter.physical_device.memory_properties().memory_types;

            // TODO: Are these deleted?
            let (screenshot_buffer, screenshot_memory) = utils::empty_buffer::<u8>(
                &gpu.device,
                &memory_types,
                Properties::CPU_VISIBLE,
                buffer::Usage::TRANSFER_DST,
                memory_size as usize,
            );

            let width = footprint.row_pitch as u32 / 4;
            let height = memory_size as u32 / footprint.row_pitch as u32;

            let submit = {
                let mut cmd_buffer = gpu.command_pool.acquire_command_buffer::<OneShot>();
                cmd_buffer.begin();

                cmd_buffer.copy_image_to_buffer(
                    image,
                    Layout::TransferSrcOptimal,
                    &screenshot_buffer,
                    &[BufferImageCopy {
                        buffer_offset: 0,
                        buffer_width: width,
                        buffer_height: height,
                        image_layers: SubresourceLayers {
                            aspects: Aspects::COLOR,
                            level: 0,
                            layers: 0..1,
                        },
                        image_offset: Offset { x: 0, y: 0, z: 0 },
                        image_extent: Extent {
                            width,
                            height,
                            depth: 1,
                        },
                    }],
                );

                cmd_buffer.finish();
                cmd_buffer
            };

            gpu.queue_group.queues[0].submit_nosemaphores(&[submit], Some(&gpu.texture_fence));

            gpu.device.wait_for_fence(&gpu.texture_fence, !0).unwrap();

            let image_bytes = {
                let data = gpu
                    .device
                    .acquire_mapping_reader::<u8>(&screenshot_memory, 0..memory_size)
                    .expect("acquire_mapping_reader failed");

                let mut image_bytes: Vec<u8> = data.to_owned();

                if swizzle {
                    for chunk in image_bytes.chunks_mut(4) {
                        let (r, rest) = chunk.split_first_mut().unwrap();
                        std::mem::swap(r, &mut rest[1]);
                    }
                }

                gpu.device.release_mapping_reader(data);

                image_bytes
            };

            ([width, height], image_bytes)
        }
    }

    pub fn capture_to_file<P: AsRef<Path>>(&mut self, capture_type: Capture, path: P) {
        use image::ColorType;

        let (size, image_bytes) = self.capture_image(capture_type);
        image::save_buffer(path, &image_bytes, size[0], size[1], ColorType::RGBA(8)).unwrap();
    }

    pub fn recording(&mut self) -> bool {
        self.recording.is_some()
    }

    pub fn start_recording<P: AsRef<Path>>(&mut self, output_dir: P) {
        self.log(format!(
            "Started recording frames to: {}",
            output_dir.as_ref().display()
        ));
        std::fs::create_dir_all(output_dir.as_ref())
            .expect("Failed to create output directory for video frame capture");
        self.recording = Some(output_dir.as_ref().to_owned());
    }

    pub fn stop_recording(&mut self) {
        self.log("Stopped recording frames");
        self.recording = None;
    }

    pub fn render_stats(&self) -> RenderStats {
        self.render_stats
    }
}

pub struct Renderer<'a> {
    draw_system: &'a mut JamBrushSystem,
    canvas_clear_color: [f32; 4],
    frame_index: SwapImageIndex,
    blit_command_buffer: Option<CommandBuffer<backend::Backend, Graphics, OneShot, Primary>>,
    sprites: Vec<(f32, SpritePushConstants)>,
    glyphs: Vec<(f32, Glyph)>,
    finished: bool,
    camera: [f32; 2],
}

impl<'a> Renderer<'a> {
    fn new(
        draw_system: &'a mut JamBrushSystem,
        canvas_clear_color: [f32; 4],
        border_clear_color: [f32; 4],
    ) -> Self {
        unsafe {
            // TODO: See next TODO
            draw_system.update_swapchain();

            let gpu = draw_system.gpu.as_mut().unwrap();
            gpu.command_pool.reset();
        }

        let frame_index: SwapImageIndex;
        let blit_command_buffer: CommandBuffer<backend::Backend, Graphics, OneShot, Primary>;

        unsafe {
            let gpu = draw_system.gpu.as_mut().unwrap();
            let (swapchain, extent, _frame_images, _frame_views, framebuffers) =
                gpu.swapchain.as_mut().unwrap();

            // TODO: handle failure
            let (index, _) = swapchain
                .acquire_image(!0, Some(&gpu.frame_semaphore), None)
                .unwrap();
            frame_index = index;

            blit_command_buffer = {
                let mut command_buffer = gpu.command_pool.acquire_command_buffer::<OneShot>();
                command_buffer.begin();

                let [vwidth, vheight] = draw_system.window_data.resolution;

                let base_width = (f64::from(vwidth) * draw_system.window_data.dpi_factor) as u32;
                let base_height = (f64::from(vheight) * draw_system.window_data.dpi_factor) as u32;

                let integer_scale =
                    std::cmp::min(extent.width / base_width, extent.height / base_height);

                let (viewport_width, viewport_height) = if integer_scale == 0 {
                    let viewport_width =
                        std::cmp::min(extent.width, (extent.height * vwidth) / vheight);
                    let viewport_height =
                        std::cmp::min(extent.height, (extent.width * vheight) / vwidth);
                    (viewport_width, viewport_height)
                } else {
                    let viewport_width = base_width * integer_scale;
                    let viewport_height = base_height * integer_scale;
                    (viewport_width, viewport_height)
                };

                let viewport_x = (extent.width - viewport_width) / 2;
                let viewport_y = (extent.height - viewport_height) / 2;

                let viewport = Viewport {
                    rect: Rect {
                        x: viewport_x as i16,
                        y: viewport_y as i16,
                        w: viewport_width as i16,
                        h: viewport_height as i16,
                    },
                    depth: 0.0..1.0,
                };

                command_buffer.set_viewports(0, &[viewport.clone()]);
                command_buffer.set_scissors(0, &[viewport.rect]);

                command_buffer.bind_graphics_pipeline(&gpu.pipeline);
                command_buffer.bind_vertex_buffers(0, vec![(&gpu.vertex_buffers[0], 0)]);
                command_buffer.bind_graphics_descriptor_sets(
                    &gpu.pipeline_layout,
                    0,
                    vec![&gpu.blit_desc_set],
                    &[],
                );

                {
                    let mut encoder = command_buffer.begin_render_pass_inline(
                        &gpu.render_pass,
                        &framebuffers[frame_index as usize],
                        viewport.rect,
                        &[ClearValue::Color(ClearColor::Float(border_clear_color))],
                    );

                    encoder.draw(0..6, 0..1);
                }

                command_buffer.finish();
                command_buffer
            };
        }

        Renderer {
            draw_system,
            canvas_clear_color,
            frame_index,
            blit_command_buffer: Some(blit_command_buffer),
            sprites: vec![],
            glyphs: vec![],
            finished: false,
            camera: [0.0, 0.0],
        }
    }

    pub fn camera(&mut self, camera: [f32; 2]) {
        self.camera = camera;
    }

    pub fn clear_camera(&mut self) {
        self.camera([0.0, 0.0]);
    }

    pub fn center_camera(&mut self, on: [f32; 2]) {
        let [rx, ry] = self.draw_system.window_data.resolution;
        let [x, y] = on;
        self.camera([x - rx as f32 / 2.0, y - ry as f32 / 2.0]);
    }

    pub fn sprite<T: Into<SpriteArgs>>(&mut self, sprite: &Sprite, args: T) {
        let args = args.into();
        self.sprite_with(sprite, &args);
    }

    pub fn sprite_with(&mut self, sprite: &Sprite, args: &SpriteArgs) {
        let (uv_origin, uv_scale, pixel_scale) =
            self.draw_system.cpu_cache.sprite_regions[sprite.id];
        let [res_x, res_y] = self.draw_system.window_data.resolution;

        let [px, py] = pixel_scale;
        let [sx, sy] = sprite.sub_uv_scale;

        let scale = args.size.unwrap_or([px * sx, py * sy]);
        let tint = srgb_to_linear(args.tint);

        let uw = uv_scale[0] * sx;
        let uh = uv_scale[1] * sy;
        let u0 = uv_origin[0] + uw * sprite.sub_uv_offset[0];
        let v0 = uv_origin[1] + uh * sprite.sub_uv_offset[1];

        let [pos_x, pos_y] = args.pos;
        let [cam_x, cam_y] = self.camera;

        let data = SpritePushConstants {
            transform: make_transform(
                [pos_x - cam_x, pos_y - cam_y],
                scale,
                [res_x as f32, res_y as f32],
            ),
            tint,
            uv_origin: [u0, v0],
            uv_scale: [uw, uh],
        };

        self.sprites.push((args.depth, data));
    }

    pub fn text<T: Into<TextArgs>, S: AsRef<str>>(
        &mut self,
        font: &Font,
        size: f32,
        text: S,
        args: T,
    ) {
        let args = args.into();
        self.text_with(font, size, text, &args);
    }

    pub fn text_with<S: AsRef<str>>(&mut self, font: &Font, size: f32, text: S, args: &TextArgs) {
        use rusttype::{Point, Scale};

        let tint = srgb_to_linear(args.tint);

        // TODO: scale/pos are in pixels - but should be in abstract screen-space units
        // TODO: copyin' a lotta glyphs here!

        let [cam_x, cam_y] = self.camera;

        let font_id = font.id;
        let font = &self.draw_system.cpu_cache.fonts[font_id];
        let glyphs = font.layout(
            text.as_ref(),
            Scale { x: size, y: size },
            Point {
                x: args.pos[0] - cam_x,
                y: args.pos[1] - cam_y,
            },
        );

        for glyph in glyphs {
            let glyph = glyph.standalone();
            self.draw_system
                .cpu_cache
                .glyph_cache
                .queue_glyph(font_id, glyph.clone());
            self.glyphs.push((
                args.depth,
                Glyph {
                    glyph,
                    tint,
                    font_id,
                },
            ));
        }
    }

    fn update_font_atlas(&mut self) {
        let glyph_cache = &mut self.draw_system.cpu_cache.glyph_cache;
        let font_atlas_image = &mut self.draw_system.cpu_cache.atlas_image;

        let atlas_height = font_atlas_image.height();
        let mut modified = false;

        glyph_cache
            .cache_queued(|dest_rect, data| {
                use rusttype::Point;

                let Point { x, y } = dest_rect.min;
                let w = dest_rect.width();
                let h = dest_rect.height();

                let mut rgba_buffer = vec![0; data.len() * 4];
                for (&alpha, rgba) in data.into_iter().zip(rgba_buffer.chunks_mut(4)) {
                    rgba[0] = 255;
                    rgba[1] = 255;
                    rgba[2] = 255;
                    rgba[3] = alpha;
                }

                let image_region = RgbaImage::from_raw(w, h, rgba_buffer).unwrap();
                font_atlas_image.copy_from(&image_region, x, y + atlas_height / 2);
                modified = true;
            })
            .unwrap();

        // TODO: Use a separate font texture, in a texture array
        if modified {
            self.draw_system.upload_atlas_texture();
        }
    }

    fn convert_glyphs_to_sprites(&mut self) {
        let [res_x, res_y] = self.draw_system.window_data.resolution;

        for (depth, glyph) in self.glyphs.drain(..) {
            use rusttype::Point;

            let Glyph {
                glyph,
                tint,
                font_id,
            } = glyph;
            let scale = glyph.scale();
            let ascent = self.draw_system.cpu_cache.fonts[font_id]
                .v_metrics(scale)
                .ascent;
            let texcoords = self
                .draw_system
                .cpu_cache
                .glyph_cache
                .rect_for(font_id, &glyph)
                .unwrap();

            if let Some((uv_rect, px_rect)) = texcoords {
                let glyph_sprite = {
                    let Point { x, y } = px_rect.min;
                    let w = px_rect.width() as f32;
                    let h = px_rect.height() as f32;

                    let Point { x: u, y: v } = uv_rect.min;
                    let uw = uv_rect.width();
                    let vh = uv_rect.height();

                    SpritePushConstants {
                        transform: make_transform(
                            [x as f32, y as f32 + ascent],
                            [w, h],
                            [res_x as f32, res_y as f32],
                        ),
                        tint,
                        uv_origin: [u, 0.5 + v / 2.0],
                        uv_scale: [uw, vh / 2.0],
                    }
                };

                self.sprites.push((depth, glyph_sprite));
            }
        }
    }

    pub fn finish(mut self) {
        let time_at_frame_start = Instant::now();
        self.draw_system.render_stats.frame_time = self.draw_system.time_at_last_frame.elapsed();
        self.draw_system.time_at_last_frame = time_at_frame_start;

        self.update_font_atlas();
        self.convert_glyphs_to_sprites();

        let gpu = self.draw_system.gpu.as_mut().unwrap();

        // Upload all sprite data to sprites_buffer
        {
            let sprite_data = &mut self.draw_system.cpu_cache.sprite_vertex_data;
            sprite_data.clear();

            self.sprites.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (depth, sprite) in &self.sprites {
                const BASE_VERTICES: &[([f32; 2], [f32; 2])] = &[
                    ([0.0, 0.0], [0.0, 0.0]),
                    ([0.0, 1.0], [0.0, 1.0]),
                    ([1.0, 0.0], [1.0, 0.0]),
                    ([0.0, 1.0], [0.0, 1.0]),
                    ([1.0, 1.0], [1.0, 1.0]),
                    ([1.0, 0.0], [1.0, 0.0]),
                ];

                // TODO: Don't even have an intermediary matrix
                for &([x, y], [u, v]) in BASE_VERTICES {
                    let [ou, ov] = sprite.uv_origin;
                    let [su, sv] = sprite.uv_scale;

                    let sx = sprite.transform[0][0];
                    let sy = sprite.transform[1][1];
                    let ox = sprite.transform[3][0];
                    let oy = sprite.transform[3][1];
                    let z = *depth / MAX_DEPTH;

                    sprite_data.push(Vertex {
                        offset: [ox + sx * x, oy + sy * y, z],
                        tint: sprite.tint,
                        uv: [ou + su * u, ov + sv * v],
                    });
                }
            }

            unsafe {
                utils::fill_buffer(&gpu.device, &mut gpu.vertex_memories[1], sprite_data);
            }
        }

        unsafe {
            let (swapchain, _extent, _frame_images, _frame_views, _framebuffers) =
                gpu.swapchain.as_mut().unwrap();

            let scene_command_buffer = {
                let mut command_buffer = gpu.command_pool.acquire_command_buffer::<OneShot>();
                command_buffer.begin();

                let [vwidth, vheight] = self.draw_system.window_data.resolution;
                let viewport = Viewport {
                    rect: Rect {
                        x: 0,
                        y: 0,
                        w: vwidth as i16,
                        h: vheight as i16,
                    },
                    depth: 0.0..1.0,
                };

                command_buffer.set_viewports(0, &[viewport.clone()]);
                command_buffer.set_scissors(0, &[viewport.rect]);

                command_buffer.bind_graphics_pipeline(&gpu.pipeline);
                command_buffer.bind_vertex_buffers(0, vec![(&gpu.vertex_buffers[1], 0)]);
                command_buffer.bind_graphics_descriptor_sets(
                    &gpu.pipeline_layout,
                    0,
                    vec![&gpu.sprites_desc_set],
                    &[],
                );

                {
                    let mut encoder = command_buffer.begin_render_pass_inline(
                        &gpu.render_pass,
                        &gpu.rtt_framebuffer,
                        viewport.rect,
                        &[ClearValue::Color(ClearColor::Float(
                            self.canvas_clear_color,
                        ))],
                    );

                    let num_verts = self.sprites.len() as u32 * 6;
                    encoder.draw(0..num_verts, 0..1);
                }

                command_buffer.finish();
                command_buffer
            };

            let scene_submission = Submission {
                command_buffers: Some(&scene_command_buffer),
                wait_semaphores: vec![(&gpu.frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)],
                signal_semaphores: vec![&gpu.scene_semaphore],
            };

            let blit_submission = Submission {
                command_buffers: Some(self.blit_command_buffer.as_ref().unwrap()),
                wait_semaphores: vec![(&gpu.scene_semaphore, PipelineStage::BOTTOM_OF_PIPE)],
                signal_semaphores: vec![&gpu.present_semaphore],
            };

            gpu.queue_group.queues[0].submit(scene_submission, None);
            gpu.queue_group.queues[0].submit(blit_submission, None);

            let result = swapchain.present(
                &mut gpu.queue_group.queues[0],
                self.frame_index,
                vec![&gpu.present_semaphore],
            );

            match result {
                Ok(_) => self.draw_system.render_stats.present_failed = false,
                Err(_) => {
                    gpu.swapchain_invalidated = true;
                    self.draw_system
                        .log("Swapchain invalidated: present failed");
                }
            }

            // TODO: Can we reuse these buffers?
            let gpu = self.draw_system.gpu.as_mut().unwrap();
            gpu.command_pool.free(vec![
                scene_command_buffer,
                self.blit_command_buffer.take().unwrap(),
            ]);

            self.finished = true;
        }

        if let Some(mut path) = self.draw_system.recording.clone() {
            path.push(format!("frame_{:06}.png", self.draw_system.recorded_frames));
            self.draw_system.capture_to_file(Capture::Canvas, &path);
            self.draw_system.recorded_frames += 1;
        }

        self.draw_system.render_stats.frame_render_time = time_at_frame_start.elapsed();
    }
}

impl<'a> Drop for Renderer<'a> {
    fn drop(&mut self) {
        if !self.finished {
            panic!("Renderer dropped without calling `finish()`");
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpriteArgs {
    pub pos: [f32; 2],
    pub depth: f32,
    pub size: Option<[f32; 2]>,
    pub tint: [f32; 4],
}

impl Default for SpriteArgs {
    fn default() -> Self {
        SpriteArgs {
            pos: [0.0, 0.0],
            size: None,
            depth: 0.0,
            tint: WHITE,
        }
    }
}

impl From<[f32; 2]> for SpriteArgs {
    fn from(pos: [f32; 2]) -> Self {
        SpriteArgs {
            pos,
            ..Default::default()
        }
    }
}

impl From<([f32; 2], f32)> for SpriteArgs {
    fn from((pos, depth): ([f32; 2], f32)) -> Self {
        SpriteArgs {
            pos,
            depth,
            ..Default::default()
        }
    }
}

impl From<([f32; 2], f32, [f32; 2])> for SpriteArgs {
    fn from((pos, depth, size): ([f32; 2], f32, [f32; 2])) -> Self {
        SpriteArgs {
            pos,
            depth,
            size: Some(size),
            ..Default::default()
        }
    }
}

impl From<([f32; 2], f32, [f32; 4])> for SpriteArgs {
    fn from((pos, depth, tint): ([f32; 2], f32, [f32; 4])) -> Self {
        SpriteArgs {
            pos,
            depth,
            tint,
            ..Default::default()
        }
    }
}

impl From<([f32; 2], [f32; 2])> for SpriteArgs {
    fn from((pos, size): ([f32; 2], [f32; 2])) -> Self {
        SpriteArgs {
            pos,
            size: Some(size),
            ..Default::default()
        }
    }
}

impl From<([f32; 2], [f32; 4])> for SpriteArgs {
    fn from((pos, tint): ([f32; 2], [f32; 4])) -> Self {
        SpriteArgs {
            pos,
            tint,
            ..Default::default()
        }
    }
}
impl From<([f32; 2], f32, [f32; 2], [f32; 4])> for SpriteArgs {
    fn from((pos, depth, size, tint): ([f32; 2], f32, [f32; 2], [f32; 4])) -> Self {
        SpriteArgs {
            pos,
            depth,
            size: Some(size),
            tint,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextArgs {
    pub pos: [f32; 2],
    pub depth: f32,
    pub tint: [f32; 4],
}

impl Default for TextArgs {
    fn default() -> Self {
        TextArgs {
            pos: [0.0, 0.0],
            depth: 0.0,
            tint: WHITE,
        }
    }
}

impl From<[f32; 2]> for TextArgs {
    fn from(pos: [f32; 2]) -> Self {
        TextArgs {
            pos,
            ..Default::default()
        }
    }
}

impl From<([f32; 2], f32)> for TextArgs {
    fn from((pos, depth): ([f32; 2], f32)) -> Self {
        TextArgs {
            pos,
            depth,
            ..Default::default()
        }
    }
}

impl From<([f32; 2], f32, [f32; 4])> for TextArgs {
    fn from((pos, depth, tint): ([f32; 2], f32, [f32; 4])) -> Self {
        TextArgs {
            pos,
            depth,
            tint,
            ..Default::default()
        }
    }
}
