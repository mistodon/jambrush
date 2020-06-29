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

#[cfg(all(target_os = "macos", not(feature = "opengl")))]
const BACKEND: &str = "metal";

#[cfg(all(windows, not(feature = "opengl"), not(feature = "dx11")))]
const BACKEND: &str = "dx12";

#[cfg(all(unix, not(target_os = "macos"), not(feature = "opengl")))]
const BACKEND: &str = "vulkan";

#[cfg(feature = "opengl")]
const BACKEND: &str = "gl";

#[cfg(feature = "dx11")]
const BACKEND: &str = "dx11";

mod gfxutils;

use std::{
    mem::ManuallyDrop,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use image::{DynamicImage, GenericImage, RgbaImage};
use rusttype::{gpu_cache::Cache as RTCache, Font as RTFont, GlyphId, PositionedGlyph};
use texture_packer::TexturePacker;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event_loop::EventLoopWindowTarget,
    window::WindowBuilder,
};

use crate::gfxutils::*;

const WHITE: [f32; 4] = [1., 1., 1., 1.];
const MAX_SPRITE_COUNT: usize = 100000; // TODO: Make dynamic
const MAX_DEPTH: f32 = 65535.; // TODO: Let this be customizable

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
    pub depth_scale_add: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct SpriteData {
    pub transform: [[f32; 4]; 4],
    pub tint: [f32; 4],
    pub uv_origin: [f32; 2],
    pub uv_scale: [f32; 2],
    pub depth_scale_add: [f32; 2],
    pub depth_mapped: bool,
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
    grid: [u32; 2],
    cell: [u32; 2],
    size: [f32; 2],
    transparent: bool,
}

impl Sprite {
    pub fn size(&self) -> [f32; 2] {
        self.size
    }
}

#[derive(Debug)]
pub struct SpriteSheet {
    id: usize,
    // TODO: Use an array for dimensions
    width: u32,
    height: u32,
    size: [f32; 2],
    transparent: bool,
}

impl SpriteSheet {
    pub fn from_shape(sprite: &Sprite, size_in_sprites: [usize; 2]) -> Self {
        let [width, height] = size_in_sprites;
        SpriteSheet {
            id: sprite.id,
            width: width as u32,
            height: height as u32,
            size: sprite.size,
            transparent: sprite.transparent,
        }
    }

    pub fn from_size(sprite: &Sprite, sprite_size: [u32; 2]) -> Self {
        let [sx, sy] = sprite_size;
        let [w, h] = sprite.size;
        let width = w as u32 / sx;
        let height = h as u32 / sy;
        SpriteSheet {
            id: sprite.id,
            width,
            height,
            size: sprite.size,
            transparent: sprite.transparent,
        }
    }

    pub fn sprite(&self, coord: [usize; 2]) -> Sprite {
        let [x, y] = coord;
        debug_assert!(
            x < self.width as usize && y < self.height as usize,
            "Sprite {:?} is out of the {:?} range allowed by this SpriteSheet.",
            [x, y],
            [self.width, self.height]
        );
        Sprite {
            id: self.id,
            grid: [self.width as u32, self.height as u32],
            cell: [coord[0] as u32, coord[1] as u32],
            size: [
                self.size[0] / self.width as f32,
                self.size[1] / self.height as f32,
            ],
            transparent: self.transparent,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Font {
    id: usize,
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
    DepthTextureAtlas,
    Canvas,
    DepthBuffer,
    // TODO: Window,
}

fn make_transform(pos: [f32; 2], scale: [f32; 2], resolution: [f32; 2]) -> [[f32; 4]; 4] {
    let [sx, sy] = resolution;
    let [w, h] = scale;
    let [x, y] = pos;
    let dx = -1. + 2. * (x / sx as f32);
    let dy = -1. + 2. * (y / sy as f32);

    [
        [(w / sx as f32) * 2., 0., 0., 0.],
        [0., (h / sy as f32) * 2., 0., 0.],
        [0., 0., 1., 0.],
        [dx, dy, 0., 1.],
    ]
}

struct InitInfo {
    atlas_size: u32,
}

struct CpuSideCache {
    sprite_textures: Vec<(RgbaImage, Option<DepthImage>)>,
    sprite_regions: Vec<([f32; 2], [f32; 2], [f32; 2])>,
    atlas_image: RgbaImage,
    depth_atlas_image: DepthImage,
    sprite_vertex_data: Vec<Vertex>,
    fonts: Vec<RTFont<'static>>,
    glyph_cache: RTCache<'static>,
}

struct GpuResources {
    #[cfg(not(feature = "opengl"))]
    _instance: backend::Instance,

    surface: backend::Surface,
    surface_color_format: Format,
    adapter: Adapter<backend::Backend>,
    device: TDevice,
    queue_group: TQueueGroup,
    command_pool: TCommandPool,
    rtt_render_pass: TRenderPass,
    blit_render_pass: TRenderPass,
    set_layout: TDescriptorSetLayout,
    pipeline_layout: TPipelineLayout,
    rtt_opaque_pipeline: TGraphicsPipeline,
    rtt_depthmapped_pipeline: TGraphicsPipeline,
    rtt_trans_pipeline: TGraphicsPipeline,
    blit_pipeline: TGraphicsPipeline,
    desc_pool: TDescriptorPool,
    sprites_desc_set: TDescriptorSet,
    depth_sprites_desc_set: TDescriptorSet,
    blit_desc_set: TDescriptorSet,
    texture_semaphore: TSemaphore,
    scene_semaphore: TSemaphore,
    frame_semaphore: TSemaphore,
    present_semaphore: TSemaphore,
    rtt_image: TImage,
    rtt_memory: TMemory,
    rtt_view: TImageView,
    rtt_sampler: TSampler,
    depth_image: TImage,
    depth_memory: TMemory,
    depth_view: TImageView,
    texture_fence: TFence,
    atlas_texture: TImage,
    atlas_memory: TMemory,
    atlas_view: TImageView,
    atlas_sampler: TSampler,
    depth_atlas_texture: TImage,
    depth_atlas_memory: TMemory,
    depth_atlas_view: TImageView,
    depth_atlas_sampler: TSampler,
    vertex_buffers: Vec<TBuffer>,
    vertex_memories: Vec<TMemory>,
    surface_extent: Extent2D,
    swapchain_invalidated: bool,
}

impl GpuResources {
    fn destroy(self) {
        let GpuResources {
            mut surface,
            device,
            command_pool,
            rtt_render_pass,
            blit_render_pass,
            set_layout,
            pipeline_layout,
            rtt_opaque_pipeline,
            rtt_depthmapped_pipeline,
            rtt_trans_pipeline,
            blit_pipeline,
            desc_pool,
            texture_semaphore,
            scene_semaphore,
            frame_semaphore,
            present_semaphore,
            rtt_image,
            rtt_memory,
            rtt_view,
            rtt_sampler,
            depth_image,
            depth_memory,
            depth_view,
            texture_fence,
            atlas_texture,
            atlas_memory,
            atlas_view,
            atlas_sampler,
            depth_atlas_texture,
            depth_atlas_memory,
            depth_atlas_view,
            depth_atlas_sampler,
            vertex_buffers,
            vertex_memories,
            ..
        } = self;

        unsafe {
            device.destroy_sampler(depth_atlas_sampler);
            device.destroy_image_view(depth_atlas_view);
            device.free_memory(depth_atlas_memory);
            device.destroy_image(depth_atlas_texture);
            device.destroy_sampler(atlas_sampler);
            device.destroy_image_view(atlas_view);
            device.free_memory(atlas_memory);
            device.destroy_image(atlas_texture);
            device.destroy_fence(texture_fence);
            device.destroy_image_view(depth_view);
            device.free_memory(depth_memory);
            device.destroy_image(depth_image);
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
            device.destroy_graphics_pipeline(blit_pipeline);
            device.destroy_graphics_pipeline(rtt_trans_pipeline);
            device.destroy_graphics_pipeline(rtt_depthmapped_pipeline);
            device.destroy_graphics_pipeline(rtt_opaque_pipeline);
            device.destroy_pipeline_layout(pipeline_layout);
            device.destroy_descriptor_set_layout(set_layout);
            device.destroy_command_pool(command_pool);
            device.destroy_render_pass(blit_render_pass);
            device.destroy_render_pass(rtt_render_pass);
            surface.unconfigure_swapchain(&device);
        }
    }
}

struct WindowData {
    window: winit::window::Window,

    resolution: [u32; 2],
    dpi_factor: f64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RenderStats {
    pub frame_count: usize,
    pub frame_time: Duration,
    pub frame_render_time: Duration,
    pub present_failed: bool,
}

#[derive(Debug, Default, Clone)]
pub struct Cursor {
    pub pos: [f32; 2],
    previous: Option<(Font, f32, GlyphId)>,
    pub start_x: f32,
}

impl From<[f32; 2]> for Cursor {
    fn from(pos: [f32; 2]) -> Self {
        Cursor {
            pos,
            previous: None,
            start_x: pos[0],
        }
    }
}

pub trait IntoLines<'a> {
    type Iter: Iterator<Item = &'a str>;

    fn into_lines(&self) -> Self::Iter;
}

impl<'a> IntoLines<'a> for &'a str {
    type Iter = std::str::Lines<'a>;

    fn into_lines(&self) -> Self::Iter {
        self.lines()
    }
}

impl<'a> IntoLines<'a> for &'a String {
    type Iter = std::str::Lines<'a>;

    fn into_lines(&self) -> Self::Iter {
        self.lines()
    }
}

impl<'a> IntoLines<'a> for &'a [&'a str] {
    type Iter = std::iter::Copied<std::slice::Iter<'a, &'a str>>;

    fn into_lines(&self) -> Self::Iter {
        self.iter().copied()
    }
}

impl<'a> IntoLines<'a> for &'a Vec<&'a str> {
    type Iter = std::iter::Copied<std::slice::Iter<'a, &'a str>>;

    fn into_lines(&self) -> Self::Iter {
        self.iter().copied()
    }
}

// TODO: Lots. Think about resolution/rebuilding RTT texture
pub struct JamBrushSystem {
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
        event_loop: &EventLoopWindowTarget<()>,
        canvas_size: Option<[u32; 2]>,
        logging: bool,
        max_texture_atlas_size: Option<u32>,
    ) -> (InitInfo, WindowData, GpuResources) {
        if logging {
            println!("Initializing window and render context");
        }

        #[cfg(not(feature = "opengl"))]
        let (window, inner_size, dpi_factor, _instance, surface, adapter) = {
            let window = window_builder.build(event_loop).unwrap();
            let dpi_factor = window.scale_factor();
            let inner_size: PhysicalSize<u32> = window.inner_size();
            let _instance =
                backend::Instance::create("JamBrush", 1).expect("Backend not supported");
            let surface = unsafe {
                _instance
                    .create_surface(&window)
                    .expect("Failed to create surface for window")
            };
            let adapter = _instance.enumerate_adapters().remove(0);
            (window, inner_size, dpi_factor, _instance, surface, adapter)
        };

        #[cfg(feature = "opengl")]
        let (window, inner_size, dpi_factor, adapter, surface) = {
            let (window, surface) = {
                let builder = backend::config_context(
                    backend::glutin::ContextBuilder::new(),
                    gfx_hal::format::Format::Rgba8Srgb,
                    None,
                )
                .with_vsync(true);

                let windowed_context = builder.build_windowed(window_builder, &event_loop).unwrap();

                let (context, window) = unsafe {
                    windowed_context
                        .make_current()
                        .expect("Unable to make context current")
                        .split()
                };

                let surface = backend::Surface::from_context(context);

                (window, surface)
            };

            let dpi_factor = window.scale_factor();
            let inner_size: PhysicalSize<u32> = window.inner_size();
            let adapter = surface.enumerate_adapters().remove(0);
            (window, inner_size, dpi_factor, adapter, surface)
        };

        let resolution = canvas_size.unwrap_or_else(|| {
            let logical_size: LogicalSize<u32> = inner_size.to_logical(dpi_factor);
            let (window_w, window_h) = logical_size.into();
            [window_w, window_h]
        });

        let (device, queue_group) = {
            use gfx_hal::queue::QueueFamily;

            let queue_family = adapter
                .queue_families
                .iter()
                .find(|family| {
                    surface.supports_queue_family(family) && family.queue_type().supports_graphics()
                })
                .expect("No compatible queue family found");

            let mut gpu = unsafe {
                adapter
                    .physical_device
                    .open(&[(queue_family, &[1.])], gfx_hal::Features::empty())
                    .expect("Failed to open device")
            };

            (gpu.device, gpu.queue_groups.pop().unwrap())
        };

        let command_pool = unsafe {
            device
                .create_command_pool(queue_group.family, CommandPoolCreateFlags::empty())
                .expect("TODO")
        };

        let surface_color_format = {
            let supported_formats = surface
                .supported_formats(&adapter.physical_device)
                .unwrap_or(vec![]);

            let default_format = *supported_formats.get(0).unwrap_or(&Format::Rgba8Srgb);

            supported_formats
                .into_iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .unwrap_or(default_format)
        };
        let surface_depth_format = Format::D32Sfloat;

        let rtt_render_pass = unsafe {
            let color_attachment = Attachment {
                format: Some(Format::Rgba8Srgb),
                samples: 1,
                ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::Present,
            };

            let depth_attachment = Attachment {
                format: Some(surface_depth_format),
                samples: 1,
                ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::DontCare),
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::DepthStencilAttachmentOptimal,
            };

            let subpass = SubpassDesc {
                colors: &[(0, Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            device
                .create_render_pass(&[color_attachment, depth_attachment], &[subpass], &[])
                .unwrap()
        };

        let blit_render_pass = unsafe {
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

            device
                .create_render_pass(&[color_attachment], &[subpass], &[])
                .unwrap()
        };

        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &[
                        DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: DescriptorType::Image {
                                ty: ImageDescriptorType::Sampled {
                                    with_sampler: false,
                                },
                            },
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
                .create_pipeline_layout(vec![&set_layout, &set_layout], &[])
                .unwrap()
        };

        let vertex_shader_module = unsafe {
            let spirv = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/compiled/sprite.vert.spv"
            ));
            let spirv = gfx_hal::pso::read_spirv(std::io::Cursor::new(spirv.as_ref()))
                .expect("Invalid SPIR-V");
            device.create_shader_module(&spirv).unwrap()
        };

        let fragment_shader_module = unsafe {
            let spirv = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/compiled/sprite.frag.spv"
            ));
            let spirv = gfx_hal::pso::read_spirv(std::io::Cursor::new(spirv.as_ref()))
                .expect("Invalid SPIR-V");
            device.create_shader_module(&spirv).unwrap()
        };

        let depth_vertex_shader_module = unsafe {
            let spirv = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/compiled/depth_sprite.vert.spv"
            ));
            let spirv = gfx_hal::pso::read_spirv(std::io::Cursor::new(spirv.as_ref()))
                .expect("Invalid SPIR-V");
            device.create_shader_module(&spirv).unwrap()
        };

        let depth_fragment_shader_module = unsafe {
            let spirv = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/compiled/depth_sprite.frag.spv"
            ));
            let spirv = gfx_hal::pso::read_spirv(std::io::Cursor::new(spirv.as_ref()))
                .expect("Invalid SPIR-V");
            device.create_shader_module(&spirv).unwrap()
        };

        let blit_vertex_shader_module = unsafe {
            let spirv = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/compiled/blit.vert.spv"
            ));
            let spirv = gfx_hal::pso::read_spirv(std::io::Cursor::new(spirv.as_ref()))
                .expect("Invalid SPIR-V");
            device.create_shader_module(&spirv).unwrap()
        };

        let blit_fragment_shader_module = unsafe {
            let spirv = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/assets/compiled/blit.frag.spv"
            ));
            let spirv = gfx_hal::pso::read_spirv(std::io::Cursor::new(spirv.as_ref()))
                .expect("Invalid SPIR-V");
            device.create_shader_module(&spirv).unwrap()
        };

        let rtt_opaque_pipeline = unsafe {
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
                main_pass: &rtt_render_pass,
            };

            let mut pipeline_desc = GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                Rasterizer::FILL,
                &pipeline_layout,
                subpass,
            );

            pipeline_desc.blender.targets.push(ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::ALPHA),
            });

            pipeline_desc.depth_stencil = DepthStencilDesc {
                depth: Some(DepthTest {
                    fun: Comparison::LessEqual,
                    write: true,
                }),
                depth_bounds: false,
                stencil: None,
            };

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

        let rtt_depthmapped_pipeline = unsafe {
            let vs_entry = EntryPoint::<backend::Backend> {
                entry: "main",
                module: &depth_vertex_shader_module,
                specialization: Default::default(),
            };

            let fs_entry = EntryPoint::<backend::Backend> {
                entry: "main",
                module: &depth_fragment_shader_module,
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
                main_pass: &rtt_render_pass,
            };

            let mut pipeline_desc = GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                Rasterizer::FILL,
                &pipeline_layout,
                subpass,
            );

            pipeline_desc.blender.targets.push(ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::ALPHA),
            });

            pipeline_desc.depth_stencil = DepthStencilDesc {
                depth: Some(DepthTest {
                    fun: Comparison::LessEqual,
                    write: true,
                }),
                depth_bounds: false,
                stencil: None,
            };

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

            pipeline_desc.attributes.push(AttributeDesc {
                location: 3,
                binding: 0,
                element: Element {
                    format: Format::Rg32Sfloat,
                    offset: 36,
                },
            });

            device
                .create_graphics_pipeline(&pipeline_desc, None)
                .expect("create_graphics_pipeline failed")
        };

        let rtt_trans_pipeline = unsafe {
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
                main_pass: &rtt_render_pass,
            };

            let mut pipeline_desc = GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                Rasterizer::FILL,
                &pipeline_layout,
                subpass,
            );

            pipeline_desc.blender.targets.push(ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::ALPHA),
            });

            pipeline_desc.depth_stencil = DepthStencilDesc {
                depth: Some(DepthTest {
                    fun: Comparison::LessEqual,
                    write: false,
                }),
                depth_bounds: false,
                stencil: None,
            };

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

        let blit_pipeline = unsafe {
            let vs_entry = EntryPoint::<backend::Backend> {
                entry: "main",
                module: &blit_vertex_shader_module,
                specialization: Default::default(),
            };

            let fs_entry = EntryPoint::<backend::Backend> {
                entry: "main",
                module: &blit_fragment_shader_module,
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
                main_pass: &blit_render_pass,
            };

            let mut pipeline_desc = GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                Rasterizer::FILL,
                &pipeline_layout,
                subpass,
            );

            pipeline_desc.blender.targets.push(ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::ALPHA),
            });

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
                    3,
                    &[
                        DescriptorRangeDesc {
                            ty: DescriptorType::Image {
                                ty: ImageDescriptorType::Sampled {
                                    with_sampler: false,
                                },
                            },
                            count: 3,
                        },
                        DescriptorRangeDesc {
                            ty: DescriptorType::Sampler,
                            count: 3,
                        },
                    ],
                    DescriptorPoolCreateFlags::empty(),
                )
                .unwrap()
        };

        let sprites_desc_set = unsafe { desc_pool.allocate_set(&set_layout).unwrap() };
        let depth_sprites_desc_set = unsafe { desc_pool.allocate_set(&set_layout).unwrap() };
        let blit_desc_set = unsafe { desc_pool.allocate_set(&set_layout).unwrap() };

        // TODO: Pipeline barrier instead of semaphores?
        let texture_semaphore = device.create_semaphore().unwrap();
        let scene_semaphore = device.create_semaphore().unwrap();
        let frame_semaphore = device.create_semaphore().unwrap();
        let present_semaphore = device.create_semaphore().unwrap();

        let memory_types = adapter.physical_device.memory_properties().memory_types;

        const QY: f32 = {
            #[cfg(feature = "opengl")]
            {
                0.
            }
            #[cfg(not(feature = "opengl"))]
            {
                1.
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
                        offset: [-1., -1., 0.],
                        uv: [0., 1. - QY],
                        tint: WHITE,
                        depth_scale_add: [0.; 2],
                    },
                    Vertex {
                        offset: [-1., 1., 0.],
                        uv: [0., QY],
                        tint: WHITE,
                        depth_scale_add: [0.; 2],
                    },
                    Vertex {
                        offset: [1., -1., 0.],
                        uv: [1., 1. - QY],
                        tint: WHITE,
                        depth_scale_add: [0.; 2],
                    },
                    Vertex {
                        offset: [-1., 1., 0.],
                        uv: [0., QY],
                        tint: WHITE,
                        depth_scale_add: [0.; 2],
                    },
                    Vertex {
                        offset: [1., 1., 0.],
                        uv: [1., QY],
                        tint: WHITE,
                        depth_scale_add: [0.; 2],
                    },
                    Vertex {
                        offset: [1., -1., 0.],
                        uv: [1., 1. - QY],
                        tint: WHITE,
                        depth_scale_add: [0.; 2],
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

        let [width, height] = resolution;
        let image_extent = Extent {
            width,
            height,
            depth: 1,
        };

        let (rtt_image, rtt_memory, rtt_view, rtt_sampler) = unsafe {
            let (rtt_image, rtt_memory, rtt_view) = utils::create_image(
                &device,
                &memory_types,
                image_extent.width,
                image_extent.height,
                Format::Rgba8Srgb,
                img::Usage::SAMPLED,
                Aspects::COLOR,
            );

            let rtt_sampler = device
                .create_sampler(&SamplerDesc::new(Filter::Nearest, WrapMode::Clamp))
                .unwrap();

            (rtt_image, rtt_memory, rtt_view, rtt_sampler)
        };

        let (depth_image, depth_memory, depth_view) = unsafe {
            utils::create_image(
                &device,
                &memory_types,
                image_extent.width,
                image_extent.height,
                surface_depth_format,
                img::Usage::DEPTH_STENCIL_ATTACHMENT,
                Aspects::DEPTH | Aspects::STENCIL,
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
                .create_sampler(&SamplerDesc::new(Filter::Nearest, WrapMode::Clamp))
                .unwrap();

            // TODO: Maybe allow pre-loading sprites here?

            (texture_image, texture_memory, texture_view, texture_sampler)
        };

        let (depth_atlas_texture, depth_atlas_memory, depth_atlas_view, depth_atlas_sampler) = unsafe {
            let (texture_image, texture_memory, texture_view) = utils::create_image(
                &device,
                &memory_types,
                atlas_size,
                atlas_size, // TODO: Probably too big
                Format::R16Uint,
                img::Usage::TRANSFER_DST | img::Usage::SAMPLED,
                Aspects::COLOR,
            );

            let texture_sampler = device
                .create_sampler(&SamplerDesc::new(Filter::Nearest, WrapMode::Clamp))
                .unwrap();

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

            device.write_descriptor_sets(vec![
                DescriptorSetWrite {
                    set: &depth_sprites_desc_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Image(&depth_atlas_view, Layout::Undefined)),
                },
                DescriptorSetWrite {
                    set: &depth_sprites_desc_set,
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Sampler(&depth_atlas_sampler)),
                },
            ]);
        }

        (
            InitInfo { atlas_size },
            WindowData {
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
                rtt_render_pass,
                blit_render_pass,
                set_layout,
                pipeline_layout,
                rtt_opaque_pipeline,
                rtt_depthmapped_pipeline,
                rtt_trans_pipeline,
                blit_pipeline,
                desc_pool,
                sprites_desc_set,
                depth_sprites_desc_set,
                blit_desc_set,
                texture_semaphore,
                scene_semaphore,
                frame_semaphore,
                present_semaphore,
                rtt_image,
                rtt_memory,
                rtt_view,
                rtt_sampler,
                depth_image,
                depth_memory,
                depth_view,
                texture_fence,
                atlas_texture,
                atlas_memory,
                atlas_view,
                atlas_sampler,
                depth_atlas_texture,
                depth_atlas_memory,
                depth_atlas_view,
                depth_atlas_sampler,
                vertex_buffers,
                vertex_memories,
                surface_extent: Extent2D {
                    width: inner_size.width,
                    height: inner_size.height,
                },
                swapchain_invalidated: true,
            },
        )
    }

    pub fn new(
        window_builder: WindowBuilder,
        event_loop: &EventLoopWindowTarget<()>,
        config: &JamBrushConfig,
    ) -> Self {
        if config.logging {
            println!("Constructing JamBrushSystem (backend={}):", BACKEND);
        }

        let (init_info, window_data, gpu) = Self::initialize_window(
            window_builder,
            event_loop,
            config.canvas_size,
            config.logging,
            config.max_texture_atlas_size,
        );

        let mut atlas_image =
            DynamicImage::new_rgba8(init_info.atlas_size, init_info.atlas_size).to_rgba();
        let depth_atlas_image = DepthImage::new(init_info.atlas_size, init_info.atlas_size);

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
            cpu_cache: CpuSideCache {
                sprite_textures: vec![],
                sprite_regions: vec![],
                atlas_image,
                depth_atlas_image,
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

    pub fn rebuild_window(
        &mut self,
        window_builder: WindowBuilder,
        event_loop: &EventLoopWindowTarget<()>,
    ) {
        let old_gpu = self.gpu.take().unwrap();
        old_gpu.destroy();

        let (_, window_data, gpu) = Self::initialize_window(
            window_builder,
            event_loop,
            Some(self.window_data.resolution),
            self.logging,
            Some(self.cpu_cache.atlas_image.width()),
        );

        self.window_data = window_data;
        self.gpu = Some(gpu);
        self.upload_atlas_texture();
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

    pub fn window(&self) -> &winit::window::Window {
        &self.window_data.window
    }

    pub fn window_resized(&mut self, _resolution: (u32, u32)) {
        // TODO: Only invalidate if changed
        let gpu = self.gpu.as_mut().unwrap();
        gpu.swapchain_invalidated = true;

        #[cfg(feature = "opengl")]
        {
            let context = gpu.surface.context();
            context.resize(_resolution.into());
            gpu.surface_extent.width = _resolution.0;
            gpu.surface_extent.height = _resolution.1;
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

    pub fn load_sprite_rgba(
        &mut self,
        size: [u32; 2],
        rgba_bytes: &[u8],
        transparent: bool,
    ) -> Sprite {
        self.load_sprite_with_depth_internal(size, rgba_bytes, transparent, None)
    }

    fn load_sprite_with_depth_internal(
        &mut self,
        size: [u32; 2],
        rgba_bytes: &[u8],
        transparent: bool,
        depth_values: Option<&[u16]>,
    ) -> Sprite {
        let sprite_index = self.cpu_cache.sprite_textures.len();

        let sprite_img: RgbaImage = RgbaImage::from_raw(size[0], size[1], rgba_bytes.to_owned())
            .expect("Failed to create image from bytes");

        let depth_img = depth_values.map(|depth_values| {
            DepthImage::from_raw(size[0], size[1], depth_values.to_owned())
                .expect("Failed to create depth image from values")
        });

        self.cpu_cache.sprite_textures.push((sprite_img, depth_img));

        self.sprite_atlas_outdated = true;

        self.log(format!(
            "Sprite {} ({} x {}) loaded and queued for upload",
            sprite_index, size[0], size[1]
        ));

        if self.debugging {
            let has_transparent_pixels = rgba_bytes
                .iter()
                .skip(3)
                .step_by(4)
                .any(|&value| value != 255);
            if has_transparent_pixels != transparent {
                match transparent {
                    true => eprintln!(
                        "WARNING: Sprite was labelled `transparent` but has no transparent pixels"
                    ),
                    false => eprintln!(
                        "WARNING: Sprite was not labelled `transparent` but has transparent pixels"
                    ),
                }
            }
        }

        Sprite {
            id: sprite_index,
            grid: [1, 1],
            cell: [0, 0],
            size: [size[0] as f32, size[1] as f32],
            transparent,
        }
    }

    pub fn load_sprite(&mut self, image_bytes: &[u8]) -> Sprite {
        let image = image::load_from_memory(&image_bytes).unwrap();
        let transparent = image.color().has_alpha();
        let image = image.to_rgba();
        let (w, h) = image.dimensions();

        self.load_sprite_rgba([w, h], &image, transparent)
    }

    pub fn load_sprite_file<P: AsRef<Path>>(&mut self, path: P) -> Sprite {
        let image_bytes = std::fs::read(path.as_ref()).unwrap();

        self.load_sprite(&image_bytes)
    }

    pub fn load_sprite_rgba_with_depth(
        &mut self,
        size: [u32; 2],
        rgba_bytes: &[u8],
        transparent: bool,
        depth_values: &[u16],
    ) -> Sprite {
        self.load_sprite_with_depth_internal(size, rgba_bytes, transparent, Some(depth_values))
    }

    pub fn load_sprite_with_depth(
        &mut self,
        image_bytes: &[u8],
        depth_image_bytes: &[u8],
    ) -> Sprite {
        let image = image::load_from_memory(&image_bytes).unwrap();
        let transparent = image.color().has_alpha();
        let image = image.to_rgba();
        let (w, h) = image.dimensions();

        let depth_image = image::load_from_memory(&depth_image_bytes).unwrap();
        let depth_image = depth_image.as_luma16().unwrap();
        assert!(depth_image.dimensions() == (w, h));

        self.load_sprite_rgba_with_depth([w, h], &image, transparent, depth_image)
    }

    pub fn load_sprite_file_with_depth<P: AsRef<Path>, Q: AsRef<Path>>(
        &mut self,
        path: P,
        depth_path: Q,
    ) -> Sprite {
        let image_bytes = std::fs::read(path.as_ref()).unwrap();
        let depth_image_bytes = std::fs::read(depth_path.as_ref()).unwrap();

        self.load_sprite_with_depth(&image_bytes, &depth_image_bytes)
    }

    // TODO: Allow reloading depth sprite too
    pub fn reload_sprite_rgba(&mut self, sprite: &Sprite, size: [u32; 2], rgba_bytes: &[u8]) {
        // TODO: Assert same size
        let [w, h] = sprite.size;
        let [w, h] = [w as u32, h as u32];

        let sprite_img: RgbaImage = RgbaImage::from_raw(size[0], size[1], rgba_bytes.to_owned())
            .expect("Failed to reload image from bytes");
        self.cpu_cache.sprite_textures[sprite.id].0 = sprite_img;

        self.sprite_atlas_outdated = true;

        self.log(format!(
            "Sprite {} ({} x {}) reloaded and queued for upload",
            sprite.id, w, h
        ));
    }

    pub fn reload_sprite(&mut self, sprite: &Sprite, image_bytes: &[u8]) {
        let image = image::load_from_memory(&image_bytes).unwrap().to_rgba();
        let (w, h) = image.dimensions();

        self.reload_sprite_rgba(sprite, [w, h], &image);
    }

    pub fn reload_sprite_file<P: AsRef<Path>>(&mut self, sprite: &Sprite, path: P) {
        let image_bytes = std::fs::read(path.as_ref()).unwrap();

        self.reload_sprite(sprite, &image_bytes);
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
            // TODO: We can store this now! Don't recreate it every time!
            let mut atlas_packer = TexturePacker::new_skyline(atlas_config);

            for (index, (texture, _)) in self.cpu_cache.sprite_textures.iter().enumerate() {
                // TODO: ugh, string keys?
                atlas_packer
                    .pack_ref(index.to_string(), texture)
                    .expect("pack_ref failed");
            }

            self.cpu_cache.sprite_regions.clear();
            for (i, (texture, depth_texture)) in self.cpu_cache.sprite_textures.iter().enumerate() {
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
                    .copy_from(texture, frame.x, frame.y)
                    .expect("Copy to atlas image failed");

                if let Some(depth_texture) = depth_texture {
                    self.cpu_cache
                        .depth_atlas_image
                        .copy_from(depth_texture, frame.x, frame.y)
                        .expect("Copy to depth atlas image failed");
                }
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
                Some(&self.cpu_cache.atlas_image),
                None,
                &gpu.atlas_texture,
            );

            utils::upload_image_data(
                &gpu.device,
                &gpu.adapter.physical_device,
                &mut gpu.command_pool,
                &mut gpu.queue_group.queues[0],
                &gpu.texture_fence,
                None,
                Some(&self.cpu_cache.depth_atlas_image),
                &gpu.depth_atlas_texture,
            );
        }
    }

    fn update_swapchain(&mut self) {
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.swapchain_invalidated {
            self.create_swapchain();
        }
    }

    fn create_swapchain(&mut self) {
        self.log("Creating swapchain");
        let gpu = self.gpu.as_mut().unwrap();
        gpu.swapchain_invalidated = false;

        let caps = gpu.surface.capabilities(&gpu.adapter.physical_device);

        let mut swapchain_config =
            SwapchainConfig::from_caps(&caps, gpu.surface_color_format, gpu.surface_extent);

        // This seems to fix some fullscreen slowdown on macOS.
        if caps.image_count.contains(&3) {
            swapchain_config.image_count = 3;
        }

        gpu.surface_extent = swapchain_config.extent;

        unsafe {
            gpu.surface
                .configure_swapchain(&gpu.device, swapchain_config)
                .expect("Failed to configure swapchain");
        }
    }

    pub fn capture_image(&mut self, capture_type: Capture) -> ([u32; 2], Vec<u8>) {
        let gpu = self.gpu.as_mut().unwrap();
        unsafe {
            gpu.device.wait_idle().unwrap();
            gpu.device.reset_fence(&gpu.texture_fence).unwrap();

            let image = match capture_type {
                Capture::Canvas => &gpu.rtt_image, // TODO: Clear up image/texture confusion
                Capture::TextureAtlas => &gpu.atlas_texture,
                Capture::DepthTextureAtlas => &gpu.depth_atlas_texture,
                Capture::DepthBuffer => &gpu.depth_image,
            };

            let bytes_per_pixel = match capture_type {
                Capture::DepthTextureAtlas => 2,
                _ => 4,
            };

            let (memory_size, width, height) = {
                let footprint = gpu.device.get_image_subresource_footprint(
                    image,
                    Subresource {
                        aspects: Aspects::COLOR,
                        level: 0,
                        layer: 0,
                    },
                );
                (
                    footprint.slice.end - footprint.slice.start,
                    footprint.row_pitch as u32 / bytes_per_pixel,
                    memory_size as u32 / footprint.row_pitch as u32,
                )
            };

            let memory_types = gpu.adapter.physical_device.memory_properties().memory_types;

            // TODO: Are these deleted?
            let (screenshot_buffer, screenshot_memory) = utils::empty_buffer::<u8>(
                &gpu.device,
                &memory_types,
                Properties::CPU_VISIBLE,
                buffer::Usage::TRANSFER_DST,
                memory_size as usize,
            );

            let submit = {
                let mut cmd_buffer = gpu.command_pool.allocate_one(Level::Primary);
                cmd_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

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

            gpu.queue_group.queues[0]
                .submit_without_semaphores(&[submit], Some(&gpu.texture_fence));

            gpu.device.wait_for_fence(&gpu.texture_fence, !0).unwrap();

            let image_bytes = {
                let mapped_memory = gpu
                    .device
                    .map_memory(&screenshot_memory, Segment::ALL)
                    .expect("TODO");

                let image_bytes = std::slice::from_raw_parts(mapped_memory, memory_size as usize);
                let image_bytes = image_bytes.to_vec();
                gpu.device.unmap_memory(&screenshot_memory);

                image_bytes
            };

            ([width, height], image_bytes)
        }
    }

    pub fn capture_to_file<P: AsRef<Path>>(&mut self, capture_type: Capture, path: P) {
        use image::ColorType;

        let (size, image_bytes) = self.capture_image(capture_type);
        let color_type = match capture_type {
            Capture::DepthTextureAtlas | Capture::DepthBuffer => ColorType::L16,
            _ => ColorType::Rgba8,
        };
        let image_bytes = match capture_type {
            // Crunch float depth down to u16s
            Capture::DepthBuffer => {
                let shorts = image_bytes.chunks(4).map(|bytes| {
                    let depth = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    (depth * 65535.0) as u16
                }).collect::<Vec<_>>();
                let byte_slice = unsafe { std::slice::from_raw_parts(shorts.as_ptr() as *const u8, shorts.len() * 2) };
                byte_slice.to_vec()
            }
            _ => image_bytes,
        };
        image::save_buffer(path, &image_bytes, size[0], size[1], color_type).unwrap();
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

    pub fn window_scale(&self) -> Option<u32> {
        self.gpu
            .as_ref()
            .map(|gpu| {
                let [vwidth, vheight] = self.window_data.resolution;
                let scale = self.window_data.dpi_factor;

                let base_width = (f64::from(vwidth) * scale) as u32;
                let base_height = (f64::from(vheight) * scale) as u32;

                let integer_scale = std::cmp::min(
                    gpu.surface_extent.width / base_width,
                    gpu.surface_extent.height / base_height,
                );

                integer_scale
            })
            .filter(|&n| n > 0)
    }

    pub fn fullscreen(&self) -> bool {
        self.window_data.window.fullscreen().is_some()
    }
}

impl Drop for JamBrushSystem {
    fn drop(&mut self) {
        let gpu = self.gpu.take().unwrap();
        gpu.destroy();
    }
}

pub struct Renderer<'a> {
    draw_system: &'a mut JamBrushSystem,
    canvas_clear_color: [f32; 4],
    surface_image: ManuallyDrop<TSurfaceImage>,
    blit_command_buffer: Option<TCommandBuffer>,
    sprites: Vec<((bool, f32), SpriteData)>,
    glyphs: Vec<(f32, Glyph)>,
    finished: bool,
    camera: [f32; 2],
    framebuffer: ManuallyDrop<TFramebuffer>,
    rtt_framebuffer: ManuallyDrop<TFramebuffer>,
}

impl<'a> Renderer<'a> {
    fn new(
        draw_system: &'a mut JamBrushSystem,
        canvas_clear_color: [f32; 4],
        border_clear_color: [f32; 4],
    ) -> Self {
        unsafe {
            draw_system.update_swapchain();

            let gpu = draw_system.gpu.as_mut().unwrap();
            gpu.command_pool.reset(false);
        }

        let surface_image: TSurfaceImage;
        let blit_command_buffer: TCommandBuffer;
        let blit_framebuffer: TFramebuffer;
        let rtt_framebuffer: TFramebuffer;

        unsafe {
            use std::borrow::Borrow;

            let integer_scale = draw_system.window_scale().unwrap_or(0);
            let gpu = draw_system.gpu.as_mut().unwrap();

            let surface_extent = gpu.surface_extent;

            // TODO: handle failure
            let (image, _) = gpu.surface.acquire_image(ACQUIRE_TIMEOUT_NS).unwrap();
            surface_image = image;

            rtt_framebuffer = gpu
                .device
                .create_framebuffer(
                    &gpu.rtt_render_pass,
                    vec![&gpu.rtt_view, &gpu.depth_view],
                    Extent {
                        width: surface_extent.width,
                        height: surface_extent.height,
                        depth: 1,
                    },
                )
                .expect("Failed to create framebuffer");

            blit_framebuffer = gpu
                .device
                .create_framebuffer(
                    &gpu.rtt_render_pass,
                    vec![surface_image.borrow(), &gpu.depth_view],
                    Extent {
                        width: surface_extent.width,
                        height: surface_extent.height,
                        depth: 1,
                    },
                )
                .expect("Failed to create RTT framebuffer");

            const ACQUIRE_TIMEOUT_NS: u64 = 1_000_000_000;

            blit_command_buffer = {
                let mut command_buffer = gpu.command_pool.allocate_one(Level::Primary);
                command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

                let [vwidth, vheight] = draw_system.window_data.resolution;

                let base_width = (f64::from(vwidth) * draw_system.window_data.dpi_factor) as u32;
                let base_height = (f64::from(vheight) * draw_system.window_data.dpi_factor) as u32;

                let (viewport_width, viewport_height) = if integer_scale == 0 {
                    let viewport_width = std::cmp::min(
                        surface_extent.width,
                        (surface_extent.height * vwidth) / vheight,
                    );
                    let viewport_height = std::cmp::min(
                        surface_extent.height,
                        (surface_extent.width * vheight) / vwidth,
                    );
                    (viewport_width, viewport_height)
                } else {
                    let viewport_width = base_width * integer_scale;
                    let viewport_height = base_height * integer_scale;
                    (viewport_width, viewport_height)
                };

                let viewport_x = (surface_extent.width - viewport_width) / 2;
                let viewport_y = (surface_extent.height - viewport_height) / 2;

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

                command_buffer.bind_graphics_pipeline(&gpu.blit_pipeline);
                command_buffer
                    .bind_vertex_buffers(0, vec![(&gpu.vertex_buffers[0], SubRange::WHOLE)]);
                command_buffer.bind_graphics_descriptor_sets(
                    &gpu.pipeline_layout,
                    0,
                    vec![&gpu.blit_desc_set],
                    &[],
                );

                command_buffer.begin_render_pass(
                    &gpu.blit_render_pass,
                    &blit_framebuffer,
                    viewport.rect,
                    &[ClearValue {
                        color: ClearColor {
                            float32: border_clear_color,
                        },
                    }],
                    SubpassContents::Inline,
                );

                command_buffer.draw(0..6, 0..1);

                command_buffer.finish();
                command_buffer
            };
        }

        Renderer {
            draw_system,
            canvas_clear_color,
            surface_image: ManuallyDrop::new(surface_image),
            blit_command_buffer: Some(blit_command_buffer),
            sprites: vec![],
            glyphs: vec![],
            finished: false,
            camera: [0., 0.],
            framebuffer: ManuallyDrop::new(blit_framebuffer),
            rtt_framebuffer: ManuallyDrop::new(rtt_framebuffer),
        }
    }

    pub fn window_scale(&self) -> Option<u32> {
        self.draw_system.window_scale()
    }

    pub fn fullscreen(&self) -> bool {
        self.draw_system.window_data.window.fullscreen().is_some()
    }

    pub fn camera(&mut self, camera: [f32; 2]) {
        self.camera = camera;
    }

    pub fn clear_camera(&mut self) {
        self.camera([0., 0.]);
    }

    pub fn center_camera(&mut self, on: [f32; 2]) {
        let [rx, ry] = self.draw_system.window_data.resolution;
        let [x, y] = on;
        self.camera([x - rx as f32 / 2., y - ry as f32 / 2.]);
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
        let [sx, sy] = [1. / sprite.grid[0] as f32, 1. / sprite.grid[1] as f32];

        let scale = args.size.unwrap_or([px * sx, py * sy]);
        let tint = srgb_to_linear(args.tint);
        let transparent = args.depth_map.is_none() && (sprite.transparent || tint[3] < 1.);

        let uw = uv_scale[0] * sx;
        let uh = uv_scale[1] * sy;
        let u0 = uv_origin[0] + uw * sprite.cell[0] as f32;
        let v0 = uv_origin[1] + uh * sprite.cell[1] as f32;

        let [pos_x, pos_y] = args.pos;
        let [cam_x, cam_y] = self.camera;

        // TODO: Calculate & store depth offset/scale & flag for whether it's depthy
        // Say max depth is 100. I draw a sprite with depth 10.
        // But then I also ask to draw with a depth_map, offset 10, and scale 10
        // This means black pixels should be 10 + 10 + 0
        // And white pixels should be 10 + 10 + 10
        // More generally, depth = (sprite.depth + offset) + texture(..) * scale
        // We should otherwise IGNORE the vertex `z` coordinate, just set depth
        // with the above info
        let depth_scale_add = match &args.depth_map {
            Some(dargs) => [
                dargs.depth_scale.unwrap_or(MAX_DEPTH) / MAX_DEPTH,
                (args.depth + dargs.depth_offset) / MAX_DEPTH,
            ],
            None => [0.; 2],
        };
        let data = SpriteData {
            transform: make_transform(
                [pos_x - cam_x, pos_y - cam_y],
                scale,
                [res_x as f32, res_y as f32],
            ),
            tint,
            uv_origin: [u0, v0],
            uv_scale: [uw, uh],
            depth_scale_add,
            depth_mapped: args.depth_map.is_some(),
        };

        self.sprites.push(((transparent, args.depth), data));
    }

    pub fn text_with<'t, S: IntoLines<'t>>(
        &mut self,
        font: &Font,
        size: f32,
        text: S,
        args: &TextArgs,
    ) -> Cursor {
        let lines = text.into_lines();

        self.prepare_glyphs(lines, font, size, args)
    }

    pub fn text<'t, T: Into<TextArgs>, S: IntoLines<'t>>(
        &mut self,
        font: &Font,
        size: f32,
        text: S,
        args: T,
    ) -> Cursor {
        let args = args.into();
        self.text_with(font, size, text, &args)
    }

    // TODO: Write some unit tests for this
    // - Refactor the insides to a standalone method
    // - Use a trait to make it generic over Fonts + MockFonts
    // - Make it less dependent on literal Glyph(Id)s
    // - Test it with a "monospace" mock font
    pub fn reflow_text<'t, C: Into<Cursor>>(
        &mut self,
        font: &Font,
        size: f32,
        cursor: C,
        max_x: f32,
        text: &'t str,
    ) -> (Cursor, Vec<&'t str>) {
        use rusttype::Scale;

        const WORD_BREAKS: &[char] = &[' ', '\n'];

        let font_handle = font;
        let font_id = font_handle.id;
        let font = &self.draw_system.cpu_cache.fonts[font_id];
        let scale = Scale { x: size, y: size };
        let line_height = {
            let metrics = font.v_metrics(scale);
            (metrics.ascent - metrics.descent) + metrics.line_gap
        };

        let mut cursor = cursor.into();
        let mut previous = cursor.previous.map(|prev| prev.2);
        let mut result = vec![];

        let mut line_start = 0;
        let mut word_start = None;
        let mut last_word_end = None;
        let mut word_lead = 0.;
        let mut word_width = 0.;
        let mut chars = text.char_indices();

        loop {
            let next = chars.next();
            let done = next.is_none();
            let mut line_end = None;
            let mut newline = None;

            let (index, ch) = next.unwrap_or((text.len(), '\n'));
            let glyph = font.glyph(ch);
            let glyph = glyph.scaled(scale);
            let kerning = previous
                .map(|prev| font.pair_kerning(scale, prev, glyph.id()))
                .unwrap_or(0.);
            let advance = match ch {
                '\n' => 0.,
                _ => glyph.h_metrics().advance_width,
            };

            if WORD_BREAKS.contains(&ch) {
                // TODO: Handle case where entire first word is too wide
                if cursor.pos[0] < max_x && (cursor.pos[0] + word_width + word_lead) >= max_x {
                    cursor.pos[0] = cursor.start_x + word_width;
                    cursor.pos[1] += line_height;
                    line_end = last_word_end;
                } else {
                    cursor.pos[0] += word_lead + word_width;
                }

                if ch == '\n' {
                    newline = Some(index);
                }

                last_word_end = Some(index);
                word_start = None;
                word_width = 0.;
                word_lead = advance;
            } else {
                word_width += advance;

                if word_start.is_some() {
                    word_width += kerning;
                } else {
                    word_start = Some(index);
                    word_lead += kerning;
                }
            }

            previous = Some(glyph.id());

            if let Some(end) = line_end {
                result.push(&text[line_start..end]);
                line_start = end + 1; // NOTE: Assumes line break is 1 byte
            }

            if done {
                result.push(&text[line_start..]);
                break;
            } else if let Some(end) = newline {
                result.push(&text[line_start..end]);
                line_start = end + 1;
                cursor.pos[0] = cursor.start_x;
                cursor.pos[1] += line_height;
            }
        }

        let result_cursor = Cursor {
            pos: cursor.pos,
            previous: previous.map(|glyph_id| (font_handle.clone(), size, glyph_id)),
            start_x: cursor.start_x,
        };

        (result_cursor, result)
    }

    fn prepare_glyphs<'t, I: Iterator<Item = &'t str>>(
        &mut self,
        mut lines: I,
        font: &Font,
        size: f32,
        args: &TextArgs,
    ) -> Cursor {
        use rusttype::{Point, Scale};

        // TODO: scale/pos are in pixels
        //  - but should be in abstract screen-space units
        // TODO: copyin' a lotta glyphs here!
        // TODO: Use prev_glyph for pre-kerning

        let tint = srgb_to_linear(args.tint);
        let [cam_x, cam_y] = self.camera;

        let font_handle = font;
        let font_id = font_handle.id;
        let font = &self.draw_system.cpu_cache.fonts[font_id];
        let scale = Scale { x: size, y: size };

        let line_height = {
            let metrics = font.v_metrics(scale);
            (metrics.ascent - metrics.descent) + metrics.line_gap
        };

        let mut cursor = args.cursor.clone();
        let mut previous = cursor.previous.map(|prev| prev.2);

        let mut line = lines.next();

        loop {
            if let Some(line) = line {
                for glyph in font.glyphs_for(line.chars()) {
                    let glyph = glyph.scaled(scale);
                    if let Some(previous) = previous {
                        cursor.pos[0] += font.pair_kerning(scale, previous, glyph.id());
                    }

                    let advance = glyph.h_metrics().advance_width;
                    let glyph = glyph.positioned(Point {
                        x: cursor.pos[0] - cam_x,
                        y: cursor.pos[1] - cam_y,
                    });
                    previous = Some(glyph.id());
                    cursor.pos[0] += advance;

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

            line = lines.next();
            if line.is_some() {
                cursor.pos[0] = cursor.start_x;
                cursor.pos[1] += line_height;
                previous = None;
            } else {
                break;
            }
        }

        Cursor {
            pos: cursor.pos,
            previous: previous.map(|glyph_id| (font_handle.clone(), size, glyph_id)),
            start_x: cursor.start_x,
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
                font_atlas_image
                    .copy_from(&image_region, x, y + atlas_height / 2)
                    .expect("Failed to copy into font atlas");
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

                    SpriteData {
                        transform: make_transform(
                            [x as f32, y as f32 + ascent],
                            [w, h],
                            [res_x as f32, res_y as f32],
                        ),
                        tint,
                        uv_origin: [u, 0.5 + v / 2.],
                        uv_scale: [uw, vh / 2.],
                        depth_scale_add: [0.; 2],
                        depth_mapped: false,
                    }
                };

                self.sprites.push(((true, depth), glyph_sprite));
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
        let first_transparent_sprite_index: Option<usize>;

        // Upload all sprite data to sprites_buffer
        {
            let sprite_data = &mut self.draw_system.cpu_cache.sprite_vertex_data;
            sprite_data.clear();

            self.sprites.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            first_transparent_sprite_index = self.sprites.iter().position(|((trans, _), _)| *trans);

            for ((_transparent, depth), sprite) in &self.sprites {
                const BASE_VERTICES: &[([f32; 2], [f32; 2])] = &[
                    ([0., 0.], [0., 0.]),
                    ([0., 1.], [0., 1.]),
                    ([1., 0.], [1., 0.]),
                    ([0., 1.], [0., 1.]),
                    ([1., 1.], [1., 1.]),
                    ([1., 0.], [1., 0.]),
                ];

                // TODO: Don't even have an intermediary matrix
                for &([x, y], [u, v]) in BASE_VERTICES {
                    let [ou, ov] = sprite.uv_origin;
                    let [su, sv] = sprite.uv_scale;

                    let sx = sprite.transform[0][0];
                    let sy = sprite.transform[1][1];
                    let ox = sprite.transform[3][0];
                    let oy = sprite.transform[3][1];
                    let z = 1. - (*depth / MAX_DEPTH);

                    sprite_data.push(Vertex {
                        offset: [ox + sx * x, oy + sy * y, z],
                        tint: sprite.tint,
                        uv: [ou + su * u, ov + sv * v],
                        depth_scale_add: sprite.depth_scale_add,
                    });
                }
            }

            unsafe {
                utils::fill_buffer(&gpu.device, &mut gpu.vertex_memories[1], sprite_data);
            }
        }

        unsafe {
            let scene_command_buffer = {
                let mut command_buffer = gpu.command_pool.allocate_one(Level::Primary);
                command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

                let [vwidth, vheight] = self.draw_system.window_data.resolution;
                let viewport = Viewport {
                    rect: Rect {
                        x: 0,
                        y: 0,
                        w: vwidth as i16,
                        h: vheight as i16,
                    },
                    depth: 0. ..1.,
                };

                command_buffer.set_viewports(0, &[viewport.clone()]);
                command_buffer.set_scissors(0, &[viewport.rect]);

                command_buffer.begin_render_pass(
                    &gpu.rtt_render_pass,
                    &self.rtt_framebuffer,
                    viewport.rect,
                    &[
                        ClearValue {
                            color: ClearColor {
                                float32: self.canvas_clear_color,
                            },
                        },
                        ClearValue {
                            depth_stencil: ClearDepthStencil {
                                depth: 1.,
                                stencil: 0,
                            },
                        },
                    ],
                    SubpassContents::Inline,
                );

                command_buffer
                    .bind_vertex_buffers(0, vec![(&gpu.vertex_buffers[1], SubRange::WHOLE)]);
                command_buffer.bind_graphics_descriptor_sets(
                    &gpu.pipeline_layout,
                    0,
                    vec![&gpu.sprites_desc_set, &gpu.depth_sprites_desc_set],
                    &[],
                );

                // Draw opaque objects
                let trans_start_sprite_index =
                    first_transparent_sprite_index.unwrap_or(self.sprites.len());
                let trans_start_index = trans_start_sprite_index as u32 * 6;

                let mut depth_mapped = self
                    .sprites
                    .first()
                    .map(|sprite| sprite.1.depth_mapped)
                    .unwrap_or(false);
                let mut start_index = 0;

                // Draw runs of opaque or depthmapped objects
                while start_index < trans_start_sprite_index {
                    let end_of_run = self.sprites[start_index..trans_start_sprite_index]
                        .iter()
                        .position(|sprite| sprite.1.depth_mapped != depth_mapped)
                        .unwrap_or(trans_start_sprite_index);
                    let pipeline = if depth_mapped {
                        &gpu.rtt_depthmapped_pipeline
                    } else {
                        &gpu.rtt_opaque_pipeline
                    };
                    let start_vertex = start_index as u32 * 6;
                    let end_vertex = end_of_run as u32 * 6;
                    command_buffer.bind_graphics_pipeline(pipeline);
                    command_buffer.draw(start_vertex..end_vertex, 0..1);

                    start_index = end_of_run;
                    depth_mapped = !depth_mapped;
                }

                // Draw transparent objects
                if first_transparent_sprite_index.is_some() {
                    command_buffer.bind_graphics_pipeline(&gpu.rtt_trans_pipeline);

                    let end_index = self.sprites.len() as u32 * 6;
                    command_buffer.draw(trans_start_index..end_index, 0..1);
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

            let result = gpu.queue_group.queues[0].present_surface(
                &mut gpu.surface,
                ManuallyDrop::take(&mut self.surface_image),
                Some(&gpu.present_semaphore),
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
        self.draw_system.render_stats.frame_count += 1;
    }
}

impl<'a> Drop for Renderer<'a> {
    fn drop(&mut self) {
        if !self.finished {
            panic!("Renderer dropped without calling `finish()`");
        }

        let gpu = self.draw_system.gpu.as_mut().unwrap();

        unsafe {
            gpu.device
                .destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffer));
            gpu.device
                .destroy_framebuffer(ManuallyDrop::take(&mut self.rtt_framebuffer));
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct DepthMapArgs {
    pub depth_offset: f32,
    pub depth_scale: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpriteArgs {
    pub pos: [f32; 2],
    pub depth: f32,
    pub size: Option<[f32; 2]>,
    pub tint: [f32; 4],
    pub depth_map: Option<DepthMapArgs>,
}

impl Default for SpriteArgs {
    fn default() -> Self {
        SpriteArgs {
            pos: [0., 0.],
            size: None,
            depth: 0.,
            tint: WHITE,
            depth_map: None,
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

#[derive(Debug, Clone)]
pub struct TextArgs {
    pub cursor: Cursor,
    pub depth: f32,
    pub tint: [f32; 4],
}

impl Default for TextArgs {
    fn default() -> Self {
        TextArgs {
            cursor: Cursor::default(),
            depth: 0.,
            tint: WHITE,
        }
    }
}

impl<P: Into<Cursor>> From<P> for TextArgs {
    fn from(pos: P) -> Self {
        TextArgs {
            cursor: pos.into(),
            ..Default::default()
        }
    }
}

impl<P: Into<Cursor>> From<(P, f32)> for TextArgs {
    fn from((pos, depth): (P, f32)) -> Self {
        TextArgs {
            cursor: pos.into(),
            depth,
            ..Default::default()
        }
    }
}

impl<P: Into<Cursor>> From<(P, [f32; 4])> for TextArgs {
    fn from((pos, tint): (P, [f32; 4])) -> Self {
        TextArgs {
            cursor: pos.into(),
            tint,
            ..Default::default()
        }
    }
}

impl<P: Into<Cursor>> From<(P, f32, [f32; 4])> for TextArgs {
    fn from((pos, depth, tint): (P, f32, [f32; 4])) -> Self {
        TextArgs {
            cursor: pos.into(),
            depth,
            tint,
            ..Default::default()
        }
    }
}
