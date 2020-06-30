pub use gfx_hal::{
    adapter::{Adapter, MemoryType, PhysicalDevice},
    buffer::{self, SubRange},
    command::{
        BufferImageCopy, ClearColor, ClearDepthStencil, ClearValue, CommandBuffer,
        CommandBufferFlags, Level, SubpassContents,
    },
    device::Device,
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{
        self as img, Access, Extent, Filter, Layout, Offset, SamplerDesc, Subresource,
        SubresourceLayers, SubresourceRange, ViewCapabilities, ViewKind, WrapMode,
    },
    memory::{Barrier, Dependencies, Properties, Segment},
    pass::{
        Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDependency,
        SubpassDesc,
    },
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        AttributeDesc, BlendState, ColorBlendDesc, ColorMask, Comparison, DepthStencilDesc,
        DepthTest, Descriptor, DescriptorPool, DescriptorPoolCreateFlags, DescriptorRangeDesc,
        DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, Element, EntryPoint,
        GraphicsPipelineDesc, GraphicsShaderSet, ImageDescriptorType, PipelineStage, Primitive,
        Rasterizer, Rect, ShaderStageFlags, StencilTest, VertexBufferDesc, VertexInputRate,
        Viewport,
    },
    queue::{family::QueueGroup, CommandQueue, Submission},
    window::{Extent2D, PresentationSurface, Surface, SwapchainConfig},
    Instance,
};

use gfx_hal::Backend;
use image::RgbaImage;

pub type DepthImage = image::ImageBuffer<image::Luma<u16>, Vec<u16>>;

pub type TBuffer = <::backend::Backend as Backend>::Buffer;
pub type TCommandBuffer = <::backend::Backend as Backend>::CommandBuffer;
pub type TCommandPool = <::backend::Backend as Backend>::CommandPool;
pub type TCommandQueue = <::backend::Backend as Backend>::CommandQueue;
pub type TDescriptorPool = <::backend::Backend as Backend>::DescriptorPool;
pub type TDescriptorSet = <::backend::Backend as Backend>::DescriptorSet;
pub type TDescriptorSetLayout = <::backend::Backend as Backend>::DescriptorSetLayout;
pub type TDevice = <::backend::Backend as Backend>::Device;
pub type TFence = <::backend::Backend as Backend>::Fence;
pub type TFramebuffer = <::backend::Backend as Backend>::Framebuffer;
pub type TGraphicsPipeline = <::backend::Backend as Backend>::GraphicsPipeline;
pub type TImage = <::backend::Backend as Backend>::Image;
pub type TImageView = <::backend::Backend as Backend>::ImageView;
pub type TMemory = <::backend::Backend as Backend>::Memory;
pub type TPhysicalDevice = <::backend::Backend as Backend>::PhysicalDevice;
pub type TPipelineLayout = <::backend::Backend as Backend>::PipelineLayout;
pub type TQueueGroup = QueueGroup<::backend::Backend>;
pub type TRenderPass = <::backend::Backend as Backend>::RenderPass;
pub type TSampler = <::backend::Backend as Backend>::Sampler;
pub type TSemaphore = <::backend::Backend as Backend>::Semaphore;
pub type TSurface = <::backend::Backend as Backend>::Surface;
pub type TSurfaceImage = <TSurface as PresentationSurface<::backend::Backend>>::SwapchainImage;

pub mod utils {
    use super::*;

    pub unsafe fn empty_buffer<Item>(
        device: &TDevice,
        memory_types: &[MemoryType],
        properties: Properties,
        usage: buffer::Usage,
        item_count: usize,
    ) -> (TBuffer, TMemory) {
        let item_count = item_count;
        let stride = ::std::mem::size_of::<Item>() as u64;
        let buffer_len = item_count as u64 * stride;
        let mut buffer = device.create_buffer(buffer_len, usage).unwrap();
        let req = device.get_buffer_requirements(&buffer);
        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, ty)| {
                req.type_mask & (1 << id) != 0 && ty.properties.contains(properties)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(upload_type, req.size).unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut buffer)
            .unwrap();

        (buffer, buffer_memory)
    }

    pub unsafe fn fill_buffer<Item: Copy>(
        device: &TDevice,
        buffer_memory: &TMemory,
        items: &[Item],
    ) {
        let stride = ::std::mem::size_of::<Item>();
        let buffer_len = items.len() * stride;

        let mapped_memory = device
            .map_memory(&buffer_memory, Segment::ALL)
            .expect("map_memory failed");

        std::ptr::copy_nonoverlapping(items.as_ptr() as *const u8, mapped_memory, buffer_len);

        device
            .flush_mapped_memory_ranges(vec![(buffer_memory, Segment::ALL)])
            .expect("flush_mapped_memory_ranges failed");

        device.unmap_memory(&buffer_memory);
    }

    pub unsafe fn create_buffer<Item: Copy>(
        device: &TDevice,
        memory_types: &[MemoryType],
        properties: Properties,
        usage: buffer::Usage,
        items: &[Item],
    ) -> (TBuffer, TMemory) {
        let (empty_buffer, mut empty_buffer_memory) =
            empty_buffer::<Item>(device, memory_types, properties, usage, items.len());

        fill_buffer::<Item>(device, &mut empty_buffer_memory, items);

        (empty_buffer, empty_buffer_memory)
    }

    pub unsafe fn create_image(
        device: &TDevice,
        memory_types: &[MemoryType],
        width: u32,
        height: u32,
        format: Format,
        usage: img::Usage,
        aspects: Aspects,
    ) -> (TImage, TMemory, TImageView) {
        let kind = img::Kind::D2(width, height, 1, 1);

        let mut image = device
            .create_image(
                kind,
                1,
                format,
                img::Tiling::Optimal,
                usage,
                ViewCapabilities::empty(),
            )
            .expect("Failed to create unbound image");

        let image_req = device.get_image_requirements(&image);

        let device_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                image_req.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let image_memory = device
            .allocate_memory(device_type, image_req.size)
            .expect("Failed to allocate image");

        device
            .bind_image_memory(&image_memory, 0, &mut image)
            .expect("Failed to bind image");

        let image_view = device
            .create_image_view(
                &image,
                img::ViewKind::D2,
                format,
                Swizzle::NO,
                img::SubresourceRange {
                    aspects,
                    levels: 0..1,
                    layers: 0..1,
                },
            )
            .expect("Failed to create image view");

        (image, image_memory, image_view)
    }

    pub unsafe fn upload_image_data(
        device: &TDevice,
        physical_device: &TPhysicalDevice,
        command_pool: &mut TCommandPool,
        queue: &mut TCommandQueue,
        fence: &TFence,
        src_image: Option<&RgbaImage>,
        src_depth: Option<&DepthImage>,
        dst_image: &TImage,
    ) {
        let (width, height) = src_image
            .map(|img| img.dimensions())
            .unwrap_or_else(|| src_depth.unwrap().dimensions());
        let image_stride = src_image.map(|_| 4_usize).unwrap_or(2_usize);

        let memory_types = physical_device.memory_properties().memory_types;
        let row_alignment_mask =
            physical_device.limits().optimal_buffer_copy_pitch_alignment as u32 - 1;
        let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
        let upload_size = u64::from(height * row_pitch);

        let (image_upload_buffer, image_upload_memory) = utils::empty_buffer::<u8>(
            &device,
            &memory_types,
            Properties::CPU_VISIBLE,
            buffer::Usage::TRANSFER_SRC,
            upload_size as usize,
        );

        {
            let mapped_memory = device
                .map_memory(&image_upload_memory, Segment::ALL)
                .expect("map_memory failed");

            if let Some(src_image) = src_image {
                for y in 0..height as usize {
                    let row = &(**src_image)[y * (width as usize) * image_stride
                        ..(y + 1) * (width as usize) * image_stride];

                    std::ptr::copy_nonoverlapping(
                        row.as_ptr(),
                        mapped_memory.offset(y as isize * row_pitch as isize),
                        width as usize * image_stride,
                    );
                }
            } else if let Some(src_image) = src_depth {
                let src_image_bytes: &[u8] = {
                    let p: *const u16 = src_image.as_ptr();
                    let len = src_image.len();
                    std::slice::from_raw_parts(p as *const u8, len * 2)
                };
                for y in 0..height as usize {
                    let row = &(*src_image_bytes)[y * (width as usize) * image_stride
                        ..(y + 1) * (width as usize) * image_stride];

                    std::ptr::copy_nonoverlapping(
                        row.as_ptr(),
                        mapped_memory.offset(y as isize * row_pitch as isize),
                        width as usize * image_stride,
                    );
                }
            }

            device
                .flush_mapped_memory_ranges(vec![(&image_upload_memory, Segment::ALL)])
                .expect("flush_mapped_memory_ranges failed");

            device.unmap_memory(&image_upload_memory);
        }

        let submit = {
            let mut cmd_buffer = command_pool.allocate_one(Level::Primary);
            cmd_buffer.begin_primary(CommandBufferFlags::empty());

            let image_barrier = Barrier::Image {
                states: (Access::empty(), Layout::Undefined)
                    ..(Access::TRANSFER_WRITE, Layout::TransferDstOptimal),
                target: dst_image,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };

            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.copy_buffer_to_image(
                &image_upload_buffer,
                dst_image,
                Layout::TransferDstOptimal,
                &[BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: row_pitch / (image_stride as u32),
                    buffer_height: height as u32,
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

            let image_barrier = Barrier::Image {
                states: (Access::TRANSFER_WRITE, Layout::TransferDstOptimal)
                    ..(Access::SHADER_READ, Layout::ShaderReadOnlyOptimal),
                target: dst_image,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };

            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.finish();
            cmd_buffer
        };

        queue.submit_without_semaphores(Some(&submit), Some(&fence));

        device.wait_for_fence(&fence, !0).unwrap();

        // TODO: Can we just pass in a command buffer?
        command_pool.free(Some(submit));

        device.destroy_buffer(image_upload_buffer);
        device.free_memory(image_upload_memory);
    }
}
