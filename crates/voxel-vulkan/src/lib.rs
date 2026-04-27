use ash::{
    Entry, Instance,
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use bytemuck::{Pod, Zeroable};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    collections::{HashMap, HashSet},
    ffi::{CStr, CString, c_void},
    io::Cursor,
    sync::Arc,
};
use vk_mem::{
    Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo, Allocator, AllocatorCreateInfo,
    MemoryUsage,
};
use voxel_core::{CHUNK_SIZE, ChunkCoord};
use voxel_mesh::{ChunkMesh, MeshVertex};
use voxel_render::{DebugDraw, FrameStats, MeshHandle, RenderError, RenderScene, RendererBackend};

pub const REQUIRED_API_VERSION: u32 = vk::API_VERSION_1_3;
const MAX_FRAMES_IN_FLIGHT: usize = 2;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

const CHUNK_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/chunk.vert.spv"));
const CHUNK_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/chunk.frag.spv"));
const OVERLAY_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/overlay.vert.spv"));
const OVERLAY_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/overlay.frag.spv"));

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanFeaturePolicy {
    pub dynamic_rendering: bool,
    pub synchronization2: bool,
    pub timeline_semaphores: bool,
    pub descriptor_indexing: bool,
}

impl Default for VulkanFeaturePolicy {
    fn default() -> Self {
        Self {
            dynamic_rendering: true,
            synchronization2: true,
            timeline_semaphores: true,
            descriptor_indexing: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VulkanRendererConfig {
    pub application_name: String,
    pub enable_validation: bool,
    pub features: VulkanFeaturePolicy,
    pub initial_extent: [u32; 2],
}

impl Default for VulkanRendererConfig {
    fn default() -> Self {
        Self {
            application_name: "voxel-engine".to_owned(),
            enable_validation: cfg!(debug_assertions),
            features: VulkanFeaturePolicy::default(),
            initial_extent: [1280, 720],
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    #[error("failed to load Vulkan entry: {0}")]
    EntryLoad(#[from] ash::LoadingError),
    #[error("window handle error: {0}")]
    WindowHandle(#[from] raw_window_handle::HandleError),
    #[error("Vulkan error: {0:?}")]
    Vk(#[from] vk::Result),
    #[error("Vulkan backend is not initialized: {0}")]
    NotInitialized(&'static str),
    #[error("no suitable Vulkan physical device found")]
    NoSuitableDevice,
    #[error("shader module SPIR-V is invalid: {0}")]
    ShaderSpv(std::io::Error),
    #[error("surface has no usable formats or present modes")]
    IncompleteSurfaceSupport,
    #[error("swapchain extent is zero")]
    ZeroExtent,
}

impl From<VulkanError> for RenderError {
    fn from(value: VulkanError) -> Self {
        RenderError::Backend(value.to_string())
    }
}

#[derive(Clone, Debug)]
pub struct VulkanCapabilities {
    pub api_version: u32,
    pub features: VulkanFeaturePolicy,
    pub device_name: String,
}

impl VulkanCapabilities {
    pub fn modern_desktop_v13() -> Self {
        Self {
            api_version: REQUIRED_API_VERSION,
            features: VulkanFeaturePolicy::default(),
            device_name: "uninitialized".to_owned(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueFamilySelection {
    pub graphics_family: u32,
    pub present_family: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceCandidate {
    pub api_version: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub has_required_extensions: bool,
    pub has_graphics_queue: bool,
    pub has_present_queue: bool,
    pub supports_dynamic_rendering: bool,
    pub supports_synchronization2: bool,
    pub has_surface_formats: bool,
    pub has_present_modes: bool,
}

pub fn score_device_candidate(candidate: DeviceCandidate) -> Option<i32> {
    if candidate.api_version < REQUIRED_API_VERSION
        || !candidate.has_required_extensions
        || !candidate.has_graphics_queue
        || !candidate.has_present_queue
        || !candidate.supports_dynamic_rendering
        || !candidate.supports_synchronization2
        || !candidate.has_surface_formats
        || !candidate.has_present_modes
    {
        return None;
    }

    Some(match candidate.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1_000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 250,
        _ => 100,
    })
}

pub fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    formats
        .iter()
        .copied()
        .find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first().copied())
}

pub fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> Option<vk::PresentModeKHR> {
    modes
        .iter()
        .copied()
        .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
        .or_else(|| {
            modes
                .iter()
                .copied()
                .find(|mode| *mode == vk::PresentModeKHR::FIFO)
        })
        .or_else(|| modes.first().copied())
}

pub fn choose_swapchain_extent(
    capabilities: vk::SurfaceCapabilitiesKHR,
    requested: [u32; 2],
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }

    vk::Extent2D {
        width: requested[0].clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: requested[1].clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

#[derive(Clone)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

struct GpuBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
}

struct GpuImage {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Allocation,
}

struct GpuChunkMesh {
    vertex: Option<GpuBuffer>,
    index: Option<GpuBuffer>,
    index_count: u32,
    chunk_coord: ChunkCoord,
}

struct OverlayMesh {
    vertex: GpuBuffer,
    index: GpuBuffer,
    index_count: u32,
}

struct FrameResources {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
    overlay: Option<OverlayMesh>,
}

struct SwapchainState {
    loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    image_layouts: Vec<vk::ImageLayout>,
    depth: GpuImage,
    depth_layout: vk::ImageLayout,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct PushConstants {
    view_proj: [[f32; 4]; 4],
    chunk_origin: [f32; 4],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct OverlayVertex {
    position: [f32; 2],
    color: [f32; 4],
}

pub struct VulkanRenderer {
    config: VulkanRendererConfig,
    capabilities: VulkanCapabilities,
    entry: Option<Entry>,
    instance: Option<Instance>,
    debug_utils: Option<debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    surface_loader: Option<surface::Instance>,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Option<ash::Device>,
    allocator: Option<Arc<Allocator>>,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    queues: Option<QueueFamilySelection>,
    upload_command_pool: vk::CommandPool,
    frames: Vec<FrameResources>,
    current_frame: usize,
    swapchain: Option<SwapchainState>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    overlay_pipeline_layout: vk::PipelineLayout,
    overlay_pipeline: vk::Pipeline,
    meshes: HashMap<MeshHandle, GpuChunkMesh>,
    retired_meshes: Vec<Vec<GpuChunkMesh>>,
    next_mesh_handle: u64,
    initialized: bool,
    swapchain_dirty: bool,
    pending_extent: [u32; 2],
}

impl VulkanRenderer {
    pub fn uninitialized(config: VulkanRendererConfig) -> Self {
        Self {
            pending_extent: config.initial_extent,
            config,
            capabilities: VulkanCapabilities::modern_desktop_v13(),
            entry: None,
            instance: None,
            debug_utils: None,
            debug_messenger: None,
            surface_loader: None,
            surface: vk::SurfaceKHR::null(),
            physical_device: vk::PhysicalDevice::null(),
            device: None,
            allocator: None,
            graphics_queue: vk::Queue::null(),
            present_queue: vk::Queue::null(),
            queues: None,
            upload_command_pool: vk::CommandPool::null(),
            frames: Vec::new(),
            current_frame: 0,
            swapchain: None,
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            overlay_pipeline_layout: vk::PipelineLayout::null(),
            overlay_pipeline: vk::Pipeline::null(),
            meshes: HashMap::new(),
            retired_meshes: (0..MAX_FRAMES_IN_FLIGHT).map(|_| Vec::new()).collect(),
            next_mesh_handle: 0,
            initialized: false,
            swapchain_dirty: false,
        }
    }

    pub fn new_for_window<W>(config: VulkanRendererConfig, window: &W) -> Result<Self, VulkanError>
    where
        W: HasDisplayHandle + HasWindowHandle + ?Sized,
    {
        Self::new_for_window_with_extent(config, window, [1280, 720])
    }

    pub fn new_for_window_with_extent<W>(
        mut config: VulkanRendererConfig,
        window: &W,
        extent: [u32; 2],
    ) -> Result<Self, VulkanError>
    where
        W: HasDisplayHandle + HasWindowHandle + ?Sized,
    {
        config.initial_extent = [extent[0].max(1), extent[1].max(1)];
        let mut renderer = Self::uninitialized(config);
        renderer.initialize(window)?;
        Ok(renderer)
    }

    pub fn config(&self) -> &VulkanRendererConfig {
        &self.config
    }

    pub fn capabilities(&self) -> &VulkanCapabilities {
        &self.capabilities
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn required_device_extensions() -> &'static [&'static str] {
        &["VK_KHR_swapchain"]
    }

    pub fn required_vulkan_13_features() -> VulkanFeaturePolicy {
        VulkanFeaturePolicy::default()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.pending_extent = [width, height];
        self.swapchain_dirty = width > 0 && height > 0;
    }

    pub fn estimate_mesh_upload_bytes(mesh: &ChunkMesh) -> u64 {
        mesh.opaque_surfaces
            .iter()
            .chain(mesh.transparent_surfaces.iter())
            .map(|surface| {
                let vertex_bytes = surface.vertices.len() * std::mem::size_of::<MeshVertex>();
                let index_bytes = surface.indices.len() * std::mem::size_of::<u32>();
                (vertex_bytes + index_bytes) as u64
            })
            .sum()
    }

    fn initialize<W>(&mut self, window: &W) -> Result<(), VulkanError>
    where
        W: HasDisplayHandle + HasWindowHandle + ?Sized,
    {
        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();

        let entry = unsafe { Entry::load()? };
        let instance = create_instance(&entry, &self.config, display_handle)?;
        let debug_utils = debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = if self.config.enable_validation {
            Some(create_debug_messenger(&debug_utils)?)
        } else {
            None
        };
        let surface_loader = surface::Instance::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?
        };
        let (physical_device, queues, support, device_name) =
            pick_physical_device(&instance, &surface_loader, surface)?;
        let (device, graphics_queue, present_queue) =
            create_device(&instance, physical_device, queues)?;

        let mut allocator_info = AllocatorCreateInfo::new(&instance, &device, physical_device);
        allocator_info.vulkan_api_version = REQUIRED_API_VERSION;
        let allocator = Arc::new(unsafe { Allocator::new(allocator_info)? });

        let upload_command_pool = create_command_pool(&device, queues.graphics_family)?;
        let frames = create_frame_resources(&device, queues.graphics_family)?;

        self.capabilities = VulkanCapabilities {
            api_version: REQUIRED_API_VERSION,
            features: VulkanFeaturePolicy::default(),
            device_name,
        };
        self.entry = Some(entry);
        self.instance = Some(instance);
        self.debug_utils = Some(debug_utils);
        self.debug_messenger = debug_messenger;
        self.surface_loader = Some(surface_loader);
        self.surface = surface;
        self.physical_device = physical_device;
        self.graphics_queue = graphics_queue;
        self.present_queue = present_queue;
        self.queues = Some(queues);
        self.device = Some(device);
        self.allocator = Some(allocator);
        self.upload_command_pool = upload_command_pool;
        self.frames = frames;
        self.initialized = true;
        self.create_or_recreate_swapchain(Some(support))?;
        Ok(())
    }

    fn device(&self) -> Result<&ash::Device, VulkanError> {
        self.device
            .as_ref()
            .ok_or(VulkanError::NotInitialized("logical device"))
    }

    fn allocator(&self) -> Result<&Arc<Allocator>, VulkanError> {
        self.allocator
            .as_ref()
            .ok_or(VulkanError::NotInitialized("allocator"))
    }

    fn create_or_recreate_swapchain(
        &mut self,
        known_support: Option<SwapchainSupport>,
    ) -> Result<(), VulkanError> {
        if self.pending_extent[0] == 0 || self.pending_extent[1] == 0 {
            return Err(VulkanError::ZeroExtent);
        }

        unsafe {
            self.device()?.device_wait_idle()?;
        }
        self.destroy_swapchain_resources();

        let support = match known_support {
            Some(support) => support,
            None => query_swapchain_support(
                self.physical_device,
                self.surface_loader.as_ref().unwrap(),
                self.surface,
            )?,
        };

        if support.formats.is_empty() || support.present_modes.is_empty() {
            return Err(VulkanError::IncompleteSurfaceSupport);
        }

        let surface_format =
            choose_surface_format(&support.formats).ok_or(VulkanError::IncompleteSurfaceSupport)?;
        let present_mode = choose_present_mode(&support.present_modes)
            .ok_or(VulkanError::IncompleteSurfaceSupport)?;
        let extent = choose_swapchain_extent(support.capabilities, self.pending_extent);
        if extent.width == 0 || extent.height == 0 {
            return Err(VulkanError::ZeroExtent);
        }

        let mut image_count = support.capabilities.min_image_count + 1;
        if support.capabilities.max_image_count > 0 {
            image_count = image_count.min(support.capabilities.max_image_count);
        }

        let queues = self.queues.expect("queues are set during initialization");
        let queue_family_indices = [queues.graphics_family, queues.present_family];
        let mut create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        if queues.graphics_family != queues.present_family {
            create_info = create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices);
        } else {
            create_info = create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        }

        let loader = swapchain::Device::new(self.instance.as_ref().unwrap(), self.device()?);
        let swapchain = unsafe { loader.create_swapchain(&create_info, None)? };
        let images = unsafe { loader.get_swapchain_images(swapchain)? };
        let image_views = images
            .iter()
            .map(|image| {
                create_image_view(
                    self.device()?,
                    *image,
                    surface_format.format,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let depth = self.create_depth_image(extent)?;

        self.swapchain = Some(SwapchainState {
            loader,
            swapchain,
            extent,
            image_layouts: vec![vk::ImageLayout::UNDEFINED; images.len()],
            images,
            image_views,
            depth,
            depth_layout: vk::ImageLayout::UNDEFINED,
        });

        self.create_pipeline(surface_format.format)?;
        self.swapchain_dirty = false;
        Ok(())
    }

    fn create_depth_image(&self, extent: vk::Extent2D) -> Result<GpuImage, VulkanError> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(DEPTH_FORMAT)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let allocation_info = AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (image, allocation) = unsafe {
            self.allocator()?
                .create_image(&image_info, &allocation_info)?
        };
        let view = create_image_view(
            self.device()?,
            image,
            DEPTH_FORMAT,
            vk::ImageAspectFlags::DEPTH,
        )?;
        Ok(GpuImage {
            image,
            view,
            allocation,
        })
    }

    fn create_pipeline(&mut self, color_format: vk::Format) -> Result<(), VulkanError> {
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                self.device()?.destroy_pipeline(self.pipeline, None);
                self.pipeline = vk::Pipeline::null();
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device()?
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.pipeline_layout = vk::PipelineLayout::null();
            }
            if self.overlay_pipeline != vk::Pipeline::null() {
                self.device()?.destroy_pipeline(self.overlay_pipeline, None);
                self.overlay_pipeline = vk::Pipeline::null();
            }
            if self.overlay_pipeline_layout != vk::PipelineLayout::null() {
                self.device()?
                    .destroy_pipeline_layout(self.overlay_pipeline_layout, None);
                self.overlay_pipeline_layout = vk::PipelineLayout::null();
            }
        }

        let device = self.device()?;
        let vert_shader = create_shader_module(device, CHUNK_VERT_SPV)?;
        let frag_shader = create_shader_module(device, CHUNK_FRAG_SPV)?;
        let main = c"main";
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(main),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(main),
        ];

        let binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<MeshVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let attributes = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 24,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32_UINT,
                offset: 32,
            },
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R32_UINT,
                offset: 36,
            },
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding))
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32);
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let color_formats = [color_format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(DEPTH_FORMAT);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering)
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| err)?[0]
        };
        unsafe {
            device.destroy_shader_module(vert_shader, None);
            device.destroy_shader_module(frag_shader, None);
        }
        self.pipeline_layout = pipeline_layout;
        self.pipeline = pipeline;
        let (overlay_pipeline_layout, overlay_pipeline) =
            self.create_overlay_pipeline(color_format)?;
        self.overlay_pipeline_layout = overlay_pipeline_layout;
        self.overlay_pipeline = overlay_pipeline;
        Ok(())
    }

    fn create_overlay_pipeline(
        &self,
        color_format: vk::Format,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline), VulkanError> {
        let device = self.device()?;
        let vert_shader = create_shader_module(device, OVERLAY_VERT_SPV)?;
        let frag_shader = create_shader_module(device, OVERLAY_FRAG_SPV)?;
        let main = c"main";
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(main),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(main),
        ];

        let binding = vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<OverlayVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let attributes = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 8,
            },
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding))
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let color_formats = [color_format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(DEPTH_FORMAT);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering)
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, err)| err)?[0]
        };
        unsafe {
            device.destroy_shader_module(vert_shader, None);
            device.destroy_shader_module(frag_shader, None);
        }
        Ok((pipeline_layout, pipeline))
    }

    fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryUsage,
        required_flags: vk::MemoryPropertyFlags,
        allocation_flags: AllocationCreateFlags,
    ) -> Result<GpuBuffer, VulkanError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let allocation_info = AllocationCreateInfo {
            usage: memory_usage,
            required_flags,
            flags: allocation_flags,
            ..Default::default()
        };
        let (buffer, allocation) = unsafe {
            self.allocator()?
                .create_buffer(&buffer_info, &allocation_info)?
        };
        Ok(GpuBuffer { buffer, allocation })
    }

    fn upload_bytes_to_buffer(
        &self,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<GpuBuffer, VulkanError> {
        let size = bytes.len().max(1) as vk::DeviceSize;
        let mut staging = self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        )?;
        unsafe {
            let data = self.allocator()?.map_memory(&mut staging.allocation)?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
            self.allocator()?.unmap_memory(&mut staging.allocation);
        }

        let device_buffer = self.create_buffer(
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryUsage::AutoPreferDevice,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            AllocationCreateFlags::empty(),
        )?;
        self.copy_buffer(staging.buffer, device_buffer.buffer, size)?;
        self.destroy_buffer(staging);
        Ok(device_buffer)
    }

    fn create_host_buffer_from_bytes(
        &self,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<GpuBuffer, VulkanError> {
        let size = bytes.len().max(1) as vk::DeviceSize;
        let mut buffer = self.create_buffer(
            size,
            usage,
            MemoryUsage::AutoPreferHost,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        )?;
        unsafe {
            let data = self.allocator()?.map_memory(&mut buffer.allocation)?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
            self.allocator()?.unmap_memory(&mut buffer.allocation);
        }
        Ok(buffer)
    }

    fn prepare_overlay_mesh(
        &mut self,
        frame_index: usize,
        scene: &RenderScene,
        extent: vk::Extent2D,
    ) -> Result<(), VulkanError> {
        if let Some(overlay) = self.frames[frame_index].overlay.take() {
            self.destroy_overlay_mesh(overlay);
        }

        let (vertices, indices) = build_debug_overlay(scene, extent);
        if vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }

        let vertex = self.create_host_buffer_from_bytes(
            bytemuck::cast_slice(&vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        let index = self.create_host_buffer_from_bytes(
            bytemuck::cast_slice(&indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        self.frames[frame_index].overlay = Some(OverlayMesh {
            vertex,
            index,
            index_count: indices.len() as u32,
        });
        Ok(())
    }

    fn copy_buffer(
        &self,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<(), VulkanError> {
        let device = self.device()?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.upload_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(command_buffer, &begin)?;
            let region = vk::BufferCopy::default().size(size);
            device.cmd_copy_buffer(command_buffer, src, dst, std::slice::from_ref(&region));
            device.end_command_buffer(command_buffer)?;
            let submit =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
            device.queue_submit(
                self.graphics_queue,
                std::slice::from_ref(&submit),
                vk::Fence::null(),
            )?;
            device.queue_wait_idle(self.graphics_queue)?;
            device.free_command_buffers(self.upload_command_pool, &[command_buffer]);
        }
        Ok(())
    }

    fn destroy_buffer(&self, mut buffer: GpuBuffer) {
        if let Some(allocator) = &self.allocator {
            unsafe {
                allocator.destroy_buffer(buffer.buffer, &mut buffer.allocation);
            }
        }
    }

    fn destroy_mesh(&self, mesh: GpuChunkMesh) {
        if let Some(vertex) = mesh.vertex {
            self.destroy_buffer(vertex);
        }
        if let Some(index) = mesh.index {
            self.destroy_buffer(index);
        }
    }

    fn retire_mesh(&mut self, mesh: GpuChunkMesh) {
        let retire_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        self.retired_meshes[retire_frame].push(mesh);
    }

    fn destroy_retired_meshes(&mut self, frame_index: usize) {
        let meshes = std::mem::take(&mut self.retired_meshes[frame_index]);
        for mesh in meshes {
            self.destroy_mesh(mesh);
        }
    }

    fn destroy_overlay_mesh(&self, overlay: OverlayMesh) {
        self.destroy_buffer(overlay.vertex);
        self.destroy_buffer(overlay.index);
    }

    fn record_command_buffer(
        &mut self,
        frame_index: usize,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        scene: &RenderScene,
    ) -> Result<(), VulkanError> {
        let device = self.device()?.clone();
        let overlay_draw = self.frames[frame_index].overlay.as_ref().map(|overlay| {
            (
                overlay.vertex.buffer,
                overlay.index.buffer,
                overlay.index_count,
            )
        });
        let swapchain = self
            .swapchain
            .as_mut()
            .ok_or(VulkanError::NotInitialized("swapchain"))?;
        unsafe {
            device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(command_buffer, &begin)?;

            transition_image(
                &device,
                command_buffer,
                swapchain.images[image_index],
                swapchain.image_layouts[image_index],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );
            swapchain.image_layouts[image_index] = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            transition_image(
                &device,
                command_buffer,
                swapchain.depth.image,
                swapchain.depth_layout,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
            );
            swapchain.depth_layout = vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;

            let color_clear = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.03, 0.05, 0.08, 1.0],
                },
            };
            let depth_clear = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            };
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(swapchain.image_views[image_index])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(color_clear);
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(swapchain.depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(depth_clear);
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            };
            let rendering = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);

            device.cmd_begin_rendering(command_buffer, &rendering);
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: swapchain.extent.width as f32,
                height: swapchain.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = render_area;
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            let aspect = swapchain.extent.width as f32 / swapchain.extent.height.max(1) as f32;
            let mut projection = scene.camera.projection_matrix(aspect);
            projection.y_axis.y *= -1.0;
            let view_proj = projection * scene.camera.view_matrix();

            for resident in scene.chunk_meshes.values() {
                let Some(mesh) = self.meshes.get(&resident.handle) else {
                    continue;
                };
                if mesh.index_count == 0 {
                    continue;
                }
                let Some(vertex) = &mesh.vertex else {
                    continue;
                };
                let Some(index) = &mesh.index else {
                    continue;
                };
                let origin = chunk_origin(mesh.chunk_coord);
                let push = PushConstants {
                    view_proj: view_proj.to_cols_array_2d(),
                    chunk_origin: [origin[0], origin[1], origin[2], 0.0],
                };
                device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex.buffer], &[0]);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    index.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
            }

            if let Some((vertex_buffer, index_buffer, index_count)) = overlay_draw {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.overlay_pipeline,
                );
                device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(command_buffer, index_count, 1, 0, 0, 0);
            }
            device.cmd_end_rendering(command_buffer);

            transition_image(
                &device,
                command_buffer,
                swapchain.images[image_index],
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::ImageAspectFlags::COLOR,
            );
            swapchain.image_layouts[image_index] = vk::ImageLayout::PRESENT_SRC_KHR;
            device.end_command_buffer(command_buffer)?;
        }
        Ok(())
    }

    fn destroy_swapchain_resources(&mut self) {
        unsafe {
            if let Some(mut swapchain) = self.swapchain.take() {
                if let Some(device) = &self.device {
                    for view in swapchain.image_views.drain(..) {
                        device.destroy_image_view(view, None);
                    }
                    device.destroy_image_view(swapchain.depth.view, None);
                    if let Some(allocator) = &self.allocator {
                        allocator
                            .destroy_image(swapchain.depth.image, &mut swapchain.depth.allocation);
                    }
                    swapchain
                        .loader
                        .destroy_swapchain(swapchain.swapchain, None);
                }
            }
            if self.pipeline != vk::Pipeline::null() {
                self.device().unwrap().destroy_pipeline(self.pipeline, None);
                self.pipeline = vk::Pipeline::null();
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device()
                    .unwrap()
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.pipeline_layout = vk::PipelineLayout::null();
            }
            if self.overlay_pipeline != vk::Pipeline::null() {
                self.device()
                    .unwrap()
                    .destroy_pipeline(self.overlay_pipeline, None);
                self.overlay_pipeline = vk::Pipeline::null();
            }
            if self.overlay_pipeline_layout != vk::PipelineLayout::null() {
                self.device()
                    .unwrap()
                    .destroy_pipeline_layout(self.overlay_pipeline_layout, None);
                self.overlay_pipeline_layout = vk::PipelineLayout::null();
            }
        }
    }
}

impl RendererBackend for VulkanRenderer {
    fn upload_chunk_mesh(&mut self, mesh: &ChunkMesh) -> Result<MeshHandle, RenderError> {
        if !self.initialized {
            return Err(VulkanError::NotInitialized("device, allocator, and upload queues").into());
        }

        self.next_mesh_handle += 1;
        let handle = MeshHandle(self.next_mesh_handle);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for surface in &mesh.opaque_surfaces {
            let base = vertices.len() as u32;
            vertices.extend_from_slice(&surface.vertices);
            indices.extend(surface.indices.iter().map(|index| index + base));
        }

        let gpu_mesh = if indices.is_empty() || vertices.is_empty() {
            GpuChunkMesh {
                vertex: None,
                index: None,
                index_count: 0,
                chunk_coord: mesh.version.chunk,
            }
        } else {
            let vertex_bytes = bytemuck::cast_slice(&vertices);
            let index_bytes = bytemuck::cast_slice(&indices);
            let vertex =
                self.upload_bytes_to_buffer(vertex_bytes, vk::BufferUsageFlags::VERTEX_BUFFER)?;
            let index =
                self.upload_bytes_to_buffer(index_bytes, vk::BufferUsageFlags::INDEX_BUFFER)?;
            GpuChunkMesh {
                vertex: Some(vertex),
                index: Some(index),
                index_count: indices.len() as u32,
                chunk_coord: mesh.version.chunk,
            }
        };

        self.meshes.insert(handle, gpu_mesh);
        Ok(handle)
    }

    fn remove_chunk_mesh(&mut self, handle: MeshHandle) {
        if let Some(mesh) = self.meshes.remove(&handle) {
            self.retire_mesh(mesh);
        }
    }

    fn render_frame(&mut self, scene: &RenderScene) -> Result<FrameStats, RenderError> {
        if !self.initialized {
            return Err(VulkanError::NotInitialized("swapchain and frame graph").into());
        }
        if self.pending_extent[0] == 0 || self.pending_extent[1] == 0 {
            return Ok(scene.stats.clone());
        }
        if self.swapchain_dirty {
            self.create_or_recreate_swapchain(None)?;
        }

        let frame_index = self.current_frame;
        let command_buffer = self.frames[frame_index].command_buffer;
        let image_available = self.frames[frame_index].image_available;
        let render_finished = self.frames[frame_index].render_finished;
        let in_flight = self.frames[frame_index].in_flight;
        unsafe {
            self.device()?
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .map_err(VulkanError::Vk)?;
        }
        self.destroy_retired_meshes(frame_index);

        let extent = self
            .swapchain
            .as_ref()
            .ok_or(VulkanError::NotInitialized("swapchain"))?
            .extent;
        self.prepare_overlay_mesh(frame_index, scene, extent)?;

        let swapchain = self
            .swapchain
            .as_ref()
            .ok_or(VulkanError::NotInitialized("swapchain"))?;
        let acquired = unsafe {
            swapchain.loader.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        };
        let (image_index, suboptimal) = match acquired {
            Ok(value) => value,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.swapchain_dirty = true;
                return Ok(scene.stats.clone());
            }
            Err(err) => return Err(VulkanError::Vk(err).into()),
        };

        unsafe {
            self.device()?
                .reset_fences(&[in_flight])
                .map_err(VulkanError::Vk)?;
        }
        self.record_command_buffer(frame_index, command_buffer, image_index as usize, scene)?;

        let wait_semaphores = [image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [render_finished];
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.device()?
                .queue_submit(
                    self.graphics_queue,
                    std::slice::from_ref(&submit_info),
                    in_flight,
                )
                .map_err(VulkanError::Vk)?;
        }

        let swapchains = [self.swapchain.as_ref().unwrap().swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe {
            self.swapchain
                .as_ref()
                .unwrap()
                .loader
                .queue_present(self.present_queue, &present_info)
        };
        match present_result {
            Ok(present_suboptimal) => {
                if suboptimal || present_suboptimal {
                    self.swapchain_dirty = true;
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                self.swapchain_dirty = true;
            }
            Err(err) => return Err(VulkanError::Vk(err).into()),
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        let mut stats = scene.stats.clone();
        stats.frame_index += 1;
        stats.visible_chunks = scene.chunk_meshes.len();
        stats.resident_chunks = self.meshes.len();
        Ok(stats)
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            if let Some(device) = &self.device {
                let _ = device.device_wait_idle();
            }

            for (_, mesh) in self.meshes.drain() {
                if let Some(vertex) = mesh.vertex {
                    if let Some(allocator) = &self.allocator {
                        let mut allocation = vertex.allocation;
                        allocator.destroy_buffer(vertex.buffer, &mut allocation);
                    }
                }
                if let Some(index) = mesh.index {
                    if let Some(allocator) = &self.allocator {
                        let mut allocation = index.allocation;
                        allocator.destroy_buffer(index.buffer, &mut allocation);
                    }
                }
            }
            for meshes in self.retired_meshes.drain(..) {
                for mesh in meshes {
                    if let Some(vertex) = mesh.vertex {
                        if let Some(allocator) = &self.allocator {
                            let mut allocation = vertex.allocation;
                            allocator.destroy_buffer(vertex.buffer, &mut allocation);
                        }
                    }
                    if let Some(index) = mesh.index {
                        if let Some(allocator) = &self.allocator {
                            let mut allocation = index.allocation;
                            allocator.destroy_buffer(index.buffer, &mut allocation);
                        }
                    }
                }
            }
            self.destroy_swapchain_resources();

            if let Some(device) = &self.device {
                let allocator = self.allocator.clone();
                for mut frame in self.frames.drain(..) {
                    if let (Some(overlay), Some(allocator)) = (frame.overlay.take(), &allocator) {
                        let mut vertex_allocation = overlay.vertex.allocation;
                        allocator.destroy_buffer(overlay.vertex.buffer, &mut vertex_allocation);
                        let mut index_allocation = overlay.index.allocation;
                        allocator.destroy_buffer(overlay.index.buffer, &mut index_allocation);
                    }
                    device.destroy_fence(frame.in_flight, None);
                    device.destroy_semaphore(frame.render_finished, None);
                    device.destroy_semaphore(frame.image_available, None);
                    device.destroy_command_pool(frame.command_pool, None);
                }
                if self.upload_command_pool != vk::CommandPool::null() {
                    device.destroy_command_pool(self.upload_command_pool, None);
                }
            }

            self.allocator.take();

            if let Some(device) = self.device.take() {
                device.destroy_device(None);
            }
            if let Some(surface_loader) = &self.surface_loader {
                if self.surface != vk::SurfaceKHR::null() {
                    surface_loader.destroy_surface(self.surface, None);
                }
            }
            if let (Some(debug_utils), Some(debug_messenger)) =
                (&self.debug_utils, self.debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
            }
            if let Some(instance) = self.instance.take() {
                instance.destroy_instance(None);
            }
        }
    }
}

fn create_instance(
    entry: &Entry,
    config: &VulkanRendererConfig,
    display_handle: raw_window_handle::RawDisplayHandle,
) -> Result<Instance, VulkanError> {
    let app_name = CString::new(config.application_name.as_str())
        .unwrap_or_else(|_| c"voxel-engine".to_owned());
    let engine_name = c"voxel-engine";
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(engine_name)
        .engine_version(1)
        .api_version(REQUIRED_API_VERSION);

    let mut extension_names = ash_window::enumerate_required_extensions(display_handle)?.to_vec();
    if config.enable_validation {
        extension_names.push(debug_utils::NAME.as_ptr());
    }

    let available_validation = config.enable_validation && validation_layer_available(entry)?;
    let layer_names = if available_validation {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_names);
    Ok(unsafe { entry.create_instance(&create_info, None)? })
}

fn validation_layer_available(entry: &Entry) -> Result<bool, VulkanError> {
    let layers = unsafe { entry.enumerate_instance_layer_properties()? };
    Ok(layers.iter().any(|layer| {
        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        name == VALIDATION_LAYER
    }))
}

fn create_debug_messenger(
    debug_utils: &debug_utils::Instance,
) -> Result<vk::DebugUtilsMessengerEXT, VulkanError> {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));
    Ok(unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? })
}

unsafe extern "system" fn vulkan_debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let message = if data.is_null() {
        c"<null>"
    } else {
        unsafe { CStr::from_ptr((*data).p_message) }
    };
    eprintln!(
        "Vulkan validation {severity:?} {ty:?}: {}",
        message.to_string_lossy()
    );
    vk::FALSE
}

fn pick_physical_device(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<
    (
        vk::PhysicalDevice,
        QueueFamilySelection,
        SwapchainSupport,
        String,
    ),
    VulkanError,
> {
    let devices = unsafe { instance.enumerate_physical_devices()? };
    let mut best = None;

    for physical_device in devices {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let queues = find_queue_families(instance, surface_loader, surface, physical_device)?;
        let support = query_swapchain_support(physical_device, surface_loader, surface)?;
        let has_required_extensions = check_device_extensions(instance, physical_device)?;
        let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan13);
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut features);
        }

        let candidate = DeviceCandidate {
            api_version: properties.api_version,
            device_type: properties.device_type,
            has_required_extensions,
            has_graphics_queue: queues.is_some(),
            has_present_queue: queues.is_some(),
            supports_dynamic_rendering: vulkan13.dynamic_rendering == vk::TRUE,
            supports_synchronization2: vulkan13.synchronization2 == vk::TRUE,
            has_surface_formats: !support.formats.is_empty(),
            has_present_modes: !support.present_modes.is_empty(),
        };
        let Some(score) = score_device_candidate(candidate) else {
            continue;
        };
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let queues = queues.expect("candidate checked queue support");
        match &best {
            Some((best_score, _, _, _, _)) if *best_score >= score => {}
            _ => best = Some((score, physical_device, queues, support, device_name)),
        }
    }

    best.map(|(_, physical_device, queues, support, name)| (physical_device, queues, support, name))
        .ok_or(VulkanError::NoSuitableDevice)
}

fn find_queue_families(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<Option<QueueFamilySelection>, VulkanError> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let mut graphics = None;
    let mut present = None;
    for (index, family) in families.iter().enumerate() {
        let index = index as u32;
        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics.get_or_insert(index);
        }
        let supports_present = unsafe {
            surface_loader.get_physical_device_surface_support(physical_device, index, surface)?
        };
        if supports_present {
            present.get_or_insert(index);
        }
        if graphics == Some(index) && present == Some(index) {
            return Ok(Some(QueueFamilySelection {
                graphics_family: index,
                present_family: index,
            }));
        }
    }

    Ok(match (graphics, present) {
        (Some(graphics_family), Some(present_family)) => Some(QueueFamilySelection {
            graphics_family,
            present_family,
        }),
        _ => None,
    })
}

fn check_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<bool, VulkanError> {
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device)? };
    let available = extensions
        .iter()
        .map(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
        .collect::<HashSet<_>>();
    Ok(available.contains(swapchain::NAME))
}

fn query_swapchain_support(
    physical_device: vk::PhysicalDevice,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<SwapchainSupport, VulkanError> {
    Ok(SwapchainSupport {
        capabilities: unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        },
        formats: unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        },
        present_modes: unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        },
    })
}

fn create_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queues: QueueFamilySelection,
) -> Result<(ash::Device, vk::Queue, vk::Queue), VulkanError> {
    let priorities = [1.0f32];
    let mut unique_families = vec![queues.graphics_family];
    if queues.present_family != queues.graphics_family {
        unique_families.push(queues.present_family);
    }
    let queue_infos = unique_families
        .iter()
        .map(|family| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*family)
                .queue_priorities(&priorities)
        })
        .collect::<Vec<_>>();

    let device_extensions = [swapchain::NAME.as_ptr()];
    let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let create_info = vk::DeviceCreateInfo::default()
        .push_next(&mut vulkan13)
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&device_extensions);
    let device = unsafe { instance.create_device(physical_device, &create_info, None)? };
    let graphics_queue = unsafe { device.get_device_queue(queues.graphics_family, 0) };
    let present_queue = unsafe { device.get_device_queue(queues.present_family, 0) };
    Ok((device, graphics_queue, present_queue))
}

fn create_command_pool(device: &ash::Device, family: u32) -> Result<vk::CommandPool, VulkanError> {
    let info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    Ok(unsafe { device.create_command_pool(&info, None)? })
}

fn create_frame_resources(
    device: &ash::Device,
    graphics_family: u32,
) -> Result<Vec<FrameResources>, VulkanError> {
    (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| {
            let command_pool = create_command_pool(device, graphics_family)?;
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            Ok(FrameResources {
                command_pool,
                command_buffer,
                image_available: unsafe { device.create_semaphore(&semaphore_info, None)? },
                render_finished: unsafe { device.create_semaphore(&semaphore_info, None)? },
                in_flight: unsafe { device.create_fence(&fence_info, None)? },
                overlay: None,
            })
        })
        .collect()
}

fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView, VulkanError> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    Ok(unsafe { device.create_image_view(&info, None)? })
}

fn create_shader_module(
    device: &ash::Device,
    bytes: &[u8],
) -> Result<vk::ShaderModule, VulkanError> {
    let words = ash::util::read_spv(&mut Cursor::new(bytes)).map_err(VulkanError::ShaderSpv)?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    Ok(unsafe { device.create_shader_module(&info, None)? })
}

fn transition_image(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
) {
    if old_layout == new_layout {
        return;
    }
    let (src_stage, src_access) = match old_layout {
        vk::ImageLayout::UNDEFINED => (
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
        ),
        vk::ImageLayout::PRESENT_SRC_KHR => (
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        _ => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
        ),
    };
    let (dst_stage, dst_access) = match new_layout {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::PRESENT_SRC_KHR => (
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
        ),
        _ => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        ),
    };
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    let dependency =
        vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
    unsafe {
        device.cmd_pipeline_barrier2(command_buffer, &dependency);
    }
}

fn chunk_origin(coord: ChunkCoord) -> [f32; 3] {
    [
        (coord.x * CHUNK_SIZE) as f32,
        (coord.y * CHUNK_SIZE) as f32,
        (coord.z * CHUNK_SIZE) as f32,
    ]
}

fn build_debug_overlay(
    scene: &RenderScene,
    extent: vk::Extent2D,
) -> (Vec<OverlayVertex>, Vec<u32>) {
    if scene.debug_draws.is_empty() || extent.width == 0 || extent.height == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let scale = 3.0;
    let margin = 12.0;
    let padding = 7.0;
    let line_height = 9.0 * scale;
    let glyph_advance = 6.0 * scale;
    let text_color = [0.86, 0.93, 1.0, 0.96];
    let panel_color = [0.015, 0.02, 0.028, 0.72];

    let mut y = margin;
    for line in scene.debug_draws.iter().filter_map(debug_draw_text) {
        let text = line.to_ascii_uppercase();
        let visible_len = text.chars().take(96).count() as f32;
        let panel_width = visible_len * glyph_advance + padding * 2.0;
        add_overlay_rect(
            &mut vertices,
            &mut indices,
            extent,
            [margin - padding, y - padding * 0.5],
            [panel_width, line_height + padding],
            panel_color,
        );
        add_overlay_text(
            &mut vertices,
            &mut indices,
            extent,
            [margin, y],
            scale,
            &text,
            text_color,
        );
        y += line_height + padding + 2.0;
        if y > extent.height as f32 - line_height {
            break;
        }
    }

    (vertices, indices)
}

fn debug_draw_text(draw: &DebugDraw) -> Option<String> {
    match draw {
        DebugDraw::TextLine { label, value } => Some(format!("{label}: {value}")),
        DebugDraw::ChunkBounds { .. } => None,
    }
}

fn add_overlay_text(
    vertices: &mut Vec<OverlayVertex>,
    indices: &mut Vec<u32>,
    extent: vk::Extent2D,
    origin: [f32; 2],
    scale: f32,
    text: &str,
    color: [f32; 4],
) {
    let mut x = origin[0];
    for ch in text.chars().take(96) {
        if ch == ' ' {
            x += 6.0 * scale;
            continue;
        }
        let glyph = glyph_rows(ch);
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..5 {
                let mask = 1 << (4 - col);
                if bits & mask != 0 {
                    add_overlay_rect(
                        vertices,
                        indices,
                        extent,
                        [x + col as f32 * scale, origin[1] + row as f32 * scale],
                        [scale, scale],
                        color,
                    );
                }
            }
        }
        x += 6.0 * scale;
    }
}

fn add_overlay_rect(
    vertices: &mut Vec<OverlayVertex>,
    indices: &mut Vec<u32>,
    extent: vk::Extent2D,
    origin: [f32; 2],
    size: [f32; 2],
    color: [f32; 4],
) {
    let x0 = origin[0];
    let y0 = origin[1];
    let x1 = origin[0] + size[0];
    let y1 = origin[1] + size[1];
    let base = vertices.len() as u32;
    vertices.extend([
        OverlayVertex {
            position: pixel_to_ndc(x0, y0, extent),
            color,
        },
        OverlayVertex {
            position: pixel_to_ndc(x1, y0, extent),
            color,
        },
        OverlayVertex {
            position: pixel_to_ndc(x1, y1, extent),
            color,
        },
        OverlayVertex {
            position: pixel_to_ndc(x0, y1, extent),
            color,
        },
    ]);
    indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn pixel_to_ndc(x: f32, y: f32, extent: vk::Extent2D) -> [f32; 2] {
    [
        x / extent.width.max(1) as f32 * 2.0 - 1.0,
        y / extent.height.max(1) as f32 * 2.0 - 1.0,
    ]
}

fn glyph_rows(ch: char) -> [u8; 7] {
    match ch {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110,
        ],
        '6' => [
            0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        ':' => [
            0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100,
        ],
        ',' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '_' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111,
        ],
        '/' => [
            0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
        ],
        '(' => [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
        ')' => [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
        '+' => [
            0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
        ],
        '%' => [
            0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011,
        ],
        _ => [
            0b01110, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_scoring_rejects_missing_required_capabilities() {
        let mut candidate = DeviceCandidate {
            api_version: REQUIRED_API_VERSION,
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
            has_required_extensions: true,
            has_graphics_queue: true,
            has_present_queue: true,
            supports_dynamic_rendering: true,
            supports_synchronization2: true,
            has_surface_formats: true,
            has_present_modes: true,
        };
        assert!(score_device_candidate(candidate).is_some());
        candidate.has_present_queue = false;
        assert_eq!(score_device_candidate(candidate), None);
    }

    #[test]
    fn swapchain_preferences_choose_expected_values() {
        let formats = [
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];
        assert_eq!(choose_surface_format(&formats).unwrap(), formats[1]);
        assert_eq!(
            choose_present_mode(&[vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX]).unwrap(),
            vk::PresentModeKHR::MAILBOX
        );
    }

    #[test]
    fn mesh_upload_byte_estimate_matches_surface_buffers() {
        let mut mesh = ChunkMesh::empty(ChunkCoord::ZERO, 0);
        mesh.opaque_surfaces[0].vertices = vec![MeshVertex::zeroed(); 4];
        mesh.opaque_surfaces[0].indices = vec![0, 1, 2, 0, 2, 3];
        assert_eq!(
            VulkanRenderer::estimate_mesh_upload_bytes(&mesh),
            (4 * std::mem::size_of::<MeshVertex>() + 6 * std::mem::size_of::<u32>()) as u64
        );
    }

    #[test]
    fn debug_overlay_builds_screen_space_geometry() {
        let mut scene = RenderScene::default();
        scene.debug_draws.push(DebugDraw::TextLine {
            label: "fps".to_owned(),
            value: "60".to_owned(),
        });
        let (vertices, indices) = build_debug_overlay(
            &scene,
            vk::Extent2D {
                width: 1280,
                height: 720,
            },
        );
        assert!(!vertices.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(indices.len() % 6, 0);
        assert!(vertices.iter().all(|vertex| {
            (-1.0..=1.0).contains(&vertex.position[0]) && (-1.0..=1.0).contains(&vertex.position[1])
        }));
    }
}
