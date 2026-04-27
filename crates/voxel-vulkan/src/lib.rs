use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use voxel_mesh::ChunkMesh;
use voxel_render::{FrameStats, MeshHandle, RenderError, RenderScene, RendererBackend};

pub const REQUIRED_API_VERSION: u32 = vk::API_VERSION_1_3;

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

#[derive(Clone, Copy, Debug)]
pub struct WindowSurfaceHandles {
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
}

#[derive(Clone, Debug)]
pub struct VulkanRendererConfig {
    pub application_name: String,
    pub enable_validation: bool,
    pub features: VulkanFeaturePolicy,
}

impl Default for VulkanRendererConfig {
    fn default() -> Self {
        Self {
            application_name: "voxel-engine".to_owned(),
            enable_validation: cfg!(debug_assertions),
            features: VulkanFeaturePolicy::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    #[error("Vulkan backend is not initialized: {0}")]
    NotInitialized(&'static str),
    #[error("Vulkan error: {0:?}")]
    Vk(vk::Result),
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
}

impl VulkanCapabilities {
    pub fn modern_desktop_v13() -> Self {
        Self {
            api_version: REQUIRED_API_VERSION,
            features: VulkanFeaturePolicy::default(),
        }
    }
}

pub struct VulkanRenderer {
    config: VulkanRendererConfig,
    capabilities: VulkanCapabilities,
    next_mesh_handle: u64,
    initialized: bool,
}

impl VulkanRenderer {
    pub fn uninitialized(config: VulkanRendererConfig) -> Self {
        Self {
            config,
            capabilities: VulkanCapabilities::modern_desktop_v13(),
            next_mesh_handle: 0,
            initialized: false,
        }
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

    pub fn estimate_mesh_upload_bytes(mesh: &ChunkMesh) -> u64 {
        mesh.opaque_surfaces
            .iter()
            .chain(mesh.transparent_surfaces.iter())
            .map(|surface| {
                let vertex_bytes =
                    surface.vertices.len() * std::mem::size_of::<voxel_mesh::MeshVertex>();
                let index_bytes = surface.indices.len() * std::mem::size_of::<u32>();
                (vertex_bytes + index_bytes) as u64
            })
            .sum()
    }
}

impl RendererBackend for VulkanRenderer {
    fn upload_chunk_mesh(&mut self, _mesh: &ChunkMesh) -> Result<MeshHandle, RenderError> {
        if !self.initialized {
            return Err(VulkanError::NotInitialized(
                "device, allocator, and upload queues are pending",
            )
            .into());
        }
        self.next_mesh_handle += 1;
        Ok(MeshHandle(self.next_mesh_handle))
    }

    fn remove_chunk_mesh(&mut self, _handle: MeshHandle) {}

    fn render_frame(&mut self, _scene: &RenderScene) -> Result<FrameStats, RenderError> {
        if !self.initialized {
            return Err(
                VulkanError::NotInitialized("swapchain and frame graph are pending").into(),
            );
        }
        Ok(FrameStats::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_policy_matches_v1_architecture() {
        let features = VulkanRenderer::required_vulkan_13_features();
        assert!(features.dynamic_rendering);
        assert!(features.synchronization2);
        assert!(features.timeline_semaphores);
        assert!(features.descriptor_indexing);
    }
}
