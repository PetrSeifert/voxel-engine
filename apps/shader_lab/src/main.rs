use voxel_vulkan::{REQUIRED_API_VERSION, VulkanRenderer, VulkanRendererConfig};

fn main() {
    let renderer = VulkanRenderer::uninitialized(VulkanRendererConfig::default());
    println!("shader_lab Vulkan API baseline: 0x{REQUIRED_API_VERSION:08x}");
    println!(
        "required extensions: {:?}",
        VulkanRenderer::required_device_extensions()
    );
    println!(
        "required features: {:?}",
        VulkanRenderer::required_vulkan_13_features()
    );
    println!("backend initialized: {}", renderer.is_initialized());
}
