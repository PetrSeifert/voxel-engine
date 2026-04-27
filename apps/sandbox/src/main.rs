fn main() -> Result<(), Box<dyn std::error::Error>> {
    voxel_runtime::run_windowed_vulkan_app(voxel_runtime::WindowedVulkanAppConfig::sandbox())
}
