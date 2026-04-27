use glam::{Mat4, Vec3};
use std::collections::HashMap;
use voxel_core::ChunkCoord;
use voxel_mesh::{ChunkMesh, MeshVersion};

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("stale mesh for {chunk:?}: submitted {submitted}, resident {resident}")]
    StaleMesh {
        chunk: ChunkCoord,
        submitted: u64,
        resident: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MeshHandle(pub u64);

#[derive(Clone, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub yaw_radians: f32,
    pub pitch_radians: f32,
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        let forward = self.forward();
        Mat4::look_to_rh(self.position, forward, Vec3::Y)
    }

    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(self.vertical_fov_radians, aspect_ratio, self.near, self.far)
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw_radians.cos() * self.pitch_radians.cos(),
            self.pitch_radians.sin(),
            self.yaw_radians.sin() * self.pitch_radians.cos(),
        )
        .normalize_or_zero()
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 48.0, -96.0),
            yaw_radians: 0.0,
            pitch_radians: -0.25,
            vertical_fov_radians: 70.0_f32.to_radians(),
            near: 0.05,
            far: 4096.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MaterialAtlas {
    pub tile_size: u32,
    pub columns: u32,
    pub rows: u32,
}

impl Default for MaterialAtlas {
    fn default() -> Self {
        Self {
            tile_size: 16,
            columns: 16,
            rows: 16,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    pub frame_index: u64,
    pub frame_time_ms: f32,
    pub visible_chunks: usize,
    pub resident_chunks: usize,
    pub mesh_queue_depth: usize,
    pub upload_bytes: u64,
}

#[derive(Clone, Debug)]
pub enum DebugDraw {
    ChunkBounds { coord: ChunkCoord },
    TextLine { label: String, value: String },
}

#[derive(Clone, Debug)]
pub struct RenderScene {
    pub camera: Camera,
    pub atlas: MaterialAtlas,
    pub chunk_meshes: HashMap<ChunkCoord, ResidentChunkMesh>,
    pub debug_draws: Vec<DebugDraw>,
    pub stats: FrameStats,
}

impl Default for RenderScene {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            atlas: MaterialAtlas::default(),
            chunk_meshes: HashMap::new(),
            debug_draws: Vec::new(),
            stats: FrameStats::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ResidentChunkMesh {
    pub handle: MeshHandle,
    pub version: MeshVersion,
    pub quad_count: usize,
}

impl RenderScene {
    pub fn add_or_replace_mesh(
        &mut self,
        handle: MeshHandle,
        mesh: &ChunkMesh,
    ) -> Result<(), RenderError> {
        if let Some(resident) = self.chunk_meshes.get(&mesh.version.chunk) {
            if mesh.version.version < resident.version.version {
                return Err(RenderError::StaleMesh {
                    chunk: mesh.version.chunk,
                    submitted: mesh.version.version,
                    resident: resident.version.version,
                });
            }
        }

        self.chunk_meshes.insert(
            mesh.version.chunk,
            ResidentChunkMesh {
                handle,
                version: mesh.version,
                quad_count: mesh.opaque_quad_count(),
            },
        );
        self.stats.resident_chunks = self.chunk_meshes.len();
        Ok(())
    }
}

pub trait RendererBackend {
    fn upload_chunk_mesh(&mut self, mesh: &ChunkMesh) -> Result<MeshHandle, RenderError>;
    fn remove_chunk_mesh(&mut self, handle: MeshHandle);
    fn render_frame(&mut self, scene: &RenderScene) -> Result<FrameStats, RenderError>;
}

#[derive(Default)]
pub struct NullRenderer {
    next_handle: u64,
}

impl RendererBackend for NullRenderer {
    fn upload_chunk_mesh(&mut self, _mesh: &ChunkMesh) -> Result<MeshHandle, RenderError> {
        self.next_handle += 1;
        Ok(MeshHandle(self.next_handle))
    }

    fn remove_chunk_mesh(&mut self, _handle: MeshHandle) {}

    fn render_frame(&mut self, scene: &RenderScene) -> Result<FrameStats, RenderError> {
        let mut stats = scene.stats.clone();
        stats.frame_index += 1;
        stats.visible_chunks = scene.chunk_meshes.len();
        stats.resident_chunks = scene.chunk_meshes.len();
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_mesh::ChunkMesh;

    #[test]
    fn scene_rejects_stale_mesh_versions() {
        let mut scene = RenderScene::default();
        let current = ChunkMesh::empty(ChunkCoord::ZERO, 2);
        scene.add_or_replace_mesh(MeshHandle(1), &current).unwrap();
        let stale = ChunkMesh::empty(ChunkCoord::ZERO, 1);
        assert!(matches!(
            scene.add_or_replace_mesh(MeshHandle(2), &stale),
            Err(RenderError::StaleMesh { .. })
        ));
    }
}
