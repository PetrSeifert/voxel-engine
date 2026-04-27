use crossbeam_channel::{Receiver, Sender, unbounded};
use glam::Vec3;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::{debug, info};
use voxel_core::{BlockState, CHUNK_SIZE, ChunkCoord, VoxelCoord};
use voxel_mesh::{ChunkMesh, mesh_chunk_greedy};
use voxel_render::{FrameStats, NullRenderer, RenderError, RenderScene, RendererBackend};
use voxel_world::{
    BlockRegistry, ChunkProvider, ChunkStreamingState, GeneratedWorld, InMemoryEditLogStore,
    StreamPlanner, TerrainGenerator,
};

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("render error: {0}")]
    Render(#[from] RenderError),
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub seed: u64,
    pub horizontal_view_distance: i32,
    pub vertical_view_distance: i32,
    pub max_chunk_jobs_per_tick: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            seed: 0x5eed,
            horizontal_view_distance: 2,
            vertical_view_distance: 1,
            max_chunk_jobs_per_tick: 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeStage {
    Generate,
    Light,
    Mesh,
    Upload,
    Render,
}

#[derive(Clone, Debug)]
pub struct TaskGraph {
    stages: Vec<RuntimeStage>,
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self {
            stages: vec![
                RuntimeStage::Generate,
                RuntimeStage::Light,
                RuntimeStage::Mesh,
                RuntimeStage::Upload,
                RuntimeStage::Render,
            ],
        }
    }
}

impl TaskGraph {
    pub fn stages(&self) -> &[RuntimeStage] {
        &self.stages
    }
}

#[derive(Clone, Debug)]
pub struct CameraController {
    pub speed_voxels_per_second: f32,
    pub sensitivity: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            speed_voxels_per_second: 48.0,
            sensitivity: 0.002,
        }
    }
}

impl CameraController {
    pub fn move_camera(&self, scene: &mut RenderScene, local_delta: Vec3, dt: Duration) {
        let forward = scene.camera.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = Vec3::Y;
        let world_delta = right * local_delta.x + up * local_delta.y + forward * local_delta.z;
        scene.camera.position += world_delta * self.speed_voxels_per_second * dt.as_secs_f32();
    }
}

#[derive(Clone, Debug)]
pub struct MeshJobResult {
    pub mesh: ChunkMesh,
    pub upload_bytes: u64,
}

pub struct WorkerChannels {
    mesh_tx: Sender<MeshJobResult>,
    mesh_rx: Receiver<MeshJobResult>,
}

impl Default for WorkerChannels {
    fn default() -> Self {
        let (mesh_tx, mesh_rx) = unbounded();
        Self { mesh_tx, mesh_rx }
    }
}

pub struct EngineRuntime<R: RendererBackend = NullRenderer> {
    config: RuntimeConfig,
    task_graph: TaskGraph,
    registry: BlockRegistry,
    world: GeneratedWorld<TerrainGenerator, InMemoryEditLogStore>,
    stream_planner: StreamPlanner,
    streaming_states: HashMap<ChunkCoord, ChunkStreamingState>,
    pending_uploads: VecDeque<MeshJobResult>,
    channels: WorkerChannels,
    renderer: R,
    scene: RenderScene,
    camera_controller: CameraController,
    last_tick: Instant,
}

impl EngineRuntime<NullRenderer> {
    pub fn new_headless(config: RuntimeConfig) -> Self {
        Self::with_renderer(config, NullRenderer::default())
    }
}

impl<R: RendererBackend> EngineRuntime<R> {
    pub fn with_renderer(config: RuntimeConfig, renderer: R) -> Self {
        let world = GeneratedWorld::new(
            TerrainGenerator::new(config.seed),
            InMemoryEditLogStore::default(),
        );
        Self {
            stream_planner: StreamPlanner::new(
                config.horizontal_view_distance,
                config.vertical_view_distance,
            ),
            config,
            task_graph: TaskGraph::default(),
            registry: BlockRegistry::default(),
            world,
            streaming_states: HashMap::new(),
            pending_uploads: VecDeque::new(),
            channels: WorkerChannels::default(),
            renderer,
            scene: RenderScene::default(),
            camera_controller: CameraController::default(),
            last_tick: Instant::now(),
        }
    }

    pub fn task_graph(&self) -> &TaskGraph {
        &self.task_graph
    }

    pub fn scene(&self) -> &RenderScene {
        &self.scene
    }

    pub fn edit_block(&mut self, voxel: VoxelCoord, state: BlockState) {
        self.world.edit_block(voxel, state);
        let (coord, _) = voxel.split_chunk_local();
        self.streaming_states
            .insert(coord, ChunkStreamingState::Generated);
    }

    pub fn tick(&mut self) -> Result<FrameStats, RuntimeError> {
        let now = Instant::now();
        let dt = now.saturating_duration_since(self.last_tick);
        self.last_tick = now;
        self.camera_controller
            .move_camera(&mut self.scene, Vec3::ZERO, dt);

        self.schedule_visible_chunks();
        self.mesh_ready_chunks();
        self.collect_mesh_results();
        self.upload_pending_meshes()?;
        let stats = self.renderer.render_frame(&self.scene)?;
        self.scene.stats = stats.clone();
        Ok(stats)
    }

    fn camera_chunk(&self) -> ChunkCoord {
        let pos = self.scene.camera.position.floor().as_ivec3();
        VoxelCoord::new(pos.x, pos.y, pos.z).split_chunk_local().0
    }

    fn schedule_visible_chunks(&mut self) {
        let tickets = self.stream_planner.tickets_around(self.camera_chunk());
        self.scene.stats.mesh_queue_depth = tickets.len();
        for ticket in tickets {
            self.streaming_states
                .entry(ticket.coord)
                .or_insert(ChunkStreamingState::Missing);
        }
    }

    fn mesh_ready_chunks(&mut self) {
        let mut queued = 0;
        let coords: Vec<_> = self
            .streaming_states
            .iter()
            .filter_map(|(coord, state)| {
                matches!(
                    state,
                    ChunkStreamingState::Missing | ChunkStreamingState::Generated
                )
                .then_some(*coord)
            })
            .collect();

        for coord in coords {
            if queued >= self.config.max_chunk_jobs_per_tick {
                break;
            }

            self.streaming_states
                .insert(coord, ChunkStreamingState::Generating);
            let chunk = self.world.get_or_generate(coord).clone();
            self.streaming_states
                .insert(coord, ChunkStreamingState::Meshing);
            let mesh = mesh_chunk_greedy(&chunk, &self.registry);
            let upload_bytes = mesh
                .opaque_surfaces
                .iter()
                .map(|surface| {
                    surface.vertices.len() * std::mem::size_of::<voxel_mesh::MeshVertex>()
                        + surface.indices.len() * std::mem::size_of::<u32>()
                })
                .sum::<usize>() as u64;
            let result = MeshJobResult { mesh, upload_bytes };
            self.channels
                .mesh_tx
                .send(result)
                .expect("runtime owns mesh receiver");
            queued += 1;
        }
    }

    fn collect_mesh_results(&mut self) {
        while let Ok(result) = self.channels.mesh_rx.try_recv() {
            self.streaming_states
                .insert(result.mesh.version.chunk, ChunkStreamingState::UploadQueued);
            self.pending_uploads.push_back(result);
        }
    }

    fn upload_pending_meshes(&mut self) -> Result<(), RuntimeError> {
        let mut upload_bytes = 0;
        while let Some(result) = self.pending_uploads.pop_front() {
            upload_bytes += result.upload_bytes;
            let handle = self.renderer.upload_chunk_mesh(&result.mesh)?;
            self.scene.add_or_replace_mesh(handle, &result.mesh)?;
            self.streaming_states
                .insert(result.mesh.version.chunk, ChunkStreamingState::Resident);
        }
        self.scene.stats.upload_bytes = upload_bytes;
        Ok(())
    }
}

pub fn chunk_coord_from_world_position(position: Vec3) -> ChunkCoord {
    let pos = position.floor().as_ivec3();
    VoxelCoord::new(pos.x, pos.y, pos.z).split_chunk_local().0
}

pub fn default_spawn_position() -> Vec3 {
    Vec3::new(0.0, CHUNK_SIZE as f32 * 2.0, -CHUNK_SIZE as f32 * 3.0)
}

pub fn install_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    debug!("tracing initialized");
    info!("voxel runtime ready");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_graph_keeps_expected_order() {
        let graph = TaskGraph::default();
        assert_eq!(
            graph.stages(),
            &[
                RuntimeStage::Generate,
                RuntimeStage::Light,
                RuntimeStage::Mesh,
                RuntimeStage::Upload,
                RuntimeStage::Render
            ]
        );
    }

    #[test]
    fn runtime_tick_streams_some_chunks() {
        let mut runtime = EngineRuntime::new_headless(RuntimeConfig {
            horizontal_view_distance: 0,
            vertical_view_distance: 0,
            max_chunk_jobs_per_tick: 1,
            ..RuntimeConfig::default()
        });
        let stats = runtime.tick().unwrap();
        assert_eq!(stats.resident_chunks, 1);
    }
}
