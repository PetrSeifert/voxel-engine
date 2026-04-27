use crossbeam_channel::{Receiver, Sender, TrySendError, bounded, unbounded};
use glam::{Vec2, Vec3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tracing::{debug, info};
use voxel_core::{BlockState, CHUNK_SIZE, ChunkCoord, VoxelCoord};
use voxel_mesh::{ChunkMesh, mesh_chunk_greedy};
use voxel_render::{
    DebugDraw, FrameStats, NullRenderer, RenderError, RenderScene, RendererBackend,
};
use voxel_vulkan::{VulkanRenderer, VulkanRendererConfig};
use voxel_world::{
    BlockEdit, BlockRegistry, Chunk, ChunkStreamingState, GeneratedWorld, InMemoryEditLogStore,
    StreamPlanner, TerrainGenerator, WorldGenerator, compute_basic_skylight,
};
use winit::{
    application::ApplicationHandler,
    event::{ButtonSource, DeviceEvent, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
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
    pub max_inflight_chunk_jobs: usize,
    pub max_mesh_uploads_per_tick: usize,
    pub chunk_worker_threads: usize,
    pub debug_overlay: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let worker_threads = thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .saturating_sub(1)
            .clamp(1, 4);
        Self {
            seed: 0x5eed,
            horizontal_view_distance: 16,
            vertical_view_distance: 8,
            max_chunk_jobs_per_tick: 2,
            max_inflight_chunk_jobs: worker_threads * 2,
            max_mesh_uploads_per_tick: 1,
            chunk_worker_threads: worker_threads,
            debug_overlay: true,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VulkanAppKind {
    Sandbox,
    ShaderLab,
}

#[derive(Clone, Debug)]
pub struct WindowedVulkanAppConfig {
    pub title: String,
    pub kind: VulkanAppKind,
    pub runtime: RuntimeConfig,
}

impl WindowedVulkanAppConfig {
    pub fn sandbox() -> Self {
        Self {
            title: "Voxel Sandbox".to_owned(),
            kind: VulkanAppKind::Sandbox,
            runtime: RuntimeConfig::default(),
        }
    }

    pub fn shader_lab() -> Self {
        Self {
            title: "Voxel Shader Lab".to_owned(),
            kind: VulkanAppKind::ShaderLab,
            runtime: RuntimeConfig {
                horizontal_view_distance: 0,
                vertical_view_distance: 0,
                max_chunk_jobs_per_tick: 1,
                ..RuntimeConfig::default()
            },
        }
    }
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
    pub boost_multiplier: f32,
    pub sensitivity: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            speed_voxels_per_second: 48.0,
            boost_multiplier: 4.0,
            sensitivity: 0.002,
        }
    }
}

impl CameraController {
    pub fn apply_input(&self, scene: &mut RenderScene, input: FlyCameraInput, dt: Duration) {
        if input.look_delta != Vec2::ZERO {
            scene.camera.yaw_radians += input.look_delta.x * self.sensitivity;
            scene.camera.pitch_radians -= input.look_delta.y * self.sensitivity;
            let pitch_limit = std::f32::consts::FRAC_PI_2 - 0.01;
            scene.camera.pitch_radians =
                scene.camera.pitch_radians.clamp(-pitch_limit, pitch_limit);
        }

        let mut local_delta = input.movement;
        if local_delta.length_squared() > 1.0 {
            local_delta = local_delta.normalize();
        }

        let speed_multiplier = if input.boost {
            self.boost_multiplier
        } else {
            1.0
        };
        let forward = scene.camera.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = Vec3::Y;
        let world_delta = right * local_delta.x + up * local_delta.y + forward * local_delta.z;
        scene.camera.position +=
            world_delta * self.speed_voxels_per_second * speed_multiplier * dt.as_secs_f32();
    }

    pub fn move_camera(&self, scene: &mut RenderScene, local_delta: Vec3, dt: Duration) {
        self.apply_input(
            scene,
            FlyCameraInput {
                movement: local_delta,
                ..FlyCameraInput::default()
            },
            dt,
        );
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlyCameraInput {
    pub movement: Vec3,
    pub look_delta: Vec2,
    pub boost: bool,
}

#[derive(Clone, Debug, Default)]
struct FlyCameraInputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    boost: bool,
    pointer_locked: bool,
    pending_look_delta: Vec2,
}

impl FlyCameraInputState {
    fn handle_key(&mut self, key: KeyCode, state: ElementState) -> bool {
        let pressed = state == ElementState::Pressed;
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => self.down = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.boost = pressed,
            _ => return false,
        }
        true
    }

    fn push_look_delta(&mut self, delta: (f64, f64)) {
        if self.pointer_locked {
            self.pending_look_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
        }
    }

    fn frame_input(&mut self) -> FlyCameraInput {
        let look_delta = std::mem::take(&mut self.pending_look_delta);
        FlyCameraInput {
            movement: Vec3::new(
                axis(self.right, self.left),
                axis(self.up, self.down),
                axis(self.forward, self.backward),
            ),
            look_delta,
            boost: self.boost,
        }
    }

    fn clear_motion(&mut self) {
        self.forward = false;
        self.backward = false;
        self.left = false;
        self.right = false;
        self.up = false;
        self.down = false;
        self.boost = false;
        self.pending_look_delta = Vec2::ZERO;
    }
}

fn axis(positive: bool, negative: bool) -> f32 {
    match (positive, negative) {
        (true, false) => 1.0,
        (false, true) => -1.0,
        _ => 0.0,
    }
}

#[derive(Clone, Debug)]
pub struct MeshJobResult {
    pub mesh: ChunkMesh,
    pub upload_bytes: u64,
}

#[derive(Clone, Debug)]
struct ChunkBuildResult {
    chunk: Chunk,
    mesh_result: MeshJobResult,
}

struct ActiveChunkSet {
    ordered: Vec<ChunkCoord>,
    coords: HashSet<ChunkCoord>,
}

#[derive(Clone, Debug)]
struct MeshJob {
    coord: ChunkCoord,
    chunk: Option<Chunk>,
    edits: Vec<BlockEdit>,
}

pub struct StreamingWorkers {
    job_tx: Sender<MeshJob>,
    result_rx: Receiver<ChunkBuildResult>,
    _threads: Vec<JoinHandle<()>>,
}

impl StreamingWorkers {
    fn new(config: &RuntimeConfig, registry: BlockRegistry) -> Self {
        let worker_count = config.chunk_worker_threads.max(1);
        let (job_tx, job_rx) = bounded(0);
        let (result_tx, result_rx) = unbounded();
        let mut threads = Vec::with_capacity(worker_count);

        for worker_index in 0..worker_count {
            let job_rx: Receiver<MeshJob> = job_rx.clone();
            let result_tx: Sender<ChunkBuildResult> = result_tx.clone();
            let registry = registry.clone();
            let generator = TerrainGenerator::new(config.seed);
            let name = format!("chunk-worker-{worker_index}");
            let thread = thread::Builder::new()
                .name(name)
                .spawn(move || {
                    while let Ok(job) = job_rx.recv() {
                        let result = run_mesh_job(job, &generator, &registry);
                        if result_tx.send(result).is_err() {
                            break;
                        }
                    }
                })
                .expect("chunk worker thread should spawn");
            threads.push(thread);
        }

        Self {
            job_tx,
            result_rx,
            _threads: threads,
        }
    }
}

fn run_mesh_job(
    MeshJob {
        coord,
        chunk,
        edits,
    }: MeshJob,
    generator: &TerrainGenerator,
    registry: &BlockRegistry,
) -> ChunkBuildResult {
    let mut chunk = chunk.unwrap_or_else(|| generator.generate_chunk(coord));
    if !edits.is_empty() {
        for edit in edits {
            let (edit_chunk, local) = edit.voxel.split_chunk_local();
            if edit_chunk == coord {
                chunk.set_block(local, edit.new_state);
            }
        }
        compute_basic_skylight(&mut chunk);
        chunk.clear_dirty();
    }

    let mesh = mesh_chunk_greedy(&chunk, registry);
    let upload_bytes = mesh_upload_bytes(&mesh);
    let mesh_result = MeshJobResult { mesh, upload_bytes };
    ChunkBuildResult { chunk, mesh_result }
}

fn mesh_upload_bytes(mesh: &ChunkMesh) -> u64 {
    mesh.opaque_surfaces
        .iter()
        .map(|surface| {
            surface.vertices.len() * std::mem::size_of::<voxel_mesh::MeshVertex>()
                + surface.indices.len() * std::mem::size_of::<u32>()
        })
        .sum::<usize>() as u64
}

pub struct EngineRuntime<R: RendererBackend = NullRenderer> {
    config: RuntimeConfig,
    task_graph: TaskGraph,
    world: GeneratedWorld<TerrainGenerator, InMemoryEditLogStore>,
    stream_planner: StreamPlanner,
    streaming_states: HashMap<ChunkCoord, ChunkStreamingState>,
    pending_uploads: VecDeque<MeshJobResult>,
    inflight_mesh_jobs: HashSet<ChunkCoord>,
    workers: StreamingWorkers,
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
        let registry = BlockRegistry::default();
        let world = GeneratedWorld::new(
            TerrainGenerator::new(config.seed),
            InMemoryEditLogStore::default(),
        );
        let workers = StreamingWorkers::new(&config, registry.clone());
        Self {
            stream_planner: StreamPlanner::new(
                config.horizontal_view_distance,
                config.vertical_view_distance,
            ),
            config,
            task_graph: TaskGraph::default(),
            world,
            streaming_states: HashMap::new(),
            pending_uploads: VecDeque::new(),
            inflight_mesh_jobs: HashSet::new(),
            workers,
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

    pub fn renderer_mut(&mut self) -> &mut R {
        &mut self.renderer
    }

    pub fn set_debug_overlay_enabled(&mut self, enabled: bool) {
        self.config.debug_overlay = enabled;
        if !enabled {
            self.scene.debug_draws.clear();
        }
    }

    pub fn debug_overlay_enabled(&self) -> bool {
        self.config.debug_overlay
    }

    pub fn edit_block(&mut self, voxel: VoxelCoord, state: BlockState) {
        self.world.edit_block(voxel, state);
        let (coord, _) = voxel.split_chunk_local();
        self.streaming_states
            .insert(coord, ChunkStreamingState::Generated);
    }

    pub fn tick(&mut self) -> Result<FrameStats, RuntimeError> {
        self.tick_with_camera_input(FlyCameraInput::default())
    }

    pub fn tick_with_camera_input(
        &mut self,
        camera_input: FlyCameraInput,
    ) -> Result<FrameStats, RuntimeError> {
        let now = Instant::now();
        let dt = now.saturating_duration_since(self.last_tick);
        self.last_tick = now;
        self.scene.stats.frame_time_ms = dt.as_secs_f32() * 1000.0;
        self.camera_controller
            .apply_input(&mut self.scene, camera_input, dt);

        let active_chunks = self.schedule_visible_chunks();
        self.evict_out_of_range_chunks(&active_chunks.coords);
        self.mesh_ready_chunks(&active_chunks.ordered);
        self.collect_mesh_results();
        self.upload_pending_meshes()?;
        self.update_debug_overlay();
        let stats = self.renderer.render_frame(&self.scene)?;
        self.scene.stats = stats.clone();
        Ok(stats)
    }

    fn camera_chunk(&self) -> ChunkCoord {
        let pos = self.scene.camera.position.floor().as_ivec3();
        VoxelCoord::new(pos.x, pos.y, pos.z).split_chunk_local().0
    }

    fn schedule_visible_chunks(&mut self) -> ActiveChunkSet {
        let tickets = self.stream_planner.tickets_around(self.camera_chunk());
        self.scene.stats.mesh_queue_depth = tickets.len();
        let mut ordered = Vec::with_capacity(tickets.len());
        let mut coords = HashSet::with_capacity(tickets.len());
        for ticket in tickets {
            ordered.push(ticket.coord);
            coords.insert(ticket.coord);
            self.streaming_states
                .entry(ticket.coord)
                .or_insert(ChunkStreamingState::Missing);
        }
        ActiveChunkSet { ordered, coords }
    }

    fn evict_out_of_range_chunks(&mut self, active_chunks: &HashSet<ChunkCoord>) {
        self.pending_uploads
            .retain(|result| active_chunks.contains(&result.mesh.version.chunk));

        let stale_coords: Vec<_> = self
            .streaming_states
            .keys()
            .copied()
            .filter(|coord| !active_chunks.contains(coord))
            .collect();

        for coord in stale_coords {
            if let Some(resident) = self.scene.remove_chunk_mesh(coord) {
                self.renderer.remove_chunk_mesh(resident.handle);
            }
            self.world.remove_chunk(coord);
            self.streaming_states.remove(&coord);
            self.inflight_mesh_jobs.remove(&coord);
        }
    }

    fn mesh_ready_chunks(&mut self, ordered_chunks: &[ChunkCoord]) {
        let mut queued = 0;
        let capacity = self
            .config
            .max_inflight_chunk_jobs
            .saturating_sub(self.inflight_mesh_jobs.len());
        if capacity == 0 {
            return;
        }

        for &coord in ordered_chunks {
            if queued >= self.config.max_chunk_jobs_per_tick || queued >= capacity {
                break;
            }
            if !matches!(
                self.streaming_states.get(&coord),
                Some(ChunkStreamingState::Missing | ChunkStreamingState::Generated)
            ) {
                continue;
            }
            if self.inflight_mesh_jobs.contains(&coord) {
                continue;
            }

            self.streaming_states
                .insert(coord, ChunkStreamingState::Generating);

            let chunk = self.world.chunks().get(coord).cloned();
            let edits = if chunk.is_some() {
                Vec::new()
            } else {
                self.world.edits_for_region(coord.region_coord())
            };
            let job = MeshJob {
                coord,
                chunk,
                edits,
            };

            match self.workers.job_tx.try_send(job) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    self.streaming_states
                        .insert(coord, ChunkStreamingState::Missing);
                    break;
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.streaming_states
                        .insert(coord, ChunkStreamingState::Missing);
                    break;
                }
            }

            self.streaming_states
                .insert(coord, ChunkStreamingState::Meshing);
            self.inflight_mesh_jobs.insert(coord);
            queued += 1;
        }
    }

    fn collect_mesh_results(&mut self) {
        while let Ok(result) = self.workers.result_rx.try_recv() {
            let coord = result.mesh_result.mesh.version.chunk;
            let was_inflight = self.inflight_mesh_jobs.remove(&coord);
            if !matches!(
                self.streaming_states.get(&coord),
                Some(ChunkStreamingState::Meshing)
            ) || !was_inflight
            {
                continue;
            }

            self.world.insert_chunk(result.chunk);
            self.streaming_states
                .insert(coord, ChunkStreamingState::UploadQueued);
            self.pending_uploads.push_back(result.mesh_result);
        }
    }

    fn upload_pending_meshes(&mut self) -> Result<(), RuntimeError> {
        let mut upload_bytes = 0;
        let mut uploaded_meshes = 0;
        while uploaded_meshes < self.config.max_mesh_uploads_per_tick {
            let Some(result) = self.pending_uploads.pop_front() else {
                break;
            };
            upload_bytes += result.upload_bytes;
            let handle = self.renderer.upload_chunk_mesh(&result.mesh)?;
            self.scene.add_or_replace_mesh(handle, &result.mesh)?;
            self.streaming_states
                .insert(result.mesh.version.chunk, ChunkStreamingState::Resident);
            uploaded_meshes += 1;
        }
        self.scene.stats.upload_bytes = upload_bytes;
        Ok(())
    }

    fn update_debug_overlay(&mut self) {
        self.scene.debug_draws.clear();
        if !self.config.debug_overlay {
            return;
        }

        let camera = &self.scene.camera;
        let camera_chunk = self.camera_chunk();
        let frame_ms = self.scene.stats.frame_time_ms;
        let fps = if frame_ms > 0.0 {
            1000.0 / frame_ms
        } else {
            0.0
        };
        let resident_chunks = self.scene.chunk_meshes.len();
        let visible_chunks = resident_chunks;
        let mesh_queue_depth = self.inflight_mesh_jobs.len() + self.pending_uploads.len();
        let upload_kib = self.scene.stats.upload_bytes as f32 / 1024.0;
        self.scene.stats.mesh_queue_depth = mesh_queue_depth;

        self.scene.debug_draws.extend([
            DebugDraw::TextLine {
                label: "frame".to_owned(),
                value: format!("{frame_ms:.2} ms  {fps:.0} fps"),
            },
            DebugDraw::TextLine {
                label: "chunks".to_owned(),
                value: format!("visible {visible_chunks}  resident {resident_chunks}"),
            },
            DebugDraw::TextLine {
                label: "queues".to_owned(),
                value: format!("mesh {mesh_queue_depth}  upload {upload_kib:.1} KiB"),
            },
            DebugDraw::TextLine {
                label: "camera".to_owned(),
                value: format!(
                    "{:.1}, {:.1}, {:.1}",
                    camera.position.x, camera.position.y, camera.position.z
                ),
            },
            DebugDraw::TextLine {
                label: "chunk".to_owned(),
                value: format!("{}, {}, {}", camera_chunk.x, camera_chunk.y, camera_chunk.z),
            },
            DebugDraw::TextLine {
                label: "controls".to_owned(),
                value: "LMB lock  Esc release  F3 hud".to_owned(),
            },
        ]);
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

pub fn run_windowed_vulkan_app(
    config: WindowedVulkanAppConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    install_tracing();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(WindowedVulkanApp::new(config))?;
    Ok(())
}

struct WindowedVulkanApp {
    runtime: Option<EngineRuntime<VulkanRenderer>>,
    window: Option<Box<dyn Window>>,
    window_id: Option<WindowId>,
    config: WindowedVulkanAppConfig,
    camera_input: FlyCameraInputState,
}

impl WindowedVulkanApp {
    fn new(config: WindowedVulkanAppConfig) -> Self {
        Self {
            runtime: None,
            window: None,
            window_id: None,
            config,
            camera_input: FlyCameraInputState::default(),
        }
    }

    fn set_pointer_locked(&mut self, locked: bool) {
        self.camera_input.pointer_locked = locked;
        if let Some(window) = &self.window {
            if locked {
                let grabbed = window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                    .is_ok();
                self.camera_input.pointer_locked = grabbed;
                window.set_cursor_visible(!grabbed);
            } else {
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
                self.camera_input.clear_motion();
            }
        }
    }

    fn initialize(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.window.is_some() {
            return Ok(());
        }

        let window = event_loop.create_window(
            WindowAttributes::default()
                .with_title(self.config.title.clone())
                .with_visible(true),
        )?;
        let size = window.surface_size();
        let extent = [size.width.max(1), size.height.max(1)];
        let renderer = VulkanRenderer::new_for_window_with_extent(
            VulkanRendererConfig {
                application_name: self.config.title.clone(),
                initial_extent: extent,
                ..VulkanRendererConfig::default()
            },
            window.as_ref(),
            extent,
        )?;
        let runtime = EngineRuntime::with_renderer(self.config.runtime.clone(), renderer);
        self.window_id = Some(window.id());
        window.request_redraw();
        self.runtime = Some(runtime);
        self.window = Some(window);
        Ok(())
    }
}

impl ApplicationHandler for WindowedVulkanApp {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        if let Err(error) = self.initialize(event_loop) {
            eprintln!("failed to initialize Vulkan app: {error}");
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Focused(false) => {
                self.set_pointer_locked(false);
            }
            WindowEvent::SurfaceResized(size) => {
                if let Some(runtime) = &mut self.runtime {
                    runtime.renderer_mut().resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::Escape && event.state == ElementState::Pressed {
                        self.set_pointer_locked(false);
                    } else if key == KeyCode::F3 && event.state == ElementState::Pressed {
                        if let Some(runtime) = &mut self.runtime {
                            runtime.set_debug_overlay_enabled(!runtime.debug_overlay_enabled());
                        }
                    } else {
                        self.camera_input.handle_key(key, event.state);
                    }
                }
            }
            WindowEvent::PointerButton {
                state,
                button: ButtonSource::Mouse(MouseButton::Left),
                ..
            } if state == ElementState::Pressed => {
                self.set_pointer_locked(true);
            }
            WindowEvent::RedrawRequested => {
                let input = self.camera_input.frame_input();
                if let Some(runtime) = &mut self.runtime {
                    if let Err(error) = runtime.tick_with_camera_input(input) {
                        eprintln!("runtime tick failed: {error}");
                        event_loop.exit();
                        return;
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &dyn ActiveEventLoop,
        _device_id: Option<winit::event::DeviceId>,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::PointerMotion { delta } = event {
            self.camera_input.push_look_delta(delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tick_until_resident<R: RendererBackend>(
        runtime: &mut EngineRuntime<R>,
        resident_chunks: usize,
    ) -> FrameStats {
        let mut stats = runtime.tick().unwrap();
        for _ in 0..100 {
            if stats.resident_chunks >= resident_chunks {
                return stats;
            }
            std::thread::sleep(Duration::from_millis(1));
            stats = runtime.tick().unwrap();
        }
        stats
    }

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
        let stats = tick_until_resident(&mut runtime, 1);
        assert_eq!(stats.resident_chunks, 1);
        assert!(!runtime.scene().debug_draws.is_empty());
    }

    #[test]
    fn runtime_schedules_nearest_chunk_first() {
        let mut runtime = EngineRuntime::new_headless(RuntimeConfig {
            horizontal_view_distance: 1,
            vertical_view_distance: 1,
            max_chunk_jobs_per_tick: 1,
            max_inflight_chunk_jobs: 1,
            max_mesh_uploads_per_tick: 1,
            ..RuntimeConfig::default()
        });

        let camera_chunk = runtime.camera_chunk();
        for _ in 0..100 {
            runtime.tick().unwrap();
            if matches!(
                runtime.streaming_states.get(&camera_chunk),
                Some(
                    ChunkStreamingState::Meshing
                        | ChunkStreamingState::UploadQueued
                        | ChunkStreamingState::Resident
                )
            ) {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        assert!(matches!(
            runtime.streaming_states.get(&camera_chunk),
            Some(
                ChunkStreamingState::Meshing
                    | ChunkStreamingState::UploadQueued
                    | ChunkStreamingState::Resident
            )
        ));
    }

    #[test]
    fn runtime_reprioritizes_after_camera_moves() {
        let mut runtime = EngineRuntime::new_headless(RuntimeConfig {
            horizontal_view_distance: 0,
            vertical_view_distance: 0,
            max_chunk_jobs_per_tick: 1,
            max_inflight_chunk_jobs: 1,
            max_mesh_uploads_per_tick: 1,
            chunk_worker_threads: 1,
            ..RuntimeConfig::default()
        });

        let stale_chunk = runtime.camera_chunk();
        runtime
            .streaming_states
            .insert(stale_chunk, ChunkStreamingState::Meshing);
        runtime.inflight_mesh_jobs.insert(stale_chunk);

        let next_chunk = stale_chunk.offset(2, 0, 0);
        let next_origin = next_chunk.min_voxel();
        runtime.scene.camera.position = Vec3::new(
            next_origin.x as f32 + 1.0,
            next_origin.y as f32 + 1.0,
            next_origin.z as f32 + 1.0,
        );

        let active_chunks = runtime.schedule_visible_chunks();
        runtime.evict_out_of_range_chunks(&active_chunks.coords);
        for _ in 0..100 {
            runtime.mesh_ready_chunks(&active_chunks.ordered);
            if matches!(
                runtime.streaming_states.get(&next_chunk),
                Some(ChunkStreamingState::Meshing)
            ) {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        assert!(!runtime.inflight_mesh_jobs.contains(&stale_chunk));
        assert!(!runtime.streaming_states.contains_key(&stale_chunk));
        assert!(matches!(
            runtime.streaming_states.get(&next_chunk),
            Some(ChunkStreamingState::Meshing)
        ));
        assert!(runtime.inflight_mesh_jobs.contains(&next_chunk));
    }

    #[test]
    fn runtime_unloads_chunks_outside_view_distance() {
        let mut runtime = EngineRuntime::new_headless(RuntimeConfig {
            horizontal_view_distance: 0,
            vertical_view_distance: 0,
            max_chunk_jobs_per_tick: 1,
            max_mesh_uploads_per_tick: 1,
            ..RuntimeConfig::default()
        });

        let initial_chunk = runtime.camera_chunk();
        tick_until_resident(&mut runtime, 1);
        assert!(runtime.scene().chunk_meshes.contains_key(&initial_chunk));
        assert!(runtime.world.chunks().get(initial_chunk).is_some());

        let next_chunk = initial_chunk.offset(2, 0, 0);
        let next_origin = next_chunk.min_voxel();
        runtime.scene.camera.position = Vec3::new(
            next_origin.x as f32 + 1.0,
            next_origin.y as f32 + 1.0,
            next_origin.z as f32 + 1.0,
        );

        tick_until_resident(&mut runtime, 1);

        assert!(!runtime.scene().chunk_meshes.contains_key(&initial_chunk));
        assert!(runtime.scene().chunk_meshes.contains_key(&next_chunk));
        assert!(runtime.world.chunks().get(initial_chunk).is_none());
        assert!(runtime.world.chunks().get(next_chunk).is_some());
        assert_eq!(runtime.scene().chunk_meshes.len(), 1);
        assert_eq!(runtime.world.chunks().len(), 1);
    }

    #[test]
    fn fly_camera_moves_forward_in_view_direction() {
        let controller = CameraController::default();
        let mut scene = RenderScene::default();
        let start = scene.camera.position;
        controller.apply_input(
            &mut scene,
            FlyCameraInput {
                movement: Vec3::Z,
                ..FlyCameraInput::default()
            },
            Duration::from_secs(1),
        );
        let delta = scene.camera.position - start;
        assert!(delta.dot(scene.camera.forward()) > 47.0);
    }

    #[test]
    fn fly_camera_accumulates_key_axes_and_consumes_mouse_delta() {
        let mut input = FlyCameraInputState::default();
        input.pointer_locked = true;
        assert!(input.handle_key(KeyCode::KeyW, ElementState::Pressed));
        assert!(input.handle_key(KeyCode::KeyD, ElementState::Pressed));
        assert!(input.handle_key(KeyCode::ShiftLeft, ElementState::Pressed));
        input.push_look_delta((10.0, -5.0));

        let frame = input.frame_input();
        assert_eq!(frame.movement, Vec3::new(1.0, 0.0, 1.0));
        assert_eq!(frame.look_delta, Vec2::new(10.0, -5.0));
        assert!(frame.boost);
        assert_eq!(input.frame_input().look_delta, Vec2::ZERO);
    }

    #[test]
    fn debug_overlay_can_be_disabled() {
        let mut runtime = EngineRuntime::new_headless(RuntimeConfig {
            horizontal_view_distance: 0,
            vertical_view_distance: 0,
            max_chunk_jobs_per_tick: 1,
            debug_overlay: false,
            ..RuntimeConfig::default()
        });
        runtime.tick().unwrap();
        assert!(runtime.scene().debug_draws.is_empty());
        runtime.set_debug_overlay_enabled(true);
        runtime.tick().unwrap();
        assert!(!runtime.scene().debug_draws.is_empty());
    }
}
