use crossbeam_channel::{Receiver, Sender, unbounded};
use glam::{Vec2, Vec3};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::{debug, info};
use voxel_core::{BlockState, CHUNK_SIZE, ChunkCoord, VoxelCoord};
use voxel_mesh::{ChunkMesh, mesh_chunk_greedy};
use voxel_render::{FrameStats, NullRenderer, RenderError, RenderScene, RendererBackend};
use voxel_vulkan::{VulkanRenderer, VulkanRendererConfig};
use voxel_world::{
    BlockRegistry, ChunkProvider, ChunkStreamingState, GeneratedWorld, InMemoryEditLogStore,
    StreamPlanner, TerrainGenerator,
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

    pub fn renderer_mut(&mut self) -> &mut R {
        &mut self.renderer
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
        self.camera_controller
            .apply_input(&mut self.scene, camera_input, dt);

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
}
