use noise::{NoiseFn, Perlin};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use voxel_core::{
    BlockId, BlockState, CHUNK_AREA, CHUNK_SIZE, CHUNK_SIZE_USIZE, CHUNK_VOLUME, ChunkCoord,
    LocalVoxelCoord, RegionCoord, VoxelCoord,
};

#[derive(Debug, thiserror::Error)]
pub enum WorldError {
    #[error("chunk {actual:?} does not match expected coordinate {expected:?}")]
    ChunkCoordMismatch {
        expected: ChunkCoord,
        actual: ChunkCoord,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockMaterial {
    pub id: BlockId,
    pub name: String,
    pub opaque: bool,
    pub atlas_tile: u16,
    pub emits_light: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockRegistry {
    materials: Vec<BlockMaterial>,
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self {
            materials: vec![
                BlockMaterial {
                    id: BlockId::AIR,
                    name: "air".to_owned(),
                    opaque: false,
                    atlas_tile: 0,
                    emits_light: 0,
                },
                BlockMaterial {
                    id: BlockId::STONE,
                    name: "stone".to_owned(),
                    opaque: true,
                    atlas_tile: 1,
                    emits_light: 0,
                },
                BlockMaterial {
                    id: BlockId::DIRT,
                    name: "dirt".to_owned(),
                    opaque: true,
                    atlas_tile: 2,
                    emits_light: 0,
                },
                BlockMaterial {
                    id: BlockId::GRASS,
                    name: "grass".to_owned(),
                    opaque: true,
                    atlas_tile: 3,
                    emits_light: 0,
                },
                BlockMaterial {
                    id: BlockId::WATER,
                    name: "water".to_owned(),
                    opaque: false,
                    atlas_tile: 4,
                    emits_light: 0,
                },
                BlockMaterial {
                    id: BlockId::SAND,
                    name: "sand".to_owned(),
                    opaque: true,
                    atlas_tile: 5,
                    emits_light: 0,
                },
                BlockMaterial {
                    id: BlockId::WOOD,
                    name: "wood".to_owned(),
                    opaque: true,
                    atlas_tile: 6,
                    emits_light: 0,
                },
                BlockMaterial {
                    id: BlockId::LEAVES,
                    name: "leaves".to_owned(),
                    opaque: false,
                    atlas_tile: 7,
                    emits_light: 0,
                },
                BlockMaterial {
                    id: BlockId::SNOW,
                    name: "snow".to_owned(),
                    opaque: true,
                    atlas_tile: 8,
                    emits_light: 0,
                },
            ],
        }
    }
}

impl BlockRegistry {
    pub fn materials(&self) -> &[BlockMaterial] {
        &self.materials
    }

    pub fn material(&self, id: BlockId) -> Option<&BlockMaterial> {
        self.materials.iter().find(|material| material.id == id)
    }

    pub fn is_opaque(&self, state: BlockState) -> bool {
        self.material(state.id)
            .map(|material| material.opaque)
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightField {
    skylight: Vec<u8>,
    block_light: Vec<u8>,
}

impl LightField {
    pub fn new() -> Self {
        Self {
            skylight: vec![0; CHUNK_VOLUME],
            block_light: vec![0; CHUNK_VOLUME],
        }
    }

    pub fn skylight(&self, local: LocalVoxelCoord) -> u8 {
        self.skylight[local.index()]
    }

    pub fn block_light(&self, local: LocalVoxelCoord) -> u8 {
        self.block_light[local.index()]
    }

    pub fn set_skylight(&mut self, local: LocalVoxelCoord, value: u8) {
        self.skylight[local.index()] = value.min(15);
    }

    pub fn set_block_light(&mut self, local: LocalVoxelCoord, value: u8) {
        self.block_light[local.index()] = value.min(15);
    }
}

impl Default for LightField {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    coord: ChunkCoord,
    blocks: Vec<BlockState>,
    light: LightField,
    version: u64,
    dirty: bool,
}

impl Chunk {
    pub fn empty(coord: ChunkCoord) -> Self {
        Self {
            coord,
            blocks: vec![BlockState::AIR; CHUNK_VOLUME],
            light: LightField::new(),
            version: 0,
            dirty: false,
        }
    }

    pub fn coord(&self) -> ChunkCoord {
        self.coord
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn blocks(&self) -> &[BlockState] {
        &self.blocks
    }

    pub fn light(&self) -> &LightField {
        &self.light
    }

    pub fn light_mut(&mut self) -> &mut LightField {
        self.dirty = true;
        &mut self.light
    }

    pub fn block(&self, local: LocalVoxelCoord) -> BlockState {
        self.blocks[local.index()]
    }

    pub fn set_block(&mut self, local: LocalVoxelCoord, state: BlockState) {
        let index = local.index();
        if self.blocks[index] != state {
            self.blocks[index] = state;
            self.version = self.version.wrapping_add(1);
            self.dirty = true;
        }
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }
}

pub trait WorldGenerator: Send + Sync {
    fn generate_chunk(&self, coord: ChunkCoord) -> Chunk;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NaturalTerrainConfig {
    pub sea_level: i32,
    pub max_tree_height: i32,
    pub max_tree_canopy_radius: i32,
}

impl Default for NaturalTerrainConfig {
    fn default() -> Self {
        Self {
            sea_level: 24,
            max_tree_height: 8,
            max_tree_canopy_radius: 3,
        }
    }
}

#[derive(Clone)]
pub struct TerrainGenerator {
    seed: u64,
    config: NaturalTerrainConfig,
    continental: Perlin,
    elevation: Perlin,
    detail: Perlin,
    mountain: Perlin,
    humidity: Perlin,
    temperature: Perlin,
    river: Perlin,
    river_warp: Perlin,
}

impl std::fmt::Debug for TerrainGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerrainGenerator")
            .field("seed", &self.seed)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug)]
struct TerrainColumn {
    height: i32,
    surface: BlockId,
    humidity: f32,
    temperature: f32,
    river: bool,
    water_level: Option<i32>,
}

impl TerrainGenerator {
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, NaturalTerrainConfig::default())
    }

    pub fn with_config(seed: u64, config: NaturalTerrainConfig) -> Self {
        Self {
            seed,
            config,
            continental: Perlin::new(seed32(seed, 0)),
            elevation: Perlin::new(seed32(seed, 1)),
            detail: Perlin::new(seed32(seed, 2)),
            mountain: Perlin::new(seed32(seed, 3)),
            humidity: Perlin::new(seed32(seed, 4)),
            temperature: Perlin::new(seed32(seed, 5)),
            river: Perlin::new(seed32(seed, 6)),
            river_warp: Perlin::new(seed32(seed, 7)),
        }
    }

    pub fn config(&self) -> NaturalTerrainConfig {
        self.config
    }

    pub fn height_at(&self, x: i32, z: i32) -> i32 {
        self.sample_column(x, z).height
    }

    fn sample_column(&self, x: i32, z: i32) -> TerrainColumn {
        let sea = self.config.sea_level;
        let continental = (self.noise2(&self.continental, x, z, 520.0)
            + self.noise2(&self.detail, x, z, 170.0) * 0.32)
            .clamp(-1.0, 1.0);
        let land = smoothstep(-0.20, 0.28, continental);
        let plains = self.noise2(&self.elevation, x, z, 190.0);
        let rolling = self.noise2(&self.detail, x, z, 68.0);
        let mountain_gate = smoothstep(
            0.30,
            0.82,
            self.noise2(&self.mountain, x, z, 420.0) * 0.72 + continental * 0.55,
        );
        let ridge = (1.0 - self.noise2(&self.mountain, x, z, 105.0).abs()).powf(2.6);
        let mountain_lift =
            mountain_gate * (ridge * 82.0 + self.noise2(&self.detail, x, z, 43.0).max(0.0) * 14.0);

        let sea_floor = sea as f32 - 12.0 + (continental + 1.0) * 7.0;
        let land_height = 28.0 + land * 12.0 + plains * 9.0 + rolling * 4.0 + mountain_lift;
        let coast_height = lerp(sea as f32 - 3.0, land_height, land);
        let mut height = if continental < -0.20 {
            sea_floor
        } else {
            coast_height
        };

        let river_noise = (self.noise2(&self.river, x, z, 235.0)
            + self.noise2(&self.river_warp, x, z, 82.0) * 0.28)
            .abs();
        let river_strength =
            smoothstep(0.070, 0.018, river_noise) * smoothstep(0.02, 0.24, continental);
        let river = river_strength > 0.20 && height > sea as f32 + 1.0;
        let river_water_level = if river {
            let coast_blend = smoothstep(0.08, 0.24, continental);
            Some(
                lerp(sea as f32, height, coast_blend)
                    .round()
                    .clamp(5.0, 128.0) as i32,
            )
        } else {
            None
        };
        if river {
            let carve_depth = (river_strength * 4.0).round().clamp(1.0, 4.0);
            height -= carve_depth;
            if let Some(water_level) = river_water_level {
                height = height.min((water_level - 1) as f32);
            }
        }

        let height = height.round().clamp(4.0, 128.0) as i32;
        let water_level = if height < sea {
            Some(sea)
        } else if river {
            river_water_level
        } else {
            None
        };
        let humidity = ((self.noise2(&self.humidity, x, z, 360.0) + 1.0) * 0.5).clamp(0.0, 1.0);
        let temperature = (((self.noise2(&self.temperature, x, z, 430.0) + 1.0) * 0.5)
            - (height as f32 - 48.0).max(0.0) / 120.0)
            .clamp(0.0, 1.0);

        let surface = if height <= sea + 1 || river {
            BlockId::SAND
        } else if height >= 88 && temperature < 0.42 {
            BlockId::SNOW
        } else if height >= 74 && mountain_gate > 0.42 {
            BlockId::STONE
        } else if continental < -0.02 || height <= sea + 3 {
            BlockId::SAND
        } else {
            BlockId::GRASS
        };

        TerrainColumn {
            height,
            surface,
            humidity,
            temperature,
            river,
            water_level,
        }
    }

    fn block_for_column_y(&self, column: TerrainColumn, world_y: i32) -> BlockState {
        if world_y > column.height {
            if column
                .water_level
                .is_some_and(|water_level| world_y <= water_level)
            {
                BlockState::new(BlockId::WATER)
            } else {
                BlockState::AIR
            }
        } else if world_y == column.height {
            BlockState::new(column.surface)
        } else if world_y >= column.height - 3 {
            if column.surface == BlockId::SAND {
                BlockState::new(BlockId::SAND)
            } else {
                BlockState::new(BlockId::DIRT)
            }
        } else {
            BlockState::new(BlockId::STONE)
        }
    }

    fn noise2(&self, noise: &Perlin, x: i32, z: i32, scale: f64) -> f32 {
        noise.get([x as f64 / scale, z as f64 / scale]) as f32
    }

    fn should_place_tree(&self, x: i32, z: i32, column: TerrainColumn) -> bool {
        if column.surface != BlockId::GRASS || column.height <= self.config.sea_level + 2 {
            return false;
        }
        if column.river || column.temperature < 0.28 || column.height > 72 {
            return false;
        }

        let forest_density = (column.humidity * 0.18 + 0.025).clamp(0.02, 0.20);
        let patch = ((self.noise2(&self.humidity, x, z, 95.0) + 1.0) * 0.5).clamp(0.0, 1.0);
        if patch < 0.42 {
            return false;
        }

        hash_unit(self.seed ^ 0x7472_6565, x, z) < forest_density
    }

    fn tree_shape(&self, x: i32, z: i32) -> (i32, i32) {
        let h = hash2(self.seed ^ 0x6f61_6b, x, z);
        let max_height = self.config.max_tree_height.max(4);
        let height = 4 + (h % (max_height - 3) as u64) as i32;
        let canopy = 2 + ((h >> 8) & 0x01) as i32;
        (
            height,
            canopy.min(self.config.max_tree_canopy_radius.max(2)),
        )
    }

    fn tree_column_pad(&self) -> i32 {
        self.config.max_tree_canopy_radius.max(2) + 1
    }

    fn place_trees_in_chunk(&self, chunk: &mut Chunk, columns: &mut TerrainColumnCache<'_>) {
        let origin = chunk.coord.min_voxel();
        let pad = self.tree_column_pad();
        for root_z in origin.z - pad..origin.z + CHUNK_SIZE + pad {
            for root_x in origin.x - pad..origin.x + CHUNK_SIZE + pad {
                let column = columns.sample(root_x, root_z);
                if !self.should_place_tree(root_x, root_z, column) {
                    continue;
                }
                self.place_tree(chunk, root_x, column.height + 1, root_z);
            }
        }
    }

    fn place_tree(&self, chunk: &mut Chunk, root_x: i32, root_y: i32, root_z: i32) {
        let (height, canopy_radius) = self.tree_shape(root_x, root_z);
        for dy in 0..height {
            set_block_if_in_chunk(
                chunk,
                VoxelCoord::new(root_x, root_y + dy, root_z),
                BlockState::new(BlockId::WOOD),
            );
        }

        let canopy_base = root_y + height - 2;
        let canopy_top = root_y + height + 1;
        for y in canopy_base..=canopy_top {
            let layer_radius = if y == canopy_top {
                canopy_radius - 1
            } else {
                canopy_radius
            };
            for z in root_z - layer_radius..=root_z + layer_radius {
                for x in root_x - layer_radius..=root_x + layer_radius {
                    let dist = (x - root_x).abs() + (z - root_z).abs();
                    if dist > layer_radius + 1 {
                        continue;
                    }
                    if x == root_x && z == root_z && y < root_y + height {
                        continue;
                    }
                    set_block_if_replaceable(
                        chunk,
                        VoxelCoord::new(x, y, z),
                        BlockState::new(BlockId::LEAVES),
                    );
                }
            }
        }
    }
}

struct TerrainColumnCache<'a> {
    generator: &'a TerrainGenerator,
    columns: HashMap<(i32, i32), TerrainColumn>,
}

impl<'a> TerrainColumnCache<'a> {
    fn with_capacity(generator: &'a TerrainGenerator, capacity: usize) -> Self {
        Self {
            generator,
            columns: HashMap::with_capacity(capacity),
        }
    }

    fn sample(&mut self, x: i32, z: i32) -> TerrainColumn {
        if let Some(column) = self.columns.get(&(x, z)) {
            return *column;
        }
        let column = self.generator.sample_column(x, z);
        self.columns.insert((x, z), column);
        column
    }
}

impl WorldGenerator for TerrainGenerator {
    fn generate_chunk(&self, coord: ChunkCoord) -> Chunk {
        let mut chunk = Chunk::empty(coord);
        let origin = coord.min_voxel();
        let cached_side = (CHUNK_SIZE + self.tree_column_pad() * 2) as usize;
        let mut columns = TerrainColumnCache::with_capacity(self, cached_side * cached_side);
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let world_x = origin.x + x;
                let world_z = origin.z + z;
                let column = columns.sample(world_x, world_z);
                for y in 0..CHUNK_SIZE {
                    let world_y = origin.y + y;
                    let state = self.block_for_column_y(column, world_y);
                    let index =
                        (y as usize * CHUNK_AREA) + (z as usize * CHUNK_SIZE_USIZE) + x as usize;
                    chunk.blocks[index] = state;
                }
            }
        }
        self.place_trees_in_chunk(&mut chunk, &mut columns);
        compute_basic_skylight(&mut chunk);
        chunk.clear_dirty();
        chunk
    }
}

pub fn compute_basic_skylight(chunk: &mut Chunk) {
    for x in 0..CHUNK_SIZE_USIZE {
        for z in 0..CHUNK_SIZE_USIZE {
            let mut light = 15;
            for y in (0..CHUNK_SIZE_USIZE).rev() {
                let index = (y * CHUNK_AREA) + (z * CHUNK_SIZE_USIZE) + x;
                if blocks_skylight(chunk.blocks[index]) {
                    light = 0;
                }
                chunk.light.skylight[index] = light;
            }
        }
    }
    chunk.dirty = true;
}

fn blocks_skylight(state: BlockState) -> bool {
    !matches!(state.id, BlockId::AIR | BlockId::WATER | BlockId::LEAVES)
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct BlockEdit {
    pub voxel: VoxelCoord,
    pub new_state: BlockState,
    pub sequence: u64,
}

impl BlockEdit {
    pub fn chunk_coord(&self) -> ChunkCoord {
        self.voxel.split_chunk_local().0
    }

    pub fn local_coord(&self) -> LocalVoxelCoord {
        self.voxel.split_chunk_local().1
    }

    pub fn region_coord(&self) -> RegionCoord {
        self.chunk_coord().region_coord()
    }
}

pub trait EditLogStore: Send + Sync {
    fn load_region(&self, region: RegionCoord) -> Vec<BlockEdit>;
    fn append(&mut self, edit: BlockEdit);
}

#[derive(Clone, Default)]
pub struct InMemoryEditLogStore {
    edits_by_region: HashMap<RegionCoord, Vec<BlockEdit>>,
}

impl EditLogStore for InMemoryEditLogStore {
    fn load_region(&self, region: RegionCoord) -> Vec<BlockEdit> {
        self.edits_by_region
            .get(&region)
            .cloned()
            .unwrap_or_default()
    }

    fn append(&mut self, edit: BlockEdit) {
        self.edits_by_region
            .entry(edit.region_coord())
            .or_default()
            .push(edit);
    }
}

#[derive(Default)]
pub struct ChunkStore {
    chunks: HashMap<ChunkCoord, Arc<Chunk>>,
}

impl ChunkStore {
    pub fn insert(&mut self, chunk: Chunk) {
        self.chunks.insert(chunk.coord(), Arc::new(chunk));
    }

    pub fn remove(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        self.chunks
            .remove(&coord)
            .map(|chunk| match Arc::try_unwrap(chunk) {
                Ok(chunk) => chunk,
                Err(shared) => (*shared).clone(),
            })
    }

    pub fn get(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord).map(Arc::as_ref)
    }

    pub fn get_arc(&self, coord: ChunkCoord) -> Option<Arc<Chunk>> {
        self.chunks.get(&coord).cloned()
    }

    pub fn get_mut(&mut self, coord: ChunkCoord) -> Option<&mut Chunk> {
        self.chunks.get_mut(&coord).map(Arc::make_mut)
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

pub trait ChunkProvider {
    fn get_or_generate(&mut self, coord: ChunkCoord) -> &Chunk;
}

pub struct GeneratedWorld<G: WorldGenerator, E: EditLogStore> {
    generator: G,
    edit_log: E,
    chunks: ChunkStore,
    next_edit_sequence: u64,
}

impl<G: WorldGenerator, E: EditLogStore> GeneratedWorld<G, E> {
    pub fn new(generator: G, edit_log: E) -> Self {
        Self {
            generator,
            edit_log,
            chunks: ChunkStore::default(),
            next_edit_sequence: 1,
        }
    }

    pub fn chunks(&self) -> &ChunkStore {
        &self.chunks
    }

    pub fn edits_for_region(&self, region: RegionCoord) -> Vec<BlockEdit> {
        self.edit_log.load_region(region)
    }

    pub fn insert_chunk(&mut self, chunk: Chunk) {
        self.chunks.insert(chunk);
    }

    pub fn remove_chunk(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        self.chunks.remove(coord)
    }

    pub fn edit_block(&mut self, voxel: VoxelCoord, new_state: BlockState) {
        let edit = BlockEdit {
            voxel,
            new_state,
            sequence: self.next_edit_sequence,
        };
        self.next_edit_sequence += 1;
        let (coord, local) = voxel.split_chunk_local();
        if let Some(chunk) = self.chunks.get_mut(coord) {
            chunk.set_block(local, new_state);
            compute_basic_skylight(chunk);
        }
        self.edit_log.append(edit);
    }

    fn generate_with_edits(&self, coord: ChunkCoord) -> Chunk {
        let mut chunk = self.generator.generate_chunk(coord);
        let region_edits = self.edit_log.load_region(coord.region_coord());
        for edit in region_edits {
            let (edit_chunk, local) = edit.voxel.split_chunk_local();
            if edit_chunk == coord {
                chunk.set_block(local, edit.new_state);
            }
        }
        compute_basic_skylight(&mut chunk);
        chunk.clear_dirty();
        chunk
    }
}

impl<G: WorldGenerator, E: EditLogStore> ChunkProvider for GeneratedWorld<G, E> {
    fn get_or_generate(&mut self, coord: ChunkCoord) -> &Chunk {
        if !self.chunks.chunks.contains_key(&coord) {
            let chunk = self.generate_with_edits(coord);
            self.chunks.insert(chunk);
        }
        self.chunks.get(coord).expect("chunk was inserted")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ChunkStreamingState {
    Missing,
    Generating,
    Generated,
    Lighting,
    Meshing,
    UploadQueued,
    Resident,
    Evicting,
}

impl ChunkStreamingState {
    pub fn can_transition_to(self, next: Self) -> bool {
        use ChunkStreamingState::*;
        matches!(
            (self, next),
            (Missing, Generating)
                | (Generating, Generated)
                | (Generated, Lighting)
                | (Lighting, Meshing)
                | (Meshing, UploadQueued)
                | (UploadQueued, Resident)
                | (Resident, Evicting)
                | (Evicting, Missing)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkTicket {
    pub coord: ChunkCoord,
    pub priority: i32,
}

#[derive(Clone, Debug)]
pub struct StreamPlanner {
    pub horizontal_radius: i32,
    pub vertical_radius: i32,
}

impl StreamPlanner {
    pub fn new(horizontal_radius: i32, vertical_radius: i32) -> Self {
        Self {
            horizontal_radius: horizontal_radius.max(0),
            vertical_radius: vertical_radius.max(0),
        }
    }

    pub fn tickets_around(&self, center: ChunkCoord) -> Vec<ChunkTicket> {
        let mut tickets = Vec::new();
        for y in -self.vertical_radius..=self.vertical_radius {
            for z in -self.horizontal_radius..=self.horizontal_radius {
                for x in -self.horizontal_radius..=self.horizontal_radius {
                    let coord = center.offset(x, y, z);
                    tickets.push(ChunkTicket {
                        coord,
                        priority: center.manhattan_distance(coord),
                    });
                }
            }
        }
        tickets.sort_by_key(|ticket| {
            (
                ticket.priority,
                ticket.coord.y,
                ticket.coord.z,
                ticket.coord.x,
            )
        });
        tickets
    }
}

fn set_block_if_in_chunk(chunk: &mut Chunk, voxel: VoxelCoord, state: BlockState) {
    let (coord, local) = voxel.split_chunk_local();
    if coord == chunk.coord {
        chunk.blocks[local.index()] = state;
    }
}

fn set_block_if_replaceable(chunk: &mut Chunk, voxel: VoxelCoord, state: BlockState) {
    let (coord, local) = voxel.split_chunk_local();
    if coord != chunk.coord {
        return;
    }
    let index = local.index();
    if matches!(chunk.blocks[index].id, BlockId::AIR | BlockId::LEAVES) {
        chunk.blocks[index] = state;
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn seed32(seed: u64, stream: u32) -> u32 {
    (seed ^ (stream as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (seed >> 32)) as u32
}

fn hash_unit(seed: u64, x: i32, z: i32) -> f32 {
    let h = hash2(seed, x, z);
    ((h >> 40) as u32) as f32 / (1u32 << 24) as f32
}

fn hash2(seed: u64, x: i32, z: i32) -> u64 {
    let mut h = seed ^ (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= (z as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_classifies_air_and_stone() {
        let registry = BlockRegistry::default();
        assert!(!registry.is_opaque(BlockState::AIR));
        assert!(registry.is_opaque(BlockState::new(BlockId::STONE)));
    }

    #[test]
    fn generated_chunks_are_deterministic() {
        let generator = TerrainGenerator::new(7);
        let a = generator.generate_chunk(ChunkCoord::ZERO);
        let b = generator.generate_chunk(ChunkCoord::ZERO);
        assert_eq!(a.blocks(), b.blocks());
    }

    #[test]
    fn different_seeds_generate_different_chunks() {
        let a = TerrainGenerator::new(7).generate_chunk(ChunkCoord::ZERO);
        let b = TerrainGenerator::new(8).generate_chunk(ChunkCoord::ZERO);
        assert_ne!(a.blocks(), b.blocks());
    }

    #[test]
    fn negative_coordinates_generate_stably() {
        let generator = TerrainGenerator::new(11);
        let coord = ChunkCoord::new(-4, 1, -7);
        assert_eq!(
            generator.generate_chunk(coord).blocks(),
            generator.generate_chunk(coord).blocks()
        );
    }

    #[test]
    fn ocean_columns_fill_water_to_sea_level() {
        let generator = TerrainGenerator::new(0x5eed);
        let sea = generator.config().sea_level;
        let (x, z, column) = find_column(&generator, |column| column.height < sea - 2)
            .expect("seed should contain ocean in sampled area");
        assert_eq!(
            generated_block(&generator, VoxelCoord::new(x, sea, z)).id,
            BlockId::WATER
        );
        assert_eq!(column.surface, BlockId::SAND);
    }

    #[test]
    fn river_corridors_fill_with_water() {
        let generator = TerrainGenerator::new(0x5eed);
        let (x, z, column) = find_column(&generator, |column| column.river)
            .expect("seed should contain a river corridor in sampled area");
        let river_water_level = column.water_level.expect("river should have local water");
        assert!(river_water_level > column.height);
        assert_eq!(
            generated_block(&generator, VoxelCoord::new(x, river_water_level, z)).id,
            BlockId::WATER
        );
        assert_eq!(
            generated_block(&generator, VoxelCoord::new(x, column.height + 1, z)).id,
            BlockId::WATER
        );
    }

    #[test]
    fn sampled_world_contains_mountain_elevations() {
        let generator = TerrainGenerator::new(0x5eed);
        let max_height = sample_columns(&generator)
            .map(|(_, _, column)| column.height)
            .max()
            .unwrap();
        assert!(max_height >= 88, "max sampled height was {max_height}");
    }

    #[test]
    fn trees_only_root_on_valid_land_surfaces() {
        let generator = TerrainGenerator::new(0x5eed);
        let (x, z, column) = sample_columns(&generator)
            .find(|&(x, z, column)| generator.should_place_tree(x, z, column))
            .expect("seed should place at least one tree in sampled area");
        assert_eq!(column.surface, BlockId::GRASS);
        assert!(column.height > generator.config().sea_level + 2);
        assert!(!column.river);
        assert_eq!(
            generated_block(&generator, VoxelCoord::new(x, column.height + 1, z)).id,
            BlockId::WOOD
        );
    }

    #[test]
    fn tree_canopies_generate_across_chunk_boundaries() {
        let generator = TerrainGenerator::new(0x5eed);
        let (root_x, root_z, column, canopy_radius) = find_boundary_tree(&generator)
            .expect("seed should place at least one boundary-crossing tree");

        let leaf_voxel = VoxelCoord::new(
            root_x + canopy_radius,
            column.height + generator.tree_shape(root_x, root_z).0,
            root_z,
        );
        let root_chunk = VoxelCoord::new(root_x, column.height + 1, root_z)
            .split_chunk_local()
            .0;
        let leaf_chunk = leaf_voxel.split_chunk_local().0;
        assert_ne!(leaf_chunk.x, root_chunk.x);
        assert_eq!(generated_block(&generator, leaf_voxel).id, BlockId::LEAVES);
    }

    #[test]
    fn edit_log_replay_changes_generated_chunk() {
        let mut log = InMemoryEditLogStore::default();
        let voxel = VoxelCoord::new(1, 1, 1);
        log.append(BlockEdit {
            voxel,
            new_state: BlockState::new(BlockId::WATER),
            sequence: 1,
        });
        let mut world = GeneratedWorld::new(TerrainGenerator::new(1), log);
        let chunk = world.get_or_generate(ChunkCoord::ZERO);
        assert_eq!(
            chunk.block(LocalVoxelCoord::new_unchecked(1, 1, 1)).id,
            BlockId::WATER
        );
    }

    #[test]
    fn water_and_leaves_do_not_block_basic_skylight() {
        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(1, 30, 1),
            BlockState::new(BlockId::WATER),
        );
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(1, 29, 1),
            BlockState::new(BlockId::LEAVES),
        );
        compute_basic_skylight(&mut chunk);

        assert_eq!(
            chunk
                .light()
                .skylight(LocalVoxelCoord::new_unchecked(1, 28, 1)),
            15
        );

        chunk.set_block(
            LocalVoxelCoord::new_unchecked(1, 27, 1),
            BlockState::new(BlockId::STONE),
        );
        compute_basic_skylight(&mut chunk);

        assert_eq!(
            chunk
                .light()
                .skylight(LocalVoxelCoord::new_unchecked(1, 26, 1)),
            0
        );
    }

    #[test]
    fn streaming_state_rejects_skipped_steps() {
        assert!(ChunkStreamingState::Missing.can_transition_to(ChunkStreamingState::Generating));
        assert!(!ChunkStreamingState::Missing.can_transition_to(ChunkStreamingState::Meshing));
    }

    fn generated_block(generator: &TerrainGenerator, voxel: VoxelCoord) -> BlockState {
        let (coord, local) = voxel.split_chunk_local();
        generator.generate_chunk(coord).block(local)
    }

    fn find_column(
        generator: &TerrainGenerator,
        predicate: impl Fn(TerrainColumn) -> bool,
    ) -> Option<(i32, i32, TerrainColumn)> {
        sample_columns(generator).find(|&(_, _, column)| predicate(column))
    }

    fn sample_columns(
        generator: &TerrainGenerator,
    ) -> impl Iterator<Item = (i32, i32, TerrainColumn)> + '_ {
        (-1024..=1024).step_by(8).flat_map(move |z| {
            (-1024..=1024)
                .step_by(8)
                .map(move |x| (x, z, generator.sample_column(x, z)))
        })
    }

    fn find_boundary_tree(generator: &TerrainGenerator) -> Option<(i32, i32, TerrainColumn, i32)> {
        for chunk_x in -32..=32 {
            for local_x in [30, 31] {
                let x = chunk_x * CHUNK_SIZE + local_x;
                for z in (-1024..=1024).step_by(4) {
                    let column = generator.sample_column(x, z);
                    if !generator.should_place_tree(x, z, column) {
                        continue;
                    }
                    let (_, canopy_radius) = generator.tree_shape(x, z);
                    if x.rem_euclid(CHUNK_SIZE) >= CHUNK_SIZE - canopy_radius {
                        return Some((x, z, column, canopy_radius));
                    }
                }
            }
        }
        None
    }
}
