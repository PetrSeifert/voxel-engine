use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use voxel_core::{
    BlockId, BlockState, CHUNK_SIZE, CHUNK_SIZE_USIZE, CHUNK_VOLUME, ChunkCoord, LocalVoxelCoord,
    RegionCoord, VoxelCoord,
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
            ],
        }
    }
}

impl BlockRegistry {
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

#[derive(Clone, Debug)]
pub struct TerrainGenerator {
    seed: u64,
}

impl TerrainGenerator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn height_at(&self, x: i32, z: i32) -> i32 {
        let h = hash2(self.seed, x, z);
        let low = (h & 0x0f) as i32;
        let mid = ((h >> 8) & 0x07) as i32;
        20 + low + mid - 4
    }
}

impl WorldGenerator for TerrainGenerator {
    fn generate_chunk(&self, coord: ChunkCoord) -> Chunk {
        let mut chunk = Chunk::empty(coord);
        let origin = coord.min_voxel();
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let world_x = origin.x + x;
                    let world_y = origin.y + y;
                    let world_z = origin.z + z;
                    let height = self.height_at(world_x, world_z);
                    let state = if world_y > height {
                        BlockState::AIR
                    } else if world_y == height {
                        BlockState::new(BlockId::GRASS)
                    } else if world_y >= height - 3 {
                        BlockState::new(BlockId::DIRT)
                    } else {
                        BlockState::new(BlockId::STONE)
                    };
                    chunk.set_block(
                        LocalVoxelCoord::new_unchecked(x as u8, y as u8, z as u8),
                        state,
                    );
                }
            }
        }
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
                let local = LocalVoxelCoord::new_unchecked(x as u8, y as u8, z as u8);
                if !chunk.block(local).is_air() {
                    light = 0;
                }
                chunk.light_mut().set_skylight(local, light);
            }
        }
    }
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

#[derive(Default)]
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
    chunks: HashMap<ChunkCoord, Chunk>,
}

impl ChunkStore {
    pub fn insert(&mut self, chunk: Chunk) {
        self.chunks.insert(chunk.coord(), chunk);
    }

    pub fn get(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    pub fn get_mut(&mut self, coord: ChunkCoord) -> Option<&mut Chunk> {
        self.chunks.get_mut(&coord)
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
    fn streaming_state_rejects_skipped_steps() {
        assert!(ChunkStreamingState::Missing.can_transition_to(ChunkStreamingState::Generating));
        assert!(!ChunkStreamingState::Missing.can_transition_to(ChunkStreamingState::Meshing));
    }
}
