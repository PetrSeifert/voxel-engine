use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use voxel_core::{
    AabbI32, BlockId, BlockState, CHUNK_AREA, CHUNK_SIZE_USIZE, ChunkCoord, Direction,
};
use voxel_world::{BlockRegistry, Chunk};

const SURFACE_QUAD_RESERVE: usize = CHUNK_AREA * Direction::ALL.len();
const VERTICES_PER_QUAD: usize = 4;
const INDICES_PER_QUAD: usize = 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MeshVersion {
    pub chunk: ChunkCoord,
    pub version: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FaceKey {
    material: BlockId,
    direction: Direction,
    ao: u8,
}

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub material: u32,
    pub ao: u32,
}

#[derive(Clone, Debug, Default)]
pub struct MeshSurface {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl MeshSurface {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }
}

#[derive(Clone, Debug)]
pub struct ChunkMesh {
    pub opaque_surfaces: Vec<MeshSurface>,
    pub transparent_surfaces: Vec<MeshSurface>,
    pub bounds: AabbI32,
    pub version: MeshVersion,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkNeighbors<'a> {
    pub neg_x: Option<&'a Chunk>,
    pub pos_x: Option<&'a Chunk>,
    pub neg_y: Option<&'a Chunk>,
    pub pos_y: Option<&'a Chunk>,
    pub neg_z: Option<&'a Chunk>,
    pub pos_z: Option<&'a Chunk>,
}

impl<'a> ChunkNeighbors<'a> {
    pub fn get(self, direction: Direction) -> Option<&'a Chunk> {
        match direction {
            Direction::NegX => self.neg_x,
            Direction::PosX => self.pos_x,
            Direction::NegY => self.neg_y,
            Direction::PosY => self.pos_y,
            Direction::NegZ => self.neg_z,
            Direction::PosZ => self.pos_z,
        }
    }
}

impl ChunkMesh {
    pub fn empty(chunk: ChunkCoord, version: u64) -> Self {
        Self {
            opaque_surfaces: vec![MeshSurface::default()],
            transparent_surfaces: vec![MeshSurface::default()],
            bounds: AabbI32::chunk_bounds(chunk),
            version: MeshVersion { chunk, version },
        }
    }

    pub fn opaque_quad_count(&self) -> usize {
        self.opaque_surfaces
            .iter()
            .map(MeshSurface::quad_count)
            .sum()
    }

    pub fn transparent_quad_count(&self) -> usize {
        self.transparent_surfaces
            .iter()
            .map(MeshSurface::quad_count)
            .sum()
    }

    pub fn quad_count(&self) -> usize {
        self.opaque_quad_count() + self.transparent_quad_count()
    }
}

pub fn mesh_chunk_greedy(chunk: &Chunk, registry: &BlockRegistry) -> ChunkMesh {
    mesh_chunk_greedy_with_neighbors(chunk, registry, ChunkNeighbors::default())
}

pub fn mesh_chunk_greedy_with_neighbors(
    chunk: &Chunk,
    registry: &BlockRegistry,
    neighbors: ChunkNeighbors<'_>,
) -> ChunkMesh {
    let mut mesh = ChunkMesh::empty(chunk.coord(), chunk.version());
    let opaque_surface = mesh
        .opaque_surfaces
        .first_mut()
        .expect("empty mesh creates one opaque surface");
    let transparent_surface = mesh
        .transparent_surfaces
        .first_mut()
        .expect("empty mesh creates one transparent surface");
    // One unmerged quad per cell on each chunk side; interior cavities can grow beyond this.
    opaque_surface
        .vertices
        .reserve(SURFACE_QUAD_RESERVE * VERTICES_PER_QUAD);
    opaque_surface
        .indices
        .reserve(SURFACE_QUAD_RESERVE * INDICES_PER_QUAD);
    transparent_surface
        .vertices
        .reserve(SURFACE_QUAD_RESERVE * VERTICES_PER_QUAD);
    transparent_surface
        .indices
        .reserve(SURFACE_QUAD_RESERVE * INDICES_PER_QUAD);

    let opacity = opacity_table(registry);
    let visibility = visibility_table(registry);
    let blocks = chunk.blocks();

    for direction in Direction::ALL {
        mesh_direction(
            blocks,
            neighbors,
            &opacity,
            &visibility,
            direction,
            opaque_surface,
            transparent_surface,
        );
    }

    mesh
}

fn mesh_direction(
    blocks: &[BlockState],
    neighbors: ChunkNeighbors<'_>,
    opacity: &[bool],
    visibility: &[bool],
    direction: Direction,
    opaque_surface: &mut MeshSurface,
    transparent_surface: &mut MeshSurface,
) {
    let axis = normal_axis(direction);
    let u_axis = (axis + 1) % 3;
    let v_axis = (axis + 2) % 3;
    let mut opaque_mask = vec![None; CHUNK_AREA];
    let mut transparent_mask = vec![None; CHUNK_AREA];

    for slice in 0..CHUNK_SIZE_USIZE {
        opaque_mask.fill(None);
        transparent_mask.fill(None);
        for v in 0..CHUNK_SIZE_USIZE {
            for u in 0..CHUNK_SIZE_USIZE {
                let mut coord = [0usize; 3];
                coord[axis] = slice;
                coord[u_axis] = u;
                coord[v_axis] = v;

                let state = block_at(blocks, coord);
                if !is_visible(state, visibility) {
                    continue;
                }

                if face_is_visible(blocks, neighbors, opacity, visibility, coord, direction) {
                    let state_opaque = is_opaque(state, opacity);
                    let ao = if state_opaque {
                        face_ao(blocks, neighbors, opacity, coord, direction)
                    } else {
                        0
                    };
                    let key = FaceKey {
                        material: state.id,
                        direction,
                        ao,
                    };
                    let mask = if state_opaque {
                        &mut opaque_mask
                    } else {
                        &mut transparent_mask
                    };
                    mask[v * CHUNK_SIZE_USIZE + u] = Some(key);
                }
            }
        }

        emit_greedy_mask(
            opaque_surface,
            &mut opaque_mask,
            axis,
            u_axis,
            v_axis,
            slice,
            direction,
        );
        emit_greedy_mask(
            transparent_surface,
            &mut transparent_mask,
            axis,
            u_axis,
            v_axis,
            slice,
            direction,
        );
    }
}

fn opacity_table(registry: &BlockRegistry) -> Vec<bool> {
    let max_id = registry
        .materials()
        .iter()
        .map(|material| material.id.0 as usize)
        .max()
        .unwrap_or(0);
    let mut opacity = vec![false; max_id + 1];
    for material in registry.materials() {
        opacity[material.id.0 as usize] = material.opaque;
    }
    opacity
}

fn visibility_table(registry: &BlockRegistry) -> Vec<bool> {
    let max_id = registry
        .materials()
        .iter()
        .map(|material| material.id.0 as usize)
        .max()
        .unwrap_or(0);
    let mut visibility = vec![false; max_id + 1];
    for material in registry.materials() {
        visibility[material.id.0 as usize] = !material.id.is_air();
    }
    visibility
}

fn is_opaque(state: BlockState, opacity: &[bool]) -> bool {
    opacity.get(state.id.0 as usize).copied().unwrap_or(false)
}

fn is_visible(state: BlockState, visibility: &[bool]) -> bool {
    visibility
        .get(state.id.0 as usize)
        .copied()
        .unwrap_or(!state.is_air())
}

fn block_index(x: usize, y: usize, z: usize) -> usize {
    y * CHUNK_AREA + z * CHUNK_SIZE_USIZE + x
}

fn block_at(blocks: &[BlockState], coord: [usize; 3]) -> BlockState {
    blocks[block_index(coord[0], coord[1], coord[2])]
}

fn emit_greedy_mask(
    surface: &mut MeshSurface,
    mask: &mut [Option<FaceKey>],
    axis: usize,
    u_axis: usize,
    v_axis: usize,
    slice: usize,
    direction: Direction,
) {
    for v in 0..CHUNK_SIZE_USIZE {
        let mut u = 0;
        while u < CHUNK_SIZE_USIZE {
            let index = v * CHUNK_SIZE_USIZE + u;
            let Some(key) = mask[index] else {
                u += 1;
                continue;
            };

            let mut width = 1;
            while u + width < CHUNK_SIZE_USIZE
                && mask[v * CHUNK_SIZE_USIZE + u + width] == Some(key)
            {
                width += 1;
            }

            let mut height = 1;
            'height: while v + height < CHUNK_SIZE_USIZE {
                for test_u in u..u + width {
                    if mask[(v + height) * CHUNK_SIZE_USIZE + test_u] != Some(key) {
                        break 'height;
                    }
                }
                height += 1;
            }

            for clear_v in v..v + height {
                for clear_u in u..u + width {
                    mask[clear_v * CHUNK_SIZE_USIZE + clear_u] = None;
                }
            }

            emit_quad(
                surface,
                QuadSpec {
                    axis,
                    u_axis,
                    v_axis,
                    slice,
                    u,
                    v,
                    width,
                    height,
                    direction,
                    key,
                },
            );
            u += width;
        }
    }
}

struct QuadSpec {
    axis: usize,
    u_axis: usize,
    v_axis: usize,
    slice: usize,
    u: usize,
    v: usize,
    width: usize,
    height: usize,
    direction: Direction,
    key: FaceKey,
}

fn emit_quad(surface: &mut MeshSurface, spec: QuadSpec) {
    let mut p00 = [0.0f32; 3];
    let mut p10 = [0.0f32; 3];
    let mut p11 = [0.0f32; 3];
    let mut p01 = [0.0f32; 3];

    let plane = if is_positive(spec.direction) {
        spec.slice + 1
    } else {
        spec.slice
    } as f32;

    p00[spec.axis] = plane;
    p10[spec.axis] = plane;
    p11[spec.axis] = plane;
    p01[spec.axis] = plane;

    p00[spec.u_axis] = spec.u as f32;
    p00[spec.v_axis] = spec.v as f32;
    p10[spec.u_axis] = (spec.u + spec.width) as f32;
    p10[spec.v_axis] = spec.v as f32;
    p11[spec.u_axis] = (spec.u + spec.width) as f32;
    p11[spec.v_axis] = (spec.v + spec.height) as f32;
    p01[spec.u_axis] = spec.u as f32;
    p01[spec.v_axis] = (spec.v + spec.height) as f32;

    let normal_i = spec.direction.normal();
    let normal = [normal_i[0] as f32, normal_i[1] as f32, normal_i[2] as f32];
    let base = surface.vertices.len() as u32;
    let mut vertices = [
        vertex(p00, normal, [0.0, 0.0], spec.key),
        vertex(p10, normal, [spec.width as f32, 0.0], spec.key),
        vertex(
            p11,
            normal,
            [spec.width as f32, spec.height as f32],
            spec.key,
        ),
        vertex(p01, normal, [0.0, spec.height as f32], spec.key),
    ];

    let flip = should_flip_winding(spec.direction, spec.u_axis, spec.v_axis);
    if flip {
        vertices.swap(1, 3);
    }

    surface.vertices.extend(vertices);
    surface
        .indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn vertex(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], key: FaceKey) -> MeshVertex {
    MeshVertex {
        position,
        normal,
        uv,
        material: key.material.0 as u32,
        ao: key.ao as u32,
    }
}

fn face_is_visible(
    blocks: &[BlockState],
    neighbors: ChunkNeighbors<'_>,
    opacity: &[bool],
    visibility: &[bool],
    coord: [usize; 3],
    direction: Direction,
) -> bool {
    let state = block_at(blocks, coord);
    let axis = normal_axis(direction);
    let neighbor_axis = if is_positive(direction) {
        coord[axis].checked_add(1)
    } else {
        coord[axis].checked_sub(1)
    };

    let Some(neighbor_axis) = neighbor_axis else {
        return neighbor_face_is_visible(state, neighbors, opacity, visibility, coord, direction);
    };
    if neighbor_axis < CHUNK_SIZE_USIZE {
        let mut neighbor = coord;
        neighbor[axis] = neighbor_axis;
        return face_visible_against(state, block_at(blocks, neighbor), opacity, visibility);
    }

    neighbor_face_is_visible(state, neighbors, opacity, visibility, coord, direction)
}

fn neighbor_face_is_visible(
    state: BlockState,
    neighbors: ChunkNeighbors<'_>,
    opacity: &[bool],
    visibility: &[bool],
    coord: [usize; 3],
    direction: Direction,
) -> bool {
    let Some(neighbor) = neighbors.get(direction) else {
        return true;
    };
    let [x, y, z] = coord;
    let neighbor_coord = match direction {
        Direction::NegX => [CHUNK_SIZE_USIZE - 1, y, z],
        Direction::PosX => [0, y, z],
        Direction::NegY => [x, CHUNK_SIZE_USIZE - 1, z],
        Direction::PosY => [x, 0, z],
        Direction::NegZ => [x, y, CHUNK_SIZE_USIZE - 1],
        Direction::PosZ => [x, y, 0],
    };
    face_visible_against(
        state,
        block_at(neighbor.blocks(), neighbor_coord),
        opacity,
        visibility,
    )
}

fn face_visible_against(
    state: BlockState,
    neighbor: BlockState,
    opacity: &[bool],
    visibility: &[bool],
) -> bool {
    if !is_visible(neighbor, visibility) {
        return true;
    }
    if state.id == neighbor.id {
        return false;
    }
    let state_opaque = is_opaque(state, opacity);
    let neighbor_opaque = is_opaque(neighbor, opacity);
    if state_opaque != neighbor_opaque {
        return true;
    }
    if !state_opaque && !neighbor_opaque {
        return state.id.0 < neighbor.id.0;
    }
    false
}

fn face_ao(
    blocks: &[BlockState],
    neighbors: ChunkNeighbors<'_>,
    opacity: &[bool],
    coord: [usize; 3],
    direction: Direction,
) -> u8 {
    let axis = normal_axis(direction);
    let normal = direction.normal();
    let mut samples = 0;
    let mut occluders = 0;
    for delta_v in [-1isize, 1] {
        for delta_u in [-1isize, 1] {
            let mut sample = [coord[0] as isize, coord[1] as isize, coord[2] as isize];
            if direction == Direction::PosY {
                sample[axis] += normal[axis] as isize;
            }
            let u_axis = (axis + 1) % 3;
            let v_axis = (axis + 2) % 3;
            sample[u_axis] += delta_u;
            sample[v_axis] += delta_v;
            let Some(sample) =
                sample_ao_block(blocks, neighbors, sample, direction != Direction::PosY)
            else {
                continue;
            };
            samples += 1;
            if sample.is_occluding(opacity) {
                occluders += 1;
            }
        }
    }
    if samples == 0 {
        0
    } else {
        ((occluders * 3) / samples) as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AoSample {
    Block(BlockState),
    Occluder,
}

impl AoSample {
    fn is_occluding(self, opacity: &[bool]) -> bool {
        match self {
            Self::Block(state) => is_opaque(state, opacity),
            Self::Occluder => true,
        }
    }
}

fn sample_ao_block(
    blocks: &[BlockState],
    neighbors: ChunkNeighbors<'_>,
    sample: [isize; 3],
    missing_corner_occludes: bool,
) -> Option<AoSample> {
    let mut local = [0usize; 3];
    let mut out_axis = None;
    for axis in 0..3 {
        if (0..CHUNK_SIZE_USIZE as isize).contains(&sample[axis]) {
            local[axis] = sample[axis] as usize;
            continue;
        }
        if sample[axis] != -1 && sample[axis] != CHUNK_SIZE_USIZE as isize {
            return None;
        }
        if out_axis.replace(axis).is_some() {
            return missing_corner_occludes.then_some(AoSample::Occluder);
        }
        local[axis] = if sample[axis] < 0 {
            CHUNK_SIZE_USIZE - 1
        } else {
            0
        };
    }

    match out_axis {
        Some(axis) => neighbors
            .get(direction_for_axis(axis, sample[axis] > 0))
            .map(|neighbor| AoSample::Block(block_at(neighbor.blocks(), local))),
        None => Some(AoSample::Block(block_at(blocks, local))),
    }
}

const fn normal_axis(direction: Direction) -> usize {
    match direction {
        Direction::NegX | Direction::PosX => 0,
        Direction::NegY | Direction::PosY => 1,
        Direction::NegZ | Direction::PosZ => 2,
    }
}

const fn direction_for_axis(axis: usize, positive: bool) -> Direction {
    match (axis, positive) {
        (0, false) => Direction::NegX,
        (0, true) => Direction::PosX,
        (1, false) => Direction::NegY,
        (1, true) => Direction::PosY,
        (2, false) => Direction::NegZ,
        (2, true) => Direction::PosZ,
        _ => unreachable!(),
    }
}

const fn is_positive(direction: Direction) -> bool {
    matches!(
        direction,
        Direction::PosX | Direction::PosY | Direction::PosZ
    )
}

fn should_flip_winding(direction: Direction, u_axis: usize, v_axis: usize) -> bool {
    let cross = match (u_axis, v_axis) {
        (0, 1) => [0, 0, 1],
        (1, 2) => [1, 0, 0],
        (2, 0) => [0, 1, 0],
        (1, 0) => [0, 0, -1],
        (2, 1) => [-1, 0, 0],
        (0, 2) => [0, -1, 0],
        _ => [0, 0, 0],
    };
    cross != direction.normal()
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::{BlockState, LocalVoxelCoord};
    use voxel_world::Chunk;

    fn solid_chunk(coord: ChunkCoord) -> Chunk {
        let mut chunk = Chunk::empty(coord);
        for index in 0..voxel_core::CHUNK_VOLUME {
            chunk.set_block(
                LocalVoxelCoord::from_index(index).unwrap(),
                BlockState::new(BlockId::STONE),
            );
        }
        chunk
    }

    #[test]
    fn empty_chunk_emits_no_quads() {
        let chunk = Chunk::empty(ChunkCoord::ZERO);
        let mesh = mesh_chunk_greedy(&chunk, &BlockRegistry::default());
        assert_eq!(mesh.opaque_quad_count(), 0);
    }

    #[test]
    fn solid_chunk_greedy_merges_to_six_quads() {
        let chunk = solid_chunk(ChunkCoord::ZERO);
        let mesh = mesh_chunk_greedy(&chunk, &BlockRegistry::default());
        assert_eq!(mesh.opaque_quad_count(), 6);
    }

    #[test]
    fn solid_neighbor_culls_shared_boundary_face() {
        let chunk = solid_chunk(ChunkCoord::ZERO);
        let pos_x = solid_chunk(ChunkCoord::new(1, 0, 0));
        let mesh = mesh_chunk_greedy_with_neighbors(
            &chunk,
            &BlockRegistry::default(),
            ChunkNeighbors {
                pos_x: Some(&pos_x),
                ..ChunkNeighbors::default()
            },
        );
        assert_eq!(mesh.opaque_quad_count(), 5);
    }

    #[test]
    fn missing_neighbor_keeps_boundary_face_visible() {
        let chunk = solid_chunk(ChunkCoord::ZERO);
        let isolated = mesh_chunk_greedy(&chunk, &BlockRegistry::default());
        let with_missing = mesh_chunk_greedy_with_neighbors(
            &chunk,
            &BlockRegistry::default(),
            ChunkNeighbors::default(),
        );
        assert_eq!(
            with_missing.opaque_quad_count(),
            isolated.opaque_quad_count()
        );
    }

    #[test]
    fn air_neighbor_keeps_boundary_face_visible() {
        let chunk = solid_chunk(ChunkCoord::ZERO);
        let pos_x = Chunk::empty(ChunkCoord::new(1, 0, 0));
        let mesh = mesh_chunk_greedy_with_neighbors(
            &chunk,
            &BlockRegistry::default(),
            ChunkNeighbors {
                pos_x: Some(&pos_x),
                ..ChunkNeighbors::default()
            },
        );
        assert!(mesh.opaque_quad_count() >= 6);
        assert!(
            mesh.opaque_surfaces[0]
                .vertices
                .iter()
                .any(|vertex| vertex.normal == [1.0, 0.0, 0.0])
        );
    }

    #[test]
    fn water_block_emits_visible_faces_while_non_opaque() {
        let registry = BlockRegistry::default();
        assert!(!registry.is_opaque(BlockState::new(BlockId::WATER)));

        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(1, 1, 1),
            BlockState::new(BlockId::WATER),
        );
        let mesh = mesh_chunk_greedy(&chunk, &registry);
        assert_eq!(mesh.opaque_quad_count(), 0);
        assert_eq!(mesh.transparent_quad_count(), 6);
    }

    #[test]
    fn leaves_emit_visible_faces_while_non_opaque() {
        let registry = BlockRegistry::default();
        assert!(!registry.is_opaque(BlockState::new(BlockId::LEAVES)));

        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(1, 1, 1),
            BlockState::new(BlockId::LEAVES),
        );
        let mesh = mesh_chunk_greedy(&chunk, &registry);
        assert_eq!(mesh.opaque_quad_count(), 0);
        assert_eq!(mesh.transparent_quad_count(), 6);
    }

    #[test]
    fn adjacent_water_blocks_do_not_emit_internal_faces() {
        let registry = BlockRegistry::default();
        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(1, 1, 1),
            BlockState::new(BlockId::WATER),
        );
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(2, 1, 1),
            BlockState::new(BlockId::WATER),
        );

        let mesh = mesh_chunk_greedy(&chunk, &registry);
        assert_eq!(mesh.opaque_quad_count(), 0);
        assert_eq!(mesh.transparent_quad_count(), 6);
    }

    #[test]
    fn edge_ao_samples_cardinal_neighbor_chunks() {
        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(31, 1, 31),
            BlockState::new(BlockId::STONE),
        );
        let mut pos_z = Chunk::empty(ChunkCoord::new(0, 0, 1));
        pos_z.set_block(
            LocalVoxelCoord::new_unchecked(31, 0, 0),
            BlockState::new(BlockId::STONE),
        );
        pos_z.set_block(
            LocalVoxelCoord::new_unchecked(31, 2, 0),
            BlockState::new(BlockId::STONE),
        );

        let mesh = mesh_chunk_greedy_with_neighbors(
            &chunk,
            &BlockRegistry::default(),
            ChunkNeighbors {
                pos_z: Some(&pos_z),
                ..ChunkNeighbors::default()
            },
        );

        assert!(
            mesh.opaque_surfaces[0]
                .vertices
                .iter()
                .any(|vertex| vertex.normal == [1.0, 0.0, 0.0] && vertex.ao == 1)
        );
    }

    #[test]
    fn corner_ao_counts_two_axis_boundary_sample_as_occluded() {
        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        let edge = (CHUNK_SIZE_USIZE - 1) as u8;
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(edge, edge, edge),
            BlockState::new(BlockId::STONE),
        );
        let opacity = opacity_table(&BlockRegistry::default());

        assert_eq!(
            face_ao(
                chunk.blocks(),
                ChunkNeighbors::default(),
                &opacity,
                [
                    CHUNK_SIZE_USIZE - 1,
                    CHUNK_SIZE_USIZE - 1,
                    CHUNK_SIZE_USIZE - 1
                ],
                Direction::PosX,
            ),
            1
        );
    }

    #[test]
    fn top_face_ao_ignores_same_level_neighbors() {
        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(1, 1, 1),
            BlockState::new(BlockId::STONE),
        );
        for z in 0..=2 {
            for x in 0..=2 {
                if x == 1 && z == 1 {
                    continue;
                }
                chunk.set_block(
                    LocalVoxelCoord::new_unchecked(x, 1, z),
                    BlockState::new(BlockId::STONE),
                );
            }
        }

        let opacity = opacity_table(&BlockRegistry::default());
        assert_eq!(
            face_ao(
                chunk.blocks(),
                ChunkNeighbors::default(),
                &opacity,
                [1, 1, 1],
                Direction::PosY,
            ),
            0
        );
    }

    #[test]
    fn top_face_ao_samples_blocks_above_the_face() {
        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(1, 1, 1),
            BlockState::new(BlockId::STONE),
        );
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(0, 2, 0),
            BlockState::new(BlockId::STONE),
        );
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(2, 2, 2),
            BlockState::new(BlockId::STONE),
        );

        let opacity = opacity_table(&BlockRegistry::default());
        assert_eq!(
            face_ao(
                chunk.blocks(),
                ChunkNeighbors::default(),
                &opacity,
                [1, 1, 1],
                Direction::PosY,
            ),
            1
        );
    }

    #[test]
    fn single_block_emits_six_quads() {
        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        chunk.set_block(
            LocalVoxelCoord::new_unchecked(1, 1, 1),
            BlockState::new(BlockId::STONE),
        );
        let mesh = mesh_chunk_greedy(&chunk, &BlockRegistry::default());
        assert_eq!(mesh.opaque_quad_count(), 6);
    }
}
