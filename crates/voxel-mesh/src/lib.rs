use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use voxel_core::{AabbI32, BlockId, CHUNK_SIZE_USIZE, ChunkCoord, Direction, LocalVoxelCoord};
use voxel_world::{BlockRegistry, Chunk};

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

impl ChunkMesh {
    pub fn empty(chunk: ChunkCoord, version: u64) -> Self {
        Self {
            opaque_surfaces: vec![MeshSurface::default()],
            transparent_surfaces: Vec::new(),
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
}

pub fn mesh_chunk_greedy(chunk: &Chunk, registry: &BlockRegistry) -> ChunkMesh {
    let mut mesh = ChunkMesh::empty(chunk.coord(), chunk.version());
    let surface = mesh
        .opaque_surfaces
        .first_mut()
        .expect("empty mesh creates one opaque surface");

    for direction in Direction::ALL {
        mesh_direction(chunk, registry, direction, surface);
    }

    mesh
}

fn mesh_direction(
    chunk: &Chunk,
    registry: &BlockRegistry,
    direction: Direction,
    surface: &mut MeshSurface,
) {
    let axis = normal_axis(direction);
    let u_axis = (axis + 1) % 3;
    let v_axis = (axis + 2) % 3;

    for slice in 0..CHUNK_SIZE_USIZE {
        let mut mask = vec![None; CHUNK_SIZE_USIZE * CHUNK_SIZE_USIZE];
        for v in 0..CHUNK_SIZE_USIZE {
            for u in 0..CHUNK_SIZE_USIZE {
                let mut coord = [0usize; 3];
                coord[axis] = slice;
                coord[u_axis] = u;
                coord[v_axis] = v;

                let local =
                    LocalVoxelCoord::new_unchecked(coord[0] as u8, coord[1] as u8, coord[2] as u8);
                let state = chunk.block(local);
                if !registry.is_opaque(state) {
                    continue;
                }

                if face_is_visible(chunk, registry, coord, direction) {
                    let ao = face_ao(chunk, registry, coord, direction);
                    mask[v * CHUNK_SIZE_USIZE + u] = Some(FaceKey {
                        material: state.id,
                        direction,
                        ao,
                    });
                }
            }
        }

        emit_greedy_mask(surface, &mut mask, axis, u_axis, v_axis, slice, direction);
    }
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
    chunk: &Chunk,
    registry: &BlockRegistry,
    coord: [usize; 3],
    direction: Direction,
) -> bool {
    let axis = normal_axis(direction);
    let neighbor_axis = if is_positive(direction) {
        coord[axis].checked_add(1)
    } else {
        coord[axis].checked_sub(1)
    };

    let Some(neighbor_axis) = neighbor_axis else {
        return true;
    };
    if neighbor_axis >= CHUNK_SIZE_USIZE {
        return true;
    }

    let mut neighbor = coord;
    neighbor[axis] = neighbor_axis;
    let local =
        LocalVoxelCoord::new_unchecked(neighbor[0] as u8, neighbor[1] as u8, neighbor[2] as u8);
    !registry.is_opaque(chunk.block(local))
}

fn face_ao(chunk: &Chunk, registry: &BlockRegistry, coord: [usize; 3], direction: Direction) -> u8 {
    let axis = normal_axis(direction);
    let mut samples = 0;
    let mut occluders = 0;
    for delta_v in [-1isize, 1] {
        for delta_u in [-1isize, 1] {
            let mut sample = coord;
            let u_axis = (axis + 1) % 3;
            let v_axis = (axis + 2) % 3;
            let Some(u) = sample[u_axis].checked_add_signed(delta_u) else {
                continue;
            };
            let Some(v) = sample[v_axis].checked_add_signed(delta_v) else {
                continue;
            };
            if u >= CHUNK_SIZE_USIZE || v >= CHUNK_SIZE_USIZE {
                continue;
            }
            sample[u_axis] = u;
            sample[v_axis] = v;
            let local =
                LocalVoxelCoord::new_unchecked(sample[0] as u8, sample[1] as u8, sample[2] as u8);
            samples += 1;
            if registry.is_opaque(chunk.block(local)) {
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

const fn normal_axis(direction: Direction) -> usize {
    match direction {
        Direction::NegX | Direction::PosX => 0,
        Direction::NegY | Direction::PosY => 1,
        Direction::NegZ | Direction::PosZ => 2,
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
    use voxel_core::BlockState;
    use voxel_world::Chunk;

    #[test]
    fn empty_chunk_emits_no_quads() {
        let chunk = Chunk::empty(ChunkCoord::ZERO);
        let mesh = mesh_chunk_greedy(&chunk, &BlockRegistry::default());
        assert_eq!(mesh.opaque_quad_count(), 0);
    }

    #[test]
    fn solid_chunk_greedy_merges_to_six_quads() {
        let mut chunk = Chunk::empty(ChunkCoord::ZERO);
        for index in 0..voxel_core::CHUNK_VOLUME {
            chunk.set_block(
                LocalVoxelCoord::from_index(index).unwrap(),
                BlockState::new(BlockId::STONE),
            );
        }
        let mesh = mesh_chunk_greedy(&chunk, &BlockRegistry::default());
        assert_eq!(mesh.opaque_quad_count(), 6);
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
