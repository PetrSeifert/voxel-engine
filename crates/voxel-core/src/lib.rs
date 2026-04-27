use serde::{Deserialize, Serialize};
use std::fmt;

pub const CHUNK_SIZE: i32 = 32;
pub const CHUNK_SIZE_U32: u32 = CHUNK_SIZE as u32;
pub const CHUNK_SIZE_USIZE: usize = CHUNK_SIZE as usize;
pub const CHUNK_AREA: usize = CHUNK_SIZE_USIZE * CHUNK_SIZE_USIZE;
pub const CHUNK_VOLUME: usize = CHUNK_AREA * CHUNK_SIZE_USIZE;
pub const REGION_SIZE_CHUNKS: i32 = 32;
pub const REGION_SIZE_CHUNKS_USIZE: usize = REGION_SIZE_CHUNKS as usize;

#[derive(
    Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct BlockId(pub u16);

impl BlockId {
    pub const AIR: Self = Self(0);
    pub const STONE: Self = Self(1);
    pub const DIRT: Self = Self(2);
    pub const GRASS: Self = Self(3);
    pub const WATER: Self = Self(4);

    pub const fn is_air(self) -> bool {
        self.0 == Self::AIR.0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct BlockState {
    pub id: BlockId,
    pub metadata: u16,
}

impl BlockState {
    pub const AIR: Self = Self {
        id: BlockId::AIR,
        metadata: 0,
    };

    pub const fn new(id: BlockId) -> Self {
        Self { id, metadata: 0 }
    }

    pub const fn with_metadata(id: BlockId, metadata: u16) -> Self {
        Self { id, metadata }
    }

    pub const fn is_air(self) -> bool {
        self.id.is_air()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Direction {
    NegX,
    PosX,
    NegY,
    PosY,
    NegZ,
    PosZ,
}

impl Direction {
    pub const ALL: [Self; 6] = [
        Self::NegX,
        Self::PosX,
        Self::NegY,
        Self::PosY,
        Self::NegZ,
        Self::PosZ,
    ];

    pub const fn normal(self) -> [i32; 3] {
        match self {
            Self::NegX => [-1, 0, 0],
            Self::PosX => [1, 0, 0],
            Self::NegY => [0, -1, 0],
            Self::PosY => [0, 1, 0],
            Self::NegZ => [0, 0, -1],
            Self::PosZ => [0, 0, 1],
        }
    }

    pub const fn opposite(self) -> Self {
        match self {
            Self::NegX => Self::PosX,
            Self::PosX => Self::NegX,
            Self::NegY => Self::PosY,
            Self::PosY => Self::NegY,
            Self::NegZ => Self::PosZ,
            Self::PosZ => Self::NegZ,
        }
    }
}

#[derive(Clone, Copy, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    pub fn region_coord(self) -> RegionCoord {
        RegionCoord::new(
            self.x.div_euclid(REGION_SIZE_CHUNKS),
            self.y.div_euclid(REGION_SIZE_CHUNKS),
            self.z.div_euclid(REGION_SIZE_CHUNKS),
        )
    }

    pub fn local_region_coord(self) -> [u8; 3] {
        [
            self.x.rem_euclid(REGION_SIZE_CHUNKS) as u8,
            self.y.rem_euclid(REGION_SIZE_CHUNKS) as u8,
            self.z.rem_euclid(REGION_SIZE_CHUNKS) as u8,
        ]
    }

    pub fn min_voxel(self) -> VoxelCoord {
        VoxelCoord::new(
            self.x * CHUNK_SIZE,
            self.y * CHUNK_SIZE,
            self.z * CHUNK_SIZE,
        )
    }

    pub fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Self::new(self.x + dx, self.y + dy, self.z + dz)
    }

    pub fn manhattan_distance(self, other: Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()
    }
}

impl fmt::Debug for ChunkCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chunk({}, {}, {})", self.x, self.y, self.z)
    }
}

#[derive(Clone, Copy, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct RegionCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl RegionCoord {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

impl fmt::Debug for RegionCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "region({}, {}, {})", self.x, self.y, self.z)
    }
}

#[derive(Clone, Copy, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct VoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl VoxelCoord {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    pub fn split_chunk_local(self) -> (ChunkCoord, LocalVoxelCoord) {
        let chunk = ChunkCoord::new(
            self.x.div_euclid(CHUNK_SIZE),
            self.y.div_euclid(CHUNK_SIZE),
            self.z.div_euclid(CHUNK_SIZE),
        );
        let local = LocalVoxelCoord::new_unchecked(
            self.x.rem_euclid(CHUNK_SIZE) as u8,
            self.y.rem_euclid(CHUNK_SIZE) as u8,
            self.z.rem_euclid(CHUNK_SIZE) as u8,
        );
        (chunk, local)
    }
}

impl fmt::Debug for VoxelCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "voxel({}, {}, {})", self.x, self.y, self.z)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct LocalVoxelCoord {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl LocalVoxelCoord {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    pub fn new(x: u8, y: u8, z: u8) -> Result<Self, CoreError> {
        let coord = Self { x, y, z };
        coord.validate()?;
        Ok(coord)
    }

    pub const fn new_unchecked(x: u8, y: u8, z: u8) -> Self {
        Self { x, y, z }
    }

    pub fn from_index(index: usize) -> Result<Self, CoreError> {
        if index >= CHUNK_VOLUME {
            return Err(CoreError::ChunkIndexOutOfRange { index });
        }
        let y = index / CHUNK_AREA;
        let rem = index % CHUNK_AREA;
        let z = rem / CHUNK_SIZE_USIZE;
        let x = rem % CHUNK_SIZE_USIZE;
        Ok(Self::new_unchecked(x as u8, y as u8, z as u8))
    }

    pub fn index(self) -> usize {
        (self.y as usize * CHUNK_AREA) + (self.z as usize * CHUNK_SIZE_USIZE) + self.x as usize
    }

    pub fn validate(self) -> Result<(), CoreError> {
        if self.x as i32 >= CHUNK_SIZE || self.y as i32 >= CHUNK_SIZE || self.z as i32 >= CHUNK_SIZE
        {
            return Err(CoreError::LocalVoxelOutOfRange { coord: self });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AabbI32 {
    pub min: VoxelCoord,
    pub max_exclusive: VoxelCoord,
}

impl AabbI32 {
    pub const fn new(min: VoxelCoord, max_exclusive: VoxelCoord) -> Self {
        Self { min, max_exclusive }
    }

    pub fn chunk_bounds(chunk: ChunkCoord) -> Self {
        let min = chunk.min_voxel();
        let max_exclusive =
            VoxelCoord::new(min.x + CHUNK_SIZE, min.y + CHUNK_SIZE, min.z + CHUNK_SIZE);
        Self::new(min, max_exclusive)
    }

    pub fn contains(self, coord: VoxelCoord) -> bool {
        coord.x >= self.min.x
            && coord.y >= self.min.y
            && coord.z >= self.min.z
            && coord.x < self.max_exclusive.x
            && coord.y < self.max_exclusive.y
            && coord.z < self.max_exclusive.z
    }
}

#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum CoreError {
    #[error("local voxel coordinate out of range for chunk: {coord:?}")]
    LocalVoxelOutOfRange { coord: LocalVoxelCoord },
    #[error("chunk index out of range: {index}")]
    ChunkIndexOutOfRange { index: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_voxel_coordinates_split_with_euclidean_rules() {
        let (chunk, local) = VoxelCoord::new(-1, -33, 32).split_chunk_local();
        assert_eq!(chunk, ChunkCoord::new(-1, -2, 1));
        assert_eq!(local, LocalVoxelCoord::new_unchecked(31, 31, 0));
    }

    #[test]
    fn local_index_round_trips() {
        for index in [0, 1, 31, 32, 1024, CHUNK_VOLUME - 1] {
            let coord = LocalVoxelCoord::from_index(index).unwrap();
            assert_eq!(coord.index(), index);
        }
    }

    #[test]
    fn negative_chunk_region_mapping_uses_floor_division() {
        let chunk = ChunkCoord::new(-1, -32, -33);
        assert_eq!(chunk.region_coord(), RegionCoord::new(-1, -1, -2));
        assert_eq!(chunk.local_region_coord(), [31, 0, 31]);
    }
}
