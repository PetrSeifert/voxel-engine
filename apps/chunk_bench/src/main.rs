use std::time::Instant;
use std::{collections::HashMap, env};
use voxel_core::ChunkCoord;
use voxel_mesh::{ChunkMesh, ChunkNeighbors, mesh_chunk_greedy, mesh_chunk_greedy_with_neighbors};
use voxel_world::{BlockRegistry, Chunk, TerrainGenerator, WorldGenerator};

fn main() {
    let generator = TerrainGenerator::new(0x5eed);
    let registry = BlockRegistry::default();
    let isolated = env::args().any(|arg| arg == "--isolated");
    let mut chunks = Vec::new();
    let mut chunk_indices = HashMap::new();

    let generation_start = Instant::now();
    for z in -8..=8 {
        for x in -8..=8 {
            let coord = ChunkCoord::new(x, 0, z);
            chunk_indices.insert(coord, chunks.len());
            chunks.push(generator.generate_chunk(coord));
        }
    }
    let generation_time = generation_start.elapsed();

    let mesh_start = Instant::now();
    let meshes: Vec<_> = if isolated {
        chunks
            .iter()
            .map(|chunk| mesh_chunk_greedy(chunk, &registry))
            .collect()
    } else {
        chunks
            .iter()
            .map(|chunk| {
                mesh_chunk_greedy_with_neighbors(
                    chunk,
                    &registry,
                    neighbors_for(chunk.coord(), &chunks, &chunk_indices),
                )
            })
            .collect()
    };
    let mesh_time = mesh_start.elapsed();

    let quads: usize = meshes.iter().map(|mesh| mesh.opaque_quad_count()).sum();
    let upload_bytes: u64 = meshes.iter().map(mesh_upload_bytes).sum();
    println!(
        "mode={} chunks={} generation_ms={:.3} mesh_ms={:.3} opaque_quads={} upload_bytes={}",
        if isolated {
            "isolated"
        } else {
            "neighbor-aware"
        },
        chunks.len(),
        generation_time.as_secs_f64() * 1000.0,
        mesh_time.as_secs_f64() * 1000.0,
        quads,
        upload_bytes
    );
}

fn neighbors_for<'a>(
    coord: ChunkCoord,
    chunks: &'a [Chunk],
    chunk_indices: &HashMap<ChunkCoord, usize>,
) -> ChunkNeighbors<'a> {
    let get = |coord| chunk_indices.get(&coord).map(|&index| &chunks[index]);
    ChunkNeighbors {
        neg_x: get(coord.offset(-1, 0, 0)),
        pos_x: get(coord.offset(1, 0, 0)),
        neg_y: get(coord.offset(0, -1, 0)),
        pos_y: get(coord.offset(0, 1, 0)),
        neg_z: get(coord.offset(0, 0, -1)),
        pos_z: get(coord.offset(0, 0, 1)),
    }
}

fn mesh_upload_bytes(mesh: &ChunkMesh) -> u64 {
    mesh.opaque_surfaces
        .iter()
        .chain(mesh.transparent_surfaces.iter())
        .map(|surface| {
            surface.vertices.len() * std::mem::size_of::<voxel_mesh::MeshVertex>()
                + surface.indices.len() * std::mem::size_of::<u32>()
        })
        .sum::<usize>() as u64
}
