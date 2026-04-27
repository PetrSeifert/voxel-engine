use std::time::Instant;
use voxel_core::ChunkCoord;
use voxel_mesh::mesh_chunk_greedy;
use voxel_world::{BlockRegistry, TerrainGenerator, WorldGenerator};

fn main() {
    let generator = TerrainGenerator::new(0x5eed);
    let registry = BlockRegistry::default();
    let mut chunks = Vec::new();

    let generation_start = Instant::now();
    for z in -2..=2 {
        for x in -2..=2 {
            chunks.push(generator.generate_chunk(ChunkCoord::new(x, 0, z)));
        }
    }
    let generation_time = generation_start.elapsed();

    let mesh_start = Instant::now();
    let meshes: Vec<_> = chunks
        .iter()
        .map(|chunk| mesh_chunk_greedy(chunk, &registry))
        .collect();
    let mesh_time = mesh_start.elapsed();

    let quads: usize = meshes.iter().map(|mesh| mesh.opaque_quad_count()).sum();
    println!(
        "chunks={} generation_ms={:.3} mesh_ms={:.3} opaque_quads={}",
        chunks.len(),
        generation_time.as_secs_f64() * 1000.0,
        mesh_time.as_secs_f64() * 1000.0,
        quads
    );
}
