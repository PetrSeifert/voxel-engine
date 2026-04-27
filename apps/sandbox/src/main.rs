use voxel_runtime::{EngineRuntime, RuntimeConfig, install_tracing};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    install_tracing();

    let mut runtime = EngineRuntime::new_headless(RuntimeConfig::default());
    println!(
        "sandbox starting with stages: {:?}",
        runtime.task_graph().stages()
    );

    for frame in 0..4 {
        let stats = runtime.tick()?;
        println!(
            "frame={frame} resident_chunks={} visible_chunks={} upload_bytes={} mesh_queue_depth={}",
            stats.resident_chunks, stats.visible_chunks, stats.upload_bytes, stats.mesh_queue_depth
        );
    }

    Ok(())
}
