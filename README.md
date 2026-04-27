# Voxel Engine

Rust/Vulkan voxel engine workspace targeting Windows and Linux desktops.

## Current Vertical Slice

- `voxel-core`: block IDs, chunk/region/voxel coordinates, bounds, constants.
- `voxel-world`: deterministic terrain generation, chunk storage, edit logs, skylight, streaming states.
- `voxel-mesh`: CPU greedy meshing for opaque block chunks.
- `voxel-render`: renderer-facing scene/backend interfaces and a null backend.
- `voxel-vulkan`: Vulkan 1.3 backend policy and `ash`/`vk-mem` integration scaffold.
- `voxel-runtime`: explicit generation/meshing/upload/render stages.
- `apps/sandbox`: runnable headless sandbox loop.
- `apps/chunk_bench`: generation and meshing benchmark.
- `apps/shader_lab`: Vulkan backend capability probe scaffold.

## Useful Commands

```powershell
cargo test
cargo run -p sandbox
cargo run -p chunk_bench
cargo run -p shader_lab
```
