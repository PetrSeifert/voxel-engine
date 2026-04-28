# chunk_bench

Runs generation and CPU greedy-meshing benchmarks over a fixed chunk grid.

```powershell
cargo run -p chunk_bench
```

By default the bench runs in neighbor-aware mode, so chunk meshes can consult
adjacent chunks and avoid emitting hidden boundary faces.

To benchmark isolated chunk meshing, pass `--isolated` to the bench binary after
Cargo's argument separator:

```powershell
cargo run -p chunk_bench -- --isolated
```

The flag is read by `chunk_bench` itself. Passing it before the `--` separator
hands it to Cargo instead of the binary.
