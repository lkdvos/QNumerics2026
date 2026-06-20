# Tensor Network Methods — QNumerics 2026

Teaching material for the **Tensor Network Methods** session of the [QNumerics 2026](https://qnumerics.org) summer school on quantum simulation (Amherst).
The session runs as two 3-hour blocks and builds up three ground-state methods — TEBD, DMRG, and symmetric DMRG — bottom-up on the [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) ecosystem.

## Contents

| Path | What it is |
|------|------------|
| [`presentations/`](presentations/) | Introductory slides (`01_TensorNetworks.key`) |
| [`notebooks/lowrank_compression.jl`](notebooks/lowrank_compression.jl) | Low-rank / SVD image-compression demo (motivation) |
| [`notebooks/tensor_primitives/`](notebooks/tensor_primitives/) | TensorKit spaces, `TensorMap`s, `@tensor` contraction, truncated factorizations, the dense `E₀` |
| [`notebooks/mps_tebd/`](notebooks/mps_tebd/) | **Hands-on I** — MPS + TEBD (`apply_gate`, `svd_truncate`, `canonicalize`) |
| [`notebooks/dmrg/`](notebooks/dmrg/) | **Hands-on II** — DMRG (MPO, environments, `H_eff`, sweeps) |
| [`notebooks/symmetries.jl`](notebooks/symmetries.jl) | **Hands-on/illustration III** — symmetric DMRG (Z₂ / SU(2) / fermions) |

Each hands-on folder ships two notebooks:

- `<name>.jl` — the **notebook** with deliberate `TODO` gaps to fill in.
- `<name>_solution.jl` — the **completed reference**.

The `.svg` files next to each notebook are the tensor-network diagrams it embeds.

## Running the notebooks (Pluto)

The notebooks are [Pluto.jl](https://plutojl.org) notebooks. 
Each one **embeds its own package environment** inside the file, so Pluto installs the right package versions automatically the first time you open it — there's no separate `instantiate` step for the notebooks.

### 1. Install Julia

Use [juliaup](https://github.com/JuliaLang/juliaup) (this material targets Julia **1.12.x**, but should be compatible with **1.10+**):

```bash
curl -fsSL https://install.julialang.org | sh   # macOS / Linux
# Windows: winget install julia -s msstore
```

### 2. Install Pluto

In a Julia REPL, install Pluto into your environment:

```julia
import Pkg; Pkg.add("Pluto")
```

### 3. Launch Pluto and open a notebook

```julia
import Pluto; Pluto.run()
```

This opens Pluto in your browser.
Paste the path to a notebook into the **"Open a notebook"** box.
This can be a path to a local file if you cloned this repository, or a direct link to one of the notebooks.

On first open, Pluto restores the notebook's embedded environment (this can take a few minutes the first time as packages precompile).
After that, cells run reactively — change a cell and everything downstream re-runs.

See the [Pluto README](https://github.com/fonsp/Pluto.jl#readme) and [plutojl.org](https://plutojl.org) for more, including keyboard shortcuts and how to work with the reactive runtime.

> **Tip:** Start with `tensor_primitives`, then `mps_tebd`, `dmrg`, and finally `symmetries` — they build on each other in that order.
> Reach for a `*_solution.jl` notebook only after attempting the gaps in the student version.
