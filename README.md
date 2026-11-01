# mixed-mlx

Mixed-precision matrix square roots on Apple Silicon. This is experiment
code, not a library. It was written to answer one question: how far down the
precision ladder can the eigenbasis, the spectrum, and the correction products
of a fixed-operator Newton iteration go before the method stops converging.

The answer, on an M3: the products can go low, the factors cannot, and the
complete solve is not faster than all-FP32. The measurements are in the
accompanying article.

## Requirements

- macOS 27.0 SDK and Xcode 27 (Metal 4.1). `make` refuses to build otherwise.
- Apple Silicon. Written and measured on a MacBook Air M3 (2024).

## Layout

| Path | Build | State |
|---|---|---|
| `src/` | `make mixed` -> `build/mixed_sqrt` | Current tree. Holds the format table, the GPU-resident solver, the storage codecs, and the reporting that produced the published numbers. |
| `3_spectral/` | `make spectral` -> `build/spectral` | Earlier tree: the fixed list of sixteen precision plans, plus the product benchmark (`make products`). |
| `2_formats/` | `make formats` -> `build/formats` | Encoding and decoding experiments for FP8 and MX formats. |
| `1_baseline/` | `make baseline` -> `build/baseline` | First baseline: Accelerate against Metal. |
| `third_party/metal-cpp` | vendored | Apple's C++ bindings, Apache-2.0. See its own `LICENSE.txt`. |

## Build and run

```sh
make mixed

# Source-level checks, no matrices.
./build/mixed_sqrt --codec-check

# One small controlled problem, one evaluation.
./build/mixed_sqrt --orders 32 --conditions 100 --warmups 2 --evaluations 3 --sweep
```

Run from the repository root: `build/mixed_sqrt` loads `metal_mixed_sqrt.metal`
by relative path and compiles it at start-up.

Main options: `--solver-backend metal|accelerate`, `--basis-precision`,
`--spectrum-precision`, `--correction-mode fp32_reorthogonalized|native_quantized`,
`--native-dispatch fused|per_stage`, `--rounding nearest|stochastic`,
`--orders`, `--conditions`, `--warmups`, `--evaluations`, `--sweep`,
`--codec-check`, `--kernel-check`. There is no `--help`; an unknown option
prints an error.

`--sweep` writes `metal_benchmark/summary.csv` and
`metal_benchmark/iterations.csv` relative to the working directory. The path
is not configurable.

## Known state

- `make check` runs `tests/matrix_codec.cpp` only. The solver, the dispatch
  fallback rules, and the Metal buffer lifetime code have no automated test.
- `./build/mixed_sqrt --kernel-check` currently fails. The failing edges are
  `fp4_e2m1` positive and negative saturation and `log_s` largest finite for
  `fp8_e4m3fn` and `fp4_e2m1`. Do not read the quantized results as validated
  until this passes.
- `artifacts/` and the plotting script inputs are not in the repository. The
  published figures were generated from local CSV output.

## License

MIT for this repository's own code. `third_party/metal-cpp` stays under
Apache-2.0.
