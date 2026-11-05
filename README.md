# Numerov 1D Eigensolver

An implementation of the Numerov 1D eigensolver in Rust and Python. The Rust implementation runs approx. 100x faster. Pass an energy range to find all eigenvalues and eigenfunctions within it.

## Python

```sh
conda env create -f environment.yml
python numerov.py lower_E upper_E
```

## Rust

```sh
cd numerov
cargo run lower_E upper_E
```
