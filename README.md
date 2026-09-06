# High-Throughput Distributed Vector Search Engine

A distributed, lightweight vector database built from scratch in **Modern C++20** with SIMD-accelerated distance kernels (AVX2/FMA) and HNSW graph indexing.

---

## Highlights

- 🏎️ **SIMD Hardware Acceleration**: Hand-crafted AVX2 + FMA vector distance kernels delivering **~8.2x speedup** (~58 ns for 768-d L2).
- ⚡ **HNSW Graph Indexing**: Hierarchical Navigable Small World algorithm with $O(\log N)$ search latency and sub-millisecond p99.
- 🛡️ **Durability & Crash Recovery**: Append-only Write-Ahead Logging (WAL) with hardware-friendly CRC32 verification and POSIX `mmap` integration.
- 🌐 **Distributed Sharding**: Virtual-node consistent hashing ring (150 vnodes/node) with automatic multi-replica routing.
- 🧵 **Multi-Threaded Concurrency**: Reader-writer locks (`std::shared_mutex`) and worker thread pool for high-throughput batch operations.
- 🐍 **Python Client SDK**: Python client interface for easy integration.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                          │
│         TCP Protocol  |  C++ Client  |  Python SDK      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  ROUTER / COORDINATOR                    │
│   Request routing, load balancing, scatter-gather        │
│   Raft-based leader election (metadata consensus)        │
└──────┬────────────────────────────┬─────────────────────┘
       │                            │
┌──────▼──────┐              ┌──────▼──────┐
│  SHARD  0   │   . . . .    │  SHARD  N   │
│  ─────────  │              │  ─────────  │
│  HNSW Index │              │  HNSW Index │
│  IVF Index  │              │  IVF Index  │
│  WAL Log    │              │  WAL Log    │
│  Mmap MMF   │              │  Mmap MMF   │
└─────────────┘              └─────────────┘
       │                            │
┌──────▼────────────────────────────▼─────┐
│           STORAGE LAYER                  │
│  Segment files | WAL | Bloom Filters     │
│  Mmap manager | Compaction daemon        │
└──────────────────────────────────────────┘
```

---

## Performance Benchmarks

Measured on Linux x86_64 (768-d embeddings, 1,000,000 iterations):

| Metric | Scalar Baseline | AVX2 + FMA SIMD | Speedup |
| :--- | :--- | :--- | :--- |
| **L2 Distance (768-d)** | 477.6 ns / op | **57.9 ns / op** | **8.24x** |
| **Dot Product (768-d)** | 481.2 ns / op | **58.2 ns / op** | **8.26x** |

---

## Directory Structure

```
.
├── include/vectordb/
│   ├── common/              # Types, configuration, thread pool
│   ├── engine/              # Distance kernels, HNSW index, IVF index, filters
│   ├── storage/             # WAL, mmap file manager, segment files
│   ├── cluster/             # Consistent hashing ring, Raft consensus
│   └── server/              # Network socket server, protocol
├── src/
│   ├── engine/              # Distance AVX2 kernels & HNSW graph implementation
│   ├── storage/             # WAL implementation
│   ├── cluster/             # Consistent hash ring implementation
│   └── server/              # Main server binary
├── tests/                   # Unit test suite (HNSW correctness, distance kernels)
├── bench/                   # Benchmark suite (SIMD vs scalar benchmarks)
├── sdk/python/              # Python client library
├── deploy/                  # Dockerfile & Kubernetes StatefulSet manifests
├── docs/                    # Architecture & design notes
├── Makefile                 # Fast C++20 build pipeline
├── CMakeLists.txt           # Standard CMake build definition
└── README.md
```

---

## Quick Start

### Build and Run

```bash
# Build the server, run unit tests, and run benchmarks
make all

# Run server directly
./bin/vectordb

# Run distance benchmark suite
make bench

# Run unit tests
make test
```

### Clean

```bash
make clean
```

---

## License

MIT
