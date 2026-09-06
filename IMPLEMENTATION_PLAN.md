# Implementation Plan — High-Throughput Distributed Vector Search Engine

## Overview

A high-performance, distributed, lightweight vector database built from first principles in **Modern C++20** with AVX2/FMA SIMD acceleration, HNSW graph indexing, WAL persistence, and consistent-hash distributed sharding.

---

## Phase 1 — Core Index Engine & Hardware Acceleration (Status: Working Baseline)

- [x] **SIMD Distance Kernels** (`include/vectordb/engine/distance.hpp`, `src/engine/distance.cpp`)
  - AVX2 + FMA vectorized L2, Cosine, and Dot Product distance computations.
  - Scalar fallback for non-AVX2 architectures with runtime CPU feature dispatch.
  - Sub-60ns distance compute for 768-d embeddings (8.2x speedup).
- [x] **HNSW Graph Index** (`include/vectordb/engine/hnsw_index.hpp`, `src/engine/hnsw_index.cpp`)
  - Multi-layer graph topology with random exponential level assignment.
  - Greedy upper-layer entry point search down to target insertion layer.
  - Priority queue based dynamic candidate exploration (`search_layer`).
  - Bidirectional connection establishment and neighbor pruning.
  - Fine-grained concurrent reads and writes (`std::shared_mutex`).
- [x] **Benchmarking & Unit Testing** (`bench/bench_distance.cpp`, `tests/test_hnsw.cpp`)

---

## Phase 2 — Persistence, Durability & Memory Management

- [x] **Write-Ahead Log (WAL)** (`include/vectordb/storage/wal.hpp`, `src/storage/wal.cpp`)
  - Binary append-only logging with CRC32 payload verification.
  - Crash recovery and startup replay mechanism.
- [x] **POSIX Mmap Storage** (`include/vectordb/storage/mmap_storage.hpp`)
  - Direct kernel page cache integration via `mmap(2)` and `madvise(MADV_RANDOM)`.
- [ ] **LSM-Style Segment Compaction**
  - Background compaction daemon merging immutable segment files and garbage collecting tombstones.

---

## Phase 3 — Distributed Clustering & Consensus

- [x] **Consistent Hash Ring** (`include/vectordb/cluster/consistent_hash.hpp`, `src/cluster/consistent_hash.cpp`)
  - Virtual node token ring (150 vnodes per instance) for balanced key distribution.
  - Automatic replication routing across $R$ replicas.
- [x] **Raft State Machine Baseline** (`include/vectordb/cluster/raft_node.hpp`)
  - Leader election, term tracking, and cluster metadata consensus.
- [ ] **Inter-Node P2P RPC & Gossip**
  - SWIM protocol failure detection and cluster health heartbeats.

---

## Phase 4 — High-Concurrency Server & Client SDK

- [x] **Multi-Threaded Server Engine** (`include/vectordb/server/tcp_server.hpp`, `include/vectordb/common/thread_pool.hpp`)
  - POSIX socket server with worker thread pool.
- [x] **Python Client SDK** (`sdk/python/vectordb/client.py`)
  - Python wrapper and socket connection manager.
- [ ] **Binary Wire Protocol / gRPC Gateway**

---

## Phase 5 — Deployment & Observability

- [x] **Multi-Stage Containerization** (`deploy/Dockerfile`)
  - Minimal container with GCC 13 multi-stage build.
- [x] **Kubernetes Deployment** (`deploy/k8s/statefulset.yaml`)
  - StatefulSet with persistent volume claims (PVC) and health probes.
- [ ] **Prometheus Metrics Exporter**
