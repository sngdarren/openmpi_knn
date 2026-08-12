# Plan: Distributed KNN with MPI — engine.cpp implementation

## Context

`Engine::KNN()` in [engine.cpp](engine.cpp) is a stub. `common.cpp` reads all input on rank 0 only (other ranks get empty vectors and `params={0,0,0}`), then calls `KNN` on all ranks, measures wall-clock time with a barrier. The goal is to parallelize KNN across MPI ranks to beat the reference benchmarks (24–40 ranks across 2 nodes with cross-node communication penalties).

## Recommended Approach: Data Distribution

**Why data distribution over query distribution:**
- Scatter data (O(N·A) total) + broadcast queries (O(Q·A/rank)) is cheaper than broadcasting the whole dataset to all 40 ranks (which would be O(N·A·39) across nodes).
- No need to replicate a potentially huge dataset on every rank.
- The local top-k merge step is cheap: O(size · max_k) per query at rank 0.

## Implementation Plan

### Phase 1 — Broadcast `Params` (all ranks need it; only rank 0 has it from `common.cpp`)

```cpp
int meta[3];
if (rank==0) { meta[0]=p.num_data; meta[1]=p.num_queries; meta[2]=p.num_attrs; }
MPI_Bcast(meta, 3, MPI_INT, 0, MPI_COMM_WORLD);
int num_data=meta[0], num_queries=meta[1], num_attrs=meta[2];
```

### Phase 2 — Scatter Dataset (rank 0 packs; all ranks receive their chunk)

- `dataset` is `vector<DataPoint>` (non-contiguous). Pack into two flat arrays:
  - `int` buffer: `[id, label]` per point → size `num_data * 2`
  - `double` buffer: `attrs[0..A-1]` per point → size `num_data * num_attrs`
- Compute per-rank chunks: `counts[r] = num_data/size + (r < num_data%size ? 1 : 0)`
- `MPI_Scatterv` both buffers. Non-root ranks pass `nullptr` for sendbuf.
- On all ranks: extract `local_labels[local_n]` and keep the flat `double` attr array for distance computation.
- Free rank-0 send buffers immediately after scatter.

### Phase 3 — Broadcast Queries (all ranks need all queries)

- Pack queries into `int[num_queries*2]` (id, k) and `double[num_queries*num_attrs]` (attrs).
- `MPI_Bcast` both buffers.

### Phase 4 — Local Top-K Computation per Rank

**Key tiebreak design — `(dist, -global_idx)` max-heap:**
- `priority_queue<pair<double,int>>` (default max-heap)
- At equal distance, pair with largest `-global_idx` (= smallest actual idx) sits at the top → gets popped first → we keep the larger-indexed point. This correctly implements "prefer larger index for ties".

**Loop structure (query-major — keeps query attrs hot in cache while streaming through data):**

```cpp
for (int qi = 0; qi < num_queries; qi++) {
    int k = q_k[qi];
    const double* qa = q_attrs + qi * num_attrs;
    auto& heap = heaps[qi];
    for (int di = 0; di < local_n; di++) {
        int gidx = local_start + di;
        const double* da = loc_data + di * num_attrs;
        // Early-exit: once partial sum > heap.top().first, skip
        double cutoff = (int(heap.size()) == k) ? heap.top().first : INF;
        double s = 0.0; bool pruned = false;
        for (int a = 0; a < num_attrs; a++) {
            double d = qa[a]-da[a]; s += d*d;
            if (s > cutoff) { pruned = true; break; }
        }
        if (int(heap.size()) < k) {
            heap.push({s, -gidx});
        } else if (!pruned && (s < heap.top().first ||
                   (s == heap.top().first && gidx > -heap.top().second))) {
            heap.pop(); heap.push({s, -gidx});
        }
    }
}
```

Edge cases handled:
- `local_n == 0` (num_data < size): inner loop doesn't run; heap stays empty; sentinel padding handles it.
- `local_n < k`: heap has fewer than k entries; global merge covers the deficit from other ranks.

### Phase 5 — Pack and Gather Local Top-K to Rank 0

- Compute `k_max = max k across all queries`
- Allocate flat send buffers: `double send_dists[num_queries * k_max]` and `int send_idxs[num_queries * k_max]`, initialized to `(INF, -1)` sentinels.
- Drain each query's heap into sorted order (drain + reverse), write to send buffer.
- `MPI_Gather` both buffers to rank 0 (fixed-size → simpler than Gatherv).
- Free send buffers and heaps after gather.

### Phase 6 — Merge and Output on Rank 0

For each query `qi` (in order 0..num_queries-1):
1. Collect all non-sentinel candidates from all ranks' blocks.
2. `std::sort` with comparator: ascending dist, then descending idx for ties.
3. `resize` to k.
4. Plurality label from `dataset[idx].label` (rank 0 still has original `dataset`). Tiebreak: largest label wins.
5. Call `reportResult(query, result, best_label)`.

## Critical Files to Modify

- **[engine.cpp](engine.cpp)** — full implementation of `Engine::KNN` (only file that needs changes)
- **[Makefile](Makefile)** — add `-march=native` to `CPPFLAGS` for AVX2 auto-vectorization of the inner distance loop (safe; grading machines are known x86-64 hardware)

## Important Notes on Correctness

- **`-global_idx` trick**: when `heap.top().second = -global_idx`, recover the actual index as `-heap.top().second`.
- **Result format for `reportResult`**: `vector<pair<double,int>>` where `.first = dist`, `.second = global_idx`. Ordered closest-first (ties: larger index first).
- **Queries output in order**: rank 0 loops `qi = 0..num_queries-1` sequentially; `local_queries` array preserves this order.
- **Label lookup**: rank 0 uses original `dataset` vector (unchanged throughout `KNN`).
- **`MPI_Scatterv` int overflow**: displacements are `int`. For `num_data * num_attrs > INT_MAX`, manual sends would be needed. For contest-scale inputs this is safe.

## Verification Steps

1. `make` — should produce `engine` and `engine.debug`
2. Run debug mode on sample input:
   ```
   echo "5 2 3
   0 1.0 2.0 3.0
   1 4.0 5.0 6.0
   0 7.0 8.0 9.0
   2 1.5 2.5 3.5
   1 4.5 5.5 6.5
   Q 2 1.0 2.1 3.0
   Q 3 7.2 8.0 9.1" | mpirun -n 2 ./engine.debug
   ```
   Expected: Label 2 for query 0 (neighbors 0, 3), Label 1 for query 1 (neighbors 2, 4, 1)
3. Test with 1, 2, 4, 8 ranks to verify correctness at all sizes
4. Test edge cases: `--n 1 --ranks 4` (fewer data than ranks), varying k values
5. On cluster: `./run_bench.sh 1` (requires `unzip inputs.zip` first)
