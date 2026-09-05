# MiniHyperVec

MiniHyperVec is the open-source proof-of-concept version of Helmsman (The Clustering Strikes Back: Building Cost-Effective and High-Performance ANNS at Scale with Helmsman, OSDI 2026).

## Artifact Scope

This repository contains an open-source proof-of-concept implementation of Helmsman. It includes:

- **Core CPU-side components** for clustered ANNS serving
- **Index conversion tool** (`spann_build_index`) to transform SPTAG-generated SPANN indexes into the HV_CONST format required by MiniHyperVec
- **Deployment and search utilities** (`minihypervec_deploy`, `multi_thread_search`)
- **SPDK/NVMe user-space I/O support** for batch cluster loading and raw vector serving

### Not Included

This proof-of-concept artifact does not include all components of the full Helmsman system as described in the paper. Notably absent are:

- **GPU-accelerated build pipeline** — The paper's Section 5.3 describes GPU optimizations for index construction; these are not included in this PoC.
- **LLSP module** — The paper's Section 5.5 discusses LLSP (Latency-aware Load Scheduling); this component is omitted from the public PoC.

As a result, this repository demonstrates the main architecture and serving path rather than reproducing all optimizations presented in the full system. Pruning-related components (Section 5.4) are present but optional (see **Build Options** below).

Both the GPU-accelerated build pipeline and the LLSP module are key items on our roadmap and are planned for inclusion in a future release.

---

## Datasets

The benchmark datasets used by this project can be downloaded from the following Hugging Face repository:

- [Red-EAD/helmsman](https://huggingface.co/datasets/Red-EAD/helmsman)

The repository contains the `SimRec` and `SimSrch` datasets, including binary vector files (`vectors.bin`) and query files (`query.i8bin`). 

---

## Prerequisites

### Required Host Tools

- `cmake` (version 3.20 or higher)
- A C++ compiler with C++20 support (e.g., `g++` 11+, `clang++` 14+)
- `make`
- `git`
- **`pkg-config`** (essential for SPDK discovery)

**Important:** The top-level CMake configuration uses `find_package(PkgConfig REQUIRED)` and discovers SPDK libraries via pkg-config metadata (see `CMakeLists.txt` line 47). **Even if you only intend to build components without NVMe dependencies (e.g., `spann_build_index`), `pkg-config` must be installed during the configuration phase.** If configuration fails with "missing PKG_CONFIG_EXECUTABLE", install pkg-config:

- **Ubuntu/Debian:** `sudo apt install pkg-config`
- **macOS:** `brew install pkg-config`
- **RHEL/CentOS:** `sudo yum install pkgconfig`

### Workspace Setup

Prepare a dedicated workspace under `/mnt/service` for project data, indexes, and third-party dependencies.

Recommended directory layout:

```text
/mnt/service/
├── 3rd_party/
├── minihyper-vec/
└── ...
```

---

## Build Third-Party Dependencies

### 1. Build SPDK

```bash
cd /mnt/service/3rd_party
git clone --recursive https://github.com/spdk/spdk.git
cd spdk
git checkout v22.09
git submodule update --init --recursive

./scripts/pkgdep.sh
./configure --prefix=/mnt/service/3rd_party/SPDK_Path --with-shared
make -j"$(nproc)"
make install
```

This installs SPDK to:

```text
/mnt/service/3rd_party/SPDK_Path
```

### 2. Build oneTBB

```bash
cd /mnt/service/3rd_party
git clone https://github.com/oneapi-src/oneTBB.git
cd oneTBB
git checkout v2021.12.0

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/mnt/service/3rd_party/TBB_Path
make -j"$(nproc)"
make install
```

This installs oneTBB to:

```text
/mnt/service/3rd_party/TBB_Path
```

### Optional Dependencies (Pruning)

When `PRUNE_ON=ON`, the CMake build expects the following libraries at these locations (adjust paths in `CMakeLists.txt` or environment if your installation differs):

- **LightGBM:** `/mnt/service/3rd_party/LightGBM_Path`  
  Must contain `include/` (headers) and `lib/` (liblib_lightgbm.so)
  
- **ONNX Runtime:** `/mnt/service/3rd_party/ONNXRT_Path`  
  Must contain `include/` (headers) and `lib/` (libonnxruntime.so)

Build and install instructions for these dependencies are not provided in this README; please refer to the official repositories:

- [LightGBM Installation Guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)
- [ONNX Runtime Build Guide](https://onnxruntime.ai/docs/build/)

**If you do not have these libraries installed and do not need pruning, simply use `PRUNE_ON=OFF` during configuration (no additional dependencies required).**

---

## Project Setup

### 1. `setup/.envrc`

This file initializes the environment variables required by MiniHyperVec.

Example:

```bash
# add SPDK
export C_INCLUDE_PATH=/mnt/service/3rd_party/SPDK_Path/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/mnt/service/3rd_party/SPDK_Path/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/service/3rd_party/SPDK_Path/lib
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/mnt/service/3rd_party/SPDK_Path/lib/pkgconfig

# add DPDK
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/service/3rd_party/spdk/dpdk/build/lib
```

Before building or running the project, load the environment:

```bash
source ./setup/.envrc
```

### 2. `setup/path_config.json`

This file stores the paths for NVMe metadata and released indexes.

Example:

```json
{
    "nvme_meta_path": "/mnt/service/minihyper-vec/tmp_file/nvme_meta/",
    "release_index_path": "/mnt/service/minihyper-vec/tmp_file/release_index/"
}
```

Field description:

* `nvme_meta_path`: directory used to store generated NVMe metadata
* `release_index_path`: directory used to store released indexes

---

## Build MiniHyperVec

### Build Options

The default CMake configuration has `PRUNE_ON=ON` (see `CMakeLists.txt` line 3). When pruning is enabled, the build requires additional third-party dependencies for LightGBM and ONNX Runtime, which implement the pruning functionality described in Section 5.4 of the paper.

If you do not require pruning features (and the corresponding pruning results in the paper), build with `PRUNE_ON=OFF`:

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_CXX_STANDARD=20 \
  -DPRUNE_ON=OFF
make -j
```

**If pruning is disabled**, the pruning-related components and experimental results from Section 5.4 will not be available in this build. The core ANNS serving pipeline, index conversion, and NVMe-based deployment will still function normally.

### Full Build with Pruning (Optional)

If you have LightGBM and ONNX Runtime installed at the paths specified in the "Optional Dependencies" section below, you can enable pruning:

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_CXX_STANDARD=20 \
  -DPRUNE_ON=ON
make -j
```

### Build Notes

* **`CMAKE_CXX_STANDARD=20`:** Although `CMakeLists.txt` defaults to C++17, please explicitly set C++20 as shown above to ensure compatibility with all project features.

---

## Index Builder

### Step 1: Build SPANN Index

MiniHyperVec uses SPANN-format indexes as input. You need to first build a SPANN index using the [SPTAG](https://github.com/microsoft/SPTAG) library.

Please refer to the [SPTAG documentation](https://github.com/microsoft/SPTAG/blob/main/docs/GettingStart.md) for detailed build and usage instructions. After building SPTAG, use its SPANN indexing pipeline to generate the following files:

* `SPTAGHeadVectors.bin` — centroid (head) vectors
* `SPTAGHeadVectorIDs.bin` — centroid vector IDs
* `SPTAGFullList.bin` — posting lists for all clusters


### Step 2: Convert to MiniHyperVec Format

`spann_build_index` is a standalone tool that converts the SPTAG-generated SPANN index into the HV_CONST format required by MiniHyperVec. It performs the following steps:

1. Load the SPANN head vectors, head IDs, and posting lists produced by SPTAG.
2. Build an HNSW index over the head (centroid) vectors.
3. Run the **Filling** algorithm to pad each cluster to a fixed size by borrowing vectors from neighboring clusters.
4. Compute per-vector norms and serialize the final index artifacts.

### Build

`spann_build_index` is enabled by default. To toggle it:

```bash
# Enabled (default)
cmake .. -DSPANN_AUG_BUILD=ON

# Disabled
cmake .. -DSPANN_AUG_BUILD=OFF
```

Build only `spann_build_index`:

```bash
make spann_build_index
```

The binary is output to `build/app/spann_build_index`.

### Usage

```bash
./build/app/spann_build_index [options]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--spann-dir` | Yes | — | SPANN index directory (must contain the input files listed below) |
| `--output-dir` | Yes | — | Output directory for the generated index |
| `--prefix` | No | `spann` | Filename prefix for all output files |
| `--fill-target-size` | No | `64` | Target cluster size after filling |
| `--fill-neighbor-topk` | No | `16` | Number of neighbor clusters to search during filling |
| `--fill-count-head` | No | `true` | Whether to count the head vector in the cluster size |
| `--fill-num-threads` | No | `32` | Number of threads for the filling stage |
| `--hnsw-M` | No | `16` | HNSW M parameter (max edges per node) |
| `--hnsw-ef-construction` | No | `200` | HNSW ef\_construction parameter |
| `--hnsw-ef-search` | No | `128` | HNSW ef\_search parameter |
| `--show-summary` | No | `true` | Print index summary before processing |

### Input Files

The `--spann-dir` directory must contain the following SPTAG-format files:

| File | Description |
|---|---|
| `SPTAGHeadVectors.bin` | Centroid (head) vectors, int8, shape = (num\_heads, dim) |
| `SPTAGHeadVectorIDs.bin` | Centroid vector IDs, int64, shape = (num\_heads,) |
| `SPTAGFullList.bin` | Posting lists for all clusters (disk layout) |

### Output Files

The tool generates the following files in `--output-dir`:

| File | Description |
|---|---|
| `{prefix}_centroids_index.bin` | HNSW index over centroid vectors |
| `{prefix}_cluster_ids.bin` | Cluster posting-list IDs (fixed-size, uint64) |
| `{prefix}_cluster_norms.bin` | Per-vector L2 norms within each cluster |
| `{prefix}_index_meta.json` | Index metadata (dim, centroid count, build params, etc.) |

### Copy Raw Data

After `spann_build_index` completes, copy the original raw dataset into the output index directory and rename it to match the prefix:

```bash
cp <raw_dataset_path> <output-dir>/{prefix}_rawdata.bin
```
This file is required by MiniHyperVec at deployment time for serving raw vectors.



After this, you can deploy the collection using `minihypervec_deploy`:

```bash
./build/test/minihypervec_deploy <collection_name>
```

---

## NVMe Initialization

Before using SPDK-managed NVMe devices, bind the target NVMe SSD to the proper userspace driver.

### 1. Check the NVMe device path

```bash
readlink -f /sys/class/nvme/nvme0
```

### 2. Load VFIO modules

```bash
sudo modprobe vfio
sudo modprobe vfio_pci
```

### 3. Check the current driver binding

```bash
lspci -k -s 0000:43:00.0
```

### 4. Run SPDK setup

```bash
cd /mnt/service/3rd_party/spdk
sudo HUGEMEM=16384 ./scripts/setup.sh
```

### 5. Verify the driver binding again

```bash
lspci -k -s 0000:43:00.0
```

---

## Generate NVMe Metadata

Whenever a new NVMe device is bound to SPDK, run `config_nvme_meta` again so that the project can rescan the device and regenerate the NVMe metadata files.

```bash
sudo ./app/config_nvme_meta
```

This tool probes the available NVMe devices and generates the metadata used by the project.

The output metadata files are stored under the directory specified by `nvme_meta_path` in `setup/path_config.json`.

---

## Deployment

Use `minihypervec_deploy` to deploy a target collection.

### Usage

```bash
./build/test/minihypervec_deploy <collection_name>
```

### Parameters

* `<collection_name>`: collection name to deploy

Example:

```bash
./build/test/minihypervec_deploy sift10m_int8_spann
```

---

## Search

Use `multi_thread_search` to run multi-threaded search and evaluate retrieval quality against ground truth.

### Usage

```bash
./build/test/multi_thread_search \
  --collection_name <collection_name> \
  --query_path <query_path> \
  --groundtruth_path <groundtruth_path> \
  --index_type <index_type> \
  --vec_type <vec_type> \
  --nprobe <nprobe> \
  --topk <topk> \
  --T <num_threads> \
  --memory_index_type <memory_index_type> \
  --memory_search_max_visits <max_visits>
```

### Parameters

* `--collection_name`
  Collection name.

* `--query_path`
  Path to the query file.

* `--groundtruth_path`
  Path to the ground-truth file used for evaluation.

* `--index_type`
  Index type.

* `--vec_type`
  Vector data type.

* `--nprobe`
  Number of probed candidates/partitions during search.

* `--topk`
  Number of nearest neighbors returned for each query.

* `--T`
  Number of threads used during search.

* `--memory_index_type`
  In-memory index type. Currently, only `HNSW` is supported.

* `--memory_search_max_visits`
  Maximum number of visited nodes during the in-memory index search stage.

### Example

```bash
./build/test/multi_thread_search \
  --collection_name <collection_name> \
  --query_path <query_path> \
  --groundtruth_path <groundtruth_path> \
  --index_type HV_CONST \
  --vec_type INT8 \
  --nprobe 36 \
  --topk 10 \
  --T 10 \
  --memory_index_type HNSW \
  --memory_search_max_visits 1800
```

---

## Notes

1. Always load the project environment before building or running:

```bash
source ./setup/.envrc
```

2. If you add or rebind NVMe devices to SPDK, rerun:

```bash
sudo ./app/config_nvme_meta
```

3. Make sure `path_config.json` points to valid writable directories.

4. Ensure that the collection name, query file, and ground-truth file are consistent with the deployed dataset and index configuration.

## Citation
If you use Helmsman in your research, please cite our papers:

```
@inproceedings{Osdi2026Helmsman,
	author = {Huang, Yuchen and Ma, Baiteng and Sun, Yiping and Shi, Yang and Chen, Xiao and Zhong, Xiaocheng and Wang, Zhiyong and Hu, Yao and Xu, Erci and Weng, Chuliang},
	title = {{The Clustering Strikes Back: Building Cost-Effective and High-Performance ANNS at Scale with Helmsman}},
	year = {2026},
    booktitle = {Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation},
    series = {OSDI '26},
    publisher = {USENIX Association},
    address = {USA},
    url = {https://www.usenix.org/conference/osdi26/presentation/huang-yuchen},
}
```

