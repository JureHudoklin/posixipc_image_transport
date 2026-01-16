# PosixIPC Image Transport

A high-performance Python library for exchanging images, point clouds, and generic numpy arrays between processes using POSIX shared memory (`posix_ipc` and `mmap`).

Designed for low-latency video processing pipelines where copying data through queues or sockets is too slow.

## Features

- **Zero-Copy (ish):** Uses shared memory mapped files.
- **Fast:** Much faster than `multiprocessing.Queue` or `sockets` for large arrays.
- **Type Safe:** Automatically handles `dtype`, `shape`, and dimensions (up to 5D).
- **Synchronization:** Uses semaphores to prevent read/write tearing.
- **Robust:** Automatically cleans up stale shared memory segments on writer startup.
- **Easy API:** Specialized methods for images, depth, point clouds, and generic arrays.

## Installation

### From Source
```bash
git clone https://github.com/JureHudoklin/posixipc_image_transport.git
cd posixipc_image_transport
pip install .
```

### From GitHub (Directly)
```bash
pip install git+https://github.com/JureHudoklin/posixipc_image_transport.git
```

## Usage

### Writer Process (Producer)
The writer creates the shared memory segment. **Note:** Only one writer should exist for a given base name.

```python
import numpy as np
import time
from ipc_transport import PosixIPCWriter

# Initialize writer with a base name
writer = PosixIPCWriter("camera_01")

while True:
    # 1. Image (Standard method)
    # Automatically handles shape (H, W, C) and updates metadata
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    writer.set_image(img)
    
    # 2. Point Cloud (Standard method)
    pc = np.random.rand(1000, 3).astype(np.float32)
    writer.set_pointcloud(pc)
    
    # 3. Generic Array (Custom named channel)
    meta_data = np.array([1, 2, 3, 4], dtype=np.int32)
    writer.set_array("my_metadata_channel", meta_data)
    
    time.sleep(0.033)
```

### Reader Process (Consumer)
The reader connects to existing shared memory.

```python
import time
from ipc_transport import PosixIPCReader

reader = PosixIPCReader("camera_01")

# Optional: Wait for connection
print("Waiting for writer...")
reader.wait_for_array("image")

while True:
    # Get latest data
    # Returns None if lock cannot be acquired immediately or channel doesn't exist
    img = reader.get_image()
    
    if img is not None:
        print(f"Received image: {img.shape}")
        
    # Get generic array
    meta = reader.get_array("my_metadata_channel")
    
    time.sleep(0.01)
```

## API Reference

### Writers
- `set_image(np.ndarray)`
- `set_depth(np.ndarray)`
- `set_pointcloud(np.ndarray)`
- `set_mask(np.ndarray)`
- `set_array(name: str, array: np.ndarray)`

### Readers
- `get_image() -> np.ndarray | None`
- `get_...() -> np.ndarray | None` (same as writer)
- `get_array(name: str) -> np.ndarray | None`
- `get_shape(name: str)`, `get_dtype(name)`, `get_ndim(name)`
- `wait_for_array(name: str, timeout: float)`

## Requirements
- Python >= 3.9
- `posix_ipc`
- `numpy`
