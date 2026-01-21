import posix_ipc
import mmap
import numpy as np
import struct
import os
import enum
import time
from typing import Optional, Tuple, Union

# Protocol definition
# Header: [Magic (4 bytes), Dtype (4 bytes), Ndim (4 bytes), Shape (4*5=20 bytes), Timestamp (8 bytes)]
# Total Header Size = 64 bytes (aligned)
HEADER_SIZE = 64 # Give plenty of room for future
MAGIC = 0x12345678
TIMESTAMP_OFFSET = 32  # Offset in bytes where timestamp is stored (after header struct)

class DataType(enum.IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    INT32 = 4
    FLOAT32 = 5
    FLOAT64 = 6
    BOOL = 7

TO_NUMPY_DTYPE = {
    DataType.UINT8: np.uint8,
    DataType.INT8: np.int8,
    DataType.UINT16: np.uint16,
    DataType.INT16: np.int16,
    DataType.INT32: np.int32,
    DataType.FLOAT32: np.float32,
    DataType.FLOAT64: np.float64,
    DataType.BOOL: bool,
}

FROM_NUMPY_DTYPE = {
    np.dtype('uint8'): DataType.UINT8,
    np.dtype('int8'): DataType.INT8,
    np.dtype('uint16'): DataType.UINT16,
    np.dtype('int16'): DataType.INT16,
    np.dtype('int32'): DataType.INT32,
    np.dtype('float32'): DataType.FLOAT32,
    np.dtype('float64'): DataType.FLOAT64,
    np.dtype('bool'): DataType.BOOL,
}

class ChannelHandler:
    """Handles a single shared memory channel (e.g. 'image' or 'depth')."""
    def __init__(self, name: str, create: bool = False, shape: Tuple = None, dtype: np.dtype = None):
        self.name = name
        self.shm_name = f"/{name}_shm"
        self.sem_name = f"/{name}_sem"
        self.create = create
        self.shm: Optional[posix_ipc.SharedMemory] = None
        self.sem: Optional[posix_ipc.Semaphore] = None
        self.mmap: Optional[mmap.mmap] = None
        self.shape = shape
        self.dtype = dtype
        self.size = 0
        
        if self.create:
            if shape is None or dtype is None:
                raise ValueError("Shape and dtype must be provided when creating a channel.")
            self._initialize_writer()
        else:
            self._initialize_reader()

    def _get_dtype_enum(self, array_dtype):
        # Handle cases like <i4, >i4, etc by getting the standardized type
        dt = np.dtype(array_dtype)
        base_type = FROM_NUMPY_DTYPE.get(dt)
        if base_type is None:
            # Try to match string representation if direct lookup fails
            if dt.kind == 'u' and dt.itemsize == 1: base_type = DataType.UINT8
            elif dt.kind == 'i' and dt.itemsize == 1: base_type = DataType.INT8
            elif dt.kind == 'u' and dt.itemsize == 2: base_type = DataType.UINT16
            elif dt.kind == 'i' and dt.itemsize == 2: base_type = DataType.INT16
            elif dt.kind == 'i' and dt.itemsize == 4: base_type = DataType.INT32
            elif dt.kind == 'f' and dt.itemsize == 4: base_type = DataType.FLOAT32
            elif dt.kind == 'f' and dt.itemsize == 8: base_type = DataType.FLOAT64
            elif dt.kind == 'b': base_type = DataType.BOOL
            
        if base_type is None:
             raise ValueError(f"Unsupported dtype: {array_dtype}")
        return base_type

    def _initialize_writer(self):
        # Calculate size
        data_size = int(np.prod(self.shape)) * self.dtype.itemsize
        total_size = HEADER_SIZE + data_size
        self.size = total_size

        # Clean up stale objects if they exist to avoid deadlock from previous crashes
        try:
            posix_ipc.unlink_semaphore(self.sem_name)
        except posix_ipc.ExistentialError:
            pass
        except Exception:
            pass
            
        try:
             posix_ipc.unlink_shared_memory(self.shm_name)
        except posix_ipc.ExistentialError:
            pass
        except Exception:
            pass

        # Create Semaphore
        self.sem = posix_ipc.Semaphore(self.sem_name, flags=posix_ipc.O_CREAT, initial_value=1)

        # Create Shared Memory
        self.shm = posix_ipc.SharedMemory(self.shm_name, flags=posix_ipc.O_CREAT, size=total_size)
        
        # Mmap
        self.mmap = mmap.mmap(self.shm.fd, total_size)
        self.shm.close_fd()

        # Write Header
        # Format: Magic(I), Dtype(I), Ndim(I), Dim0(I), Dim1(I), Dim2(I), Dim3(I), Dim4(I)
        dims = list(self.shape) + [0] * (5 - len(self.shape)) # Pad to 5 dims
        dtype_enum = self._get_dtype_enum(self.dtype)
        
        header_data = struct.pack("IIIIIIII", MAGIC, dtype_enum, len(self.shape), dims[0], dims[1], dims[2], dims[3], dims[4])
        self.mmap.seek(0)
        self.mmap.write(header_data)

    def _initialize_reader(self):
        try:
            self.sem = posix_ipc.Semaphore(self.sem_name)
            self.shm = posix_ipc.SharedMemory(self.shm_name)
        except posix_ipc.ExistentialError:
            # Channel might not be ready yet
            raise FileNotFoundError(f"Channel {self.name} not found.")

        # Map header first to read size
        temp_mmap = mmap.mmap(self.shm.fd, HEADER_SIZE, prot=mmap.PROT_READ)
        magic, dtype_int, ndim, d0, d1, d2, d3, d4 = struct.unpack("IIIIIIII", temp_mmap[:32])
        
        if magic != MAGIC:
            raise ValueError("Invalid shared memory header magic.")
            
        dims = [d0, d1, d2, d3, d4][:ndim]
        self.shape = tuple(dims)
        self.dtype = np.dtype(TO_NUMPY_DTYPE[dtype_int])
        
        data_size = int(np.prod(self.shape)) * self.dtype.itemsize
        self.size = HEADER_SIZE + data_size
        
        temp_mmap.close()
        
        # Remap full size
        self.mmap = mmap.mmap(self.shm.fd, self.size) # Default prot is read/write
        self.shm.close_fd()

    def write(self, array: np.ndarray, timestamp: Optional[float] = None):
        if array.shape != self.shape:
             raise ValueError(f"Shape mismatch: Expected {self.shape}, got {array.shape}")
        if array.dtype != self.dtype:
             # Allow cast if safe? strict for now
             if not np.can_cast(array.dtype, self.dtype):
                 raise ValueError(f"Dtype mismatch: Expected {self.dtype}, got {array.dtype}")

        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = time.time()

        # Acquire lock
        try:
            self.sem.acquire(timeout=1.0) # 1 sec timeout
            
            # Write timestamp
            self.mmap.seek(TIMESTAMP_OFFSET)
            self.mmap.write(struct.pack("d", timestamp))
            
            # Write data
            # Using np.ndarray with buffer mechanism
            
            # Note: We need to be careful with layout. 
            # We construct a numpy array wrapping the mmap buffer at the offset
            # and copy the input array into it.
            
            dest_array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.mmap, offset=HEADER_SIZE)
            np.copyto(dest_array, array)
            
        except posix_ipc.BusyError:
            print(f"Warning: Write timeout on {self.name}")
        finally:
            self.sem.release()

    def read(self, return_timestamp: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[float]]]:
        try:
            self.sem.acquire(timeout=1.0)
            
            # Read timestamp
            self.mmap.seek(TIMESTAMP_OFFSET)
            timestamp_bytes = self.mmap.read(8)
            timestamp = struct.unpack("d", timestamp_bytes)[0] if timestamp_bytes else 0.0
            
            # Copy data out to avoid corruption during read if writer writes
            # Alternatively, we could return a view but that puts the lock concept in jeopardy 
            # if we release the lock and user is still reading.
            # Best is to copy.
            
            src_array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.mmap, offset=HEADER_SIZE)
            data = src_array.copy()
            
            if return_timestamp:
                return data, timestamp
            return data
            
        except posix_ipc.BusyError:
            # print(f"Warning: Read timeout on {self.name}")
            if return_timestamp:
                return None, None
            return None
        finally:
            self.sem.release()
            
    def cleanup(self):
        if self.mmap is not None and not self.mmap.closed:
            try:
                self.mmap.close()
            except ValueError:
                pass
        self.mmap = None
        
        # Only writer should unlink? Or explicit cleanup?
        # Usually writer manages lifecycle. 
        if self.create:
             if self.shm:
                 try: self.shm.unlink() 
                 except posix_ipc.ExistentialError: pass
                 except Exception: pass
                 self.shm = None
             if self.sem:
                 try: self.sem.unlink()
                 except posix_ipc.ExistentialError: pass
                 except Exception: pass
                 self.sem = None

    def close(self):
        if self.mmap is not None and not self.mmap.closed:
            try:
                self.mmap.close()
            except ValueError:
                pass
        self.mmap = None


class PosixIPCWriter:
    def __init__(self, base_name: str):
        self.base_name = base_name
        self.channels = {} # type: dict[str, ChannelHandler]

    def _ensure_channel(self, suffix: str, data: np.ndarray):
        channel_name = f"{self.base_name}_{suffix}"
        if suffix not in self.channels:
            # We need to assume new channel creation
            # If dimensions change, we rely on the user to request a new writer or we detect change
            self.channels[suffix] = ChannelHandler(channel_name, create=True, shape=data.shape, dtype=data.dtype)
        else:
             ch = self.channels[suffix]
             if ch.shape != data.shape or ch.dtype != data.dtype:
                 # Re-creation logic or error?
                 # For simplicity, warn and recreate (which unlinks old one)
                 # But unlinking old one breaks readers.
                 # Let's enforce static shapes for now.
                 if ch.shape != data.shape:
                    raise ValueError(f"Data shape changed for {suffix}. Old: {ch.shape}, New: {data.shape}. Dynamic resizing not supported yet.")
                 pass # Dtypes should match or be castable
        
        return self.channels[suffix]

    def set_image(self, image: np.ndarray, timestamp: Optional[float] = None):
        ch = self._ensure_channel("image", image)
        ch.write(image, timestamp)

    def set_depth(self, depth: np.ndarray, timestamp: Optional[float] = None):
        ch = self._ensure_channel("depth", depth)
        ch.write(depth, timestamp)

    def set_pointcloud(self, pc: np.ndarray, timestamp: Optional[float] = None):
        ch = self._ensure_channel("pointcloud", pc)
        ch.write(pc, timestamp)
        
    def set_mask(self, mask: np.ndarray, timestamp: Optional[float] = None):
        ch = self._ensure_channel("mask", mask)
        ch.write(mask, timestamp)

    def set_array(self, name: str, array: np.ndarray, timestamp: Optional[float] = None):
        ch = self._ensure_channel(name, array)
        ch.write(array, timestamp)

    def cleanup(self):
        for ch in self.channels.values():
            ch.cleanup()

    def __del__(self):
        self.cleanup()

class PosixIPCReader:
    def __init__(self, base_name: str):
        self.base_name = base_name
        self.channels = {} # type: dict[str, ChannelHandler]
        
    def _get_channel(self, suffix: str) -> Optional[ChannelHandler]:
        if suffix in self.channels:
            return self.channels[suffix]
        
        channel_name = f"{self.base_name}_{suffix}"
        try:
            ch = ChannelHandler(channel_name, create=False)
            self.channels[suffix] = ch
            return ch
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error connecting to {suffix}: {e}")
            return None

    def get_image(self, return_timestamp: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[float]]]:
        ch = self._get_channel("image")
        if ch: return ch.read(return_timestamp)
        return (None, None) if return_timestamp else None

    def get_depth(self, return_timestamp: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[float]]]:
        ch = self._get_channel("depth")
        if ch: return ch.read(return_timestamp)
        return (None, None) if return_timestamp else None

    def get_pointcloud(self, return_timestamp: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[float]]]:
        ch = self._get_channel("pointcloud")
        if ch: return ch.read(return_timestamp)
        return (None, None) if return_timestamp else None

    def get_mask(self, return_timestamp: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[float]]]:
        ch = self._get_channel("mask")
        if ch: return ch.read(return_timestamp)
        return (None, None) if return_timestamp else None
        
    def get_array(self, name: str, return_timestamp: bool = False) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[float]]]:
        ch = self._get_channel(name)
        if ch: return ch.read(return_timestamp)
        return (None, None) if return_timestamp else None

    def get_shape(self, name: str) -> Optional[Tuple]:
        """Get the shape of the array for a given channel name (e.g. 'image', 'depth')."""
        ch = self._get_channel(name)
        if ch: return ch.shape
        return None

    def get_dtype(self, name: str) -> Optional[np.dtype]:
        """Get the dtype of the array for a given channel name."""
        ch = self._get_channel(name)
        if ch: return ch.dtype
        return None

    def get_ndim(self, name: str) -> Optional[int]:
        """Get the number of dimensions of the array for a given channel name."""
        ch = self._get_channel(name)
        if ch: return len(ch.shape)
        return None

    def wait_for_array(self, name: str, timeout: float = 5.0) -> bool:
        """Waits for a channel with the given name to verify it exists."""
        start = time.time()
        while time.time() - start < timeout:
            ch = self._get_channel(name)
            if ch is not None:
                return True
            time.sleep(0.01)
        return False

    def close(self):
        for ch in self.channels.values():
            ch.close()
    
    def __del__(self):
        self.close()
