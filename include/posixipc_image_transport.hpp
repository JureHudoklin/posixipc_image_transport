#pragma once

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <memory>
#include <algorithm>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>

namespace posixipc_image_transport {

static const size_t HEADER_SIZE = 64;
static const uint32_t MAGIC = 0x12345678;
static const size_t TIMESTAMP_OFFSET = 32;  // Offset in bytes where timestamp is stored

enum class DataType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    INT32 = 4,
    FLOAT32 = 5,
    FLOAT64 = 6,
    BOOL = 7
};

struct Header {
    uint32_t magic;
    DataType dtype;
    uint32_t ndim;
    uint32_t dims[5];
    // This structure is 32 bytes.
    // The total header size in the file is 64 bytes, meaning 32 bytes of padding follow this struct.
};

inline size_t get_dtype_size(DataType dt) {
    switch (dt) {
        case DataType::UINT8: return 1;
        case DataType::INT8: return 1;
        case DataType::UINT16: return 2;
        case DataType::INT16: return 2;
        case DataType::INT32: return 4;
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT64: return 8;
        case DataType::BOOL: return 1; // standard bool is implementation defined, but numpy bool is 1 byte
        default: throw std::runtime_error("Unknown dtype size");
    }
}

class ChannelHandler {
public:
    std::string name;
    std::string shm_name;
    std::string sem_name;
    bool create;
    
    int shm_fd = -1;
    sem_t* sem = SEM_FAILED;
    void* mmap_addr = MAP_FAILED;
    size_t size = 0;
    
    std::vector<uint32_t> shape;
    DataType dtype;

    ChannelHandler(const std::string& name, bool create, const std::vector<uint32_t>& shape_ = {}, DataType dtype_ = DataType::UINT8)
        : name(name), create(create), shape(shape_), dtype(dtype_) {
        
        shm_name = "/" + name + "_shm";
        sem_name = "/" + name + "_sem";
        
        if (create) {
            initialize_writer();
        } else {
            initialize_reader();
        }
    }
    
    ~ChannelHandler() {
        close_resources();
    }
    
    void initialize_writer() {
        size_t data_size = get_dtype_size(dtype);
        for (auto s : shape) data_size *= s;
        size = HEADER_SIZE + data_size;
        
        // Cleanup stale
        sem_unlink(sem_name.c_str());
        shm_unlink(shm_name.c_str());
        
        // Create Sem
        sem = sem_open(sem_name.c_str(), O_CREAT, 0666, 1);
        if (sem == SEM_FAILED) {
            throw std::runtime_error("Failed to create semaphore: " + sem_name);
        }
        
        // Create Shm
        shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            throw std::runtime_error("Failed to create shm: " + shm_name);
        }
        
        if (ftruncate(shm_fd, size) == -1) {
            throw std::runtime_error("Failed to truncate shm");
        }
        
        // Mmap
        mmap_addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (mmap_addr == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap");
        }
        
        // Write Header
        uint32_t dims[5] = {0};
        for (size_t i = 0; i < shape.size() && i < 5; ++i) {
            dims[i] = shape[i];
        }
        
        struct Header h;
        h.magic = MAGIC;
        h.dtype = dtype;
        h.ndim = (uint32_t)shape.size();
        for(int i=0; i<5; i++) h.dims[i] = dims[i];
        
        // We write the header at offset 0
        memcpy(mmap_addr, &h, sizeof(h));
    }
    
    void initialize_reader() {
        sem = sem_open(sem_name.c_str(), 0);
        if (sem == SEM_FAILED) {
            throw std::runtime_error("Failed to open semaphore" + sem_name);
        }
        
        shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
        if (shm_fd == -1) {
            throw std::runtime_error("Failed to open shm" + shm_name);
        }
        
        // Map header first to read size
        void* header_map = mmap(NULL, HEADER_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
        if (header_map == MAP_FAILED) {
             throw std::runtime_error("Failed to mmap header");
        }
        
        Header h;
        memcpy(&h, header_map, sizeof(h));
        munmap(header_map, HEADER_SIZE); // Close temp map
        
        if (h.magic != MAGIC) {
            throw std::runtime_error("Invalid magic in shared memory header");
        }
        
        shape.clear();
        for(uint32_t i=0; i<h.ndim; i++) {
            shape.push_back(h.dims[i]);
        }
        dtype = h.dtype;
        
        size_t data_size = get_dtype_size(dtype);
        for (auto s : shape) data_size *= s;
        size = HEADER_SIZE + data_size;
        
        mmap_addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (mmap_addr == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap full size");
        }
    }
    
    void write(const void* data, size_t data_len, double timestamp = -1.0) {
        if (sem_wait(sem) == -1) {
            // Handle error, maybe EINTR
             std::cerr << "Warning: sem_wait failed" << std::endl;
             return;
        }
        
        // Safety check on size
        size_t expected_size = size - HEADER_SIZE;
        if (data_len != expected_size) {
            sem_post(sem);
            throw std::runtime_error("Data size mismatch");
        }
        
        // Use current time if no timestamp provided
        if (timestamp < 0) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            timestamp = ts.tv_sec + ts.tv_nsec / 1e9;
        }
        
        // Write timestamp at offset
        memcpy((char*)mmap_addr + TIMESTAMP_OFFSET, &timestamp, sizeof(double));
        
        memcpy((char*)mmap_addr + HEADER_SIZE, data, data_len);
        
        sem_post(sem);
    }
    
    void read(void* dest, size_t dest_len, double* timestamp = nullptr) {
        if (sem_wait(sem) == -1) {
            std::cerr << "Warning: sem_wait failed" << std::endl;
            return;
        }
        
        size_t expected_size = size - HEADER_SIZE;
        if (dest_len != expected_size) {
             sem_post(sem);
             throw std::runtime_error("Destination size mismatch");
        }
        
        // Read timestamp if requested
        if (timestamp != nullptr) {
            memcpy(timestamp, (char*)mmap_addr + TIMESTAMP_OFFSET, sizeof(double));
        }
        
        memcpy(dest, (char*)mmap_addr + HEADER_SIZE, dest_len);
        
        sem_post(sem);
    }
    
    void close_resources() {
        if (mmap_addr != MAP_FAILED) {
            munmap(mmap_addr, size);
            mmap_addr = MAP_FAILED;
        }
        if (shm_fd != -1) {
            close(shm_fd);
            shm_fd = -1;
            if (create) {
                shm_unlink(shm_name.c_str());
            }
        }
        if (sem != SEM_FAILED) {
            sem_close(sem);
            if (create) {
                sem_unlink(sem_name.c_str());
            }
            sem = SEM_FAILED;
        }
    }
};

class PosixIPCWriter {
    std::string base_name;
    std::map<std::string, std::shared_ptr<ChannelHandler>> channels;
    
public:
    PosixIPCWriter(const std::string& base_name) : base_name(base_name) {}
    
    std::shared_ptr<ChannelHandler> ensure_channel(const std::string& suffix, const std::vector<uint32_t>& shape, DataType dtype) {
        std::string name = base_name + "_" + suffix;
        if (channels.find(suffix) == channels.end()) {
            channels[suffix] = std::make_shared<ChannelHandler>(name, true, shape, dtype);
        }
         // Assumes shape/dtype validation is done or managed by ChannelHandler logic (which currently is strict on creation)
        return channels[suffix];
    }
    
    void set_data(const std::string& suffix, const void* data, size_t data_byte_size, const std::vector<uint32_t>& shape, DataType dtype, double timestamp = -1.0) {
        auto ch = ensure_channel(suffix, shape, dtype);
        ch->write(data, data_byte_size, timestamp);
    }
    
    // Helper helpers
    void set_image(const void* data, size_t size, uint32_t height, uint32_t width, uint32_t channels, DataType dtype=DataType::UINT8, double timestamp = -1.0) {
        set_data("image", data, size, {height, width, channels}, dtype, timestamp);
    }
};

class PosixIPCReader {
    std::string base_name;
    std::map<std::string, std::shared_ptr<ChannelHandler>> channels;

public:
    PosixIPCReader(const std::string& base_name) : base_name(base_name) {}
    
    std::shared_ptr<ChannelHandler> get_channel(const std::string& suffix) {
         if (channels.find(suffix) != channels.end()) {
            return channels[suffix];
        }
        
        std::string name = base_name + "_" + suffix;
        try {
            auto ch = std::make_shared<ChannelHandler>(name, false);
            channels[suffix] = ch;
            return ch;
        } catch (const std::exception& e) {
            // std::cerr << "Could not open channel " << suffix << ": " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    bool read(const std::string& suffix, void* dest, size_t dest_size, double* timestamp = nullptr) {
        auto ch = get_channel(suffix);
        if (!ch) return false;
        
        // Validate size?
        size_t expected_size = get_dtype_size(ch->dtype);
        for(auto s : ch->shape) expected_size *= s;
        
        if (dest_size != expected_size) {
             std::cerr << "Dest size mismatch for " << suffix << ". Expected " << expected_size << std::endl;
             return false;
        }
        
        ch->read(dest, dest_size, timestamp);
        return true;
    }
    
    std::vector<uint32_t> get_shape(const std::string& suffix) {
        auto ch = get_channel(suffix);
        if (ch) return ch->shape;
        return {};
    }
};

}
