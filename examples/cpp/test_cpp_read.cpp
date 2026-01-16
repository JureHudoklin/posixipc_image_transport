#include "../../include/posixipc_image_transport.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace posixipc_image_transport;

int main() {
    std::cout << "C++ Reader: Waiting for writer 'test_cpp_read'..." << std::endl;
    PosixIPCReader reader("test_cpp_read");
    
    // Wait for image channel
    int attempts = 0;
    while(reader.get_channel("image") == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        attempts++;
        if(attempts > 50) {
            std::cerr << "Timeout waiting for writer" << std::endl;
            return 1;
        }
    }
    
    std::cout << "C++ Reader: Connected!" << std::endl;
    
    auto shape = reader.get_shape("image");
    std::cout << "Shape: " << shape[0] << "x" << shape[1] << "x" << shape[2] << std::endl;
    
    size_t data_size = shape[0] * shape[1] * shape[2]; // uint8
    std::vector<uint8_t> buffer(data_size);
    
    for(int i=0; i<10; ++i) {
        if(reader.read("image", buffer.data(), buffer.size())) {
            std::cout << "Read frame " << i << ", First pixel B: " << (int)buffer[0] << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return 0;
}
