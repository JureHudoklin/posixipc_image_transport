#include "../../include/posixipc_image_transport.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace posixipc_image_transport;

int main() {
    std::cout << "C++ Writer: Starting 'test_cpp_write'..." << std::endl;
    PosixIPCWriter writer("test_cpp_write");
    
    int width = 640;
    int height = 480;
    int channels = 3;
    size_t size = width * height * channels;
    std::vector<uint8_t> buffer(size, 0);
    
    for(int i=0; i<50; ++i) {
        // Fill red channel
        for(size_t j=0; j<size; j+=3) {
            buffer[j+2] = (uint8_t)(i % 255);
        }
        
        writer.set_image(buffer.data(), size, height, width, channels);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "C++ Writer: Finished." << std::endl;
    return 0;
}
