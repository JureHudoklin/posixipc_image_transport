import numpy as np
import time
from ipc_transport import PosixIPCWriter

def main():
    writer = PosixIPCWriter("test_cpp_read")
    print("Python Writer: Started. Writing to 'test_cpp_read'")
    
    # Create a simple gradient image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    count = 0
    while True:
        img[:, :, 0] = count % 255 # Blue channel cycles
        writer.set_image(img)
        count += 1
        time.sleep(0.1)
        if count > 50: break # Run for 5 seconds
    
    print("Python Writer: Finished.")

if __name__ == "__main__":
    main()
