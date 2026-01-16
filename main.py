import time
import numpy as np
import multiprocessing
import os
from ipc_transport import PosixIPCWriter, PosixIPCReader

def writer_process(base_name, iterations=20):
    print(f"[Writer] Starting with base_name: {base_name}")
    writer = PosixIPCWriter(base_name)
    
    # Create dummy data
    img_shape = (480, 640, 3)
    depth_shape = (480, 640)
    pc_shape = (1000, 3)
    
    for i in range(iterations):
        # Generate dynamic data
        img = np.full(img_shape, i % 255, dtype=np.uint8)
        depth = np.full(depth_shape, i * 10.0, dtype=np.float32)
        pc = np.random.rand(*pc_shape).astype(np.float32) * i
        
        # Write
        writer.set_image(img)
        writer.set_depth(depth)
        writer.set_pointcloud(pc)
        
        # print(f"[Writer] Wrote frame {i}")
        time.sleep(0.05)
        
    print("[Writer] Done.")
    # Keep alive for a bit so reader can finish
    time.sleep(1)
    writer.cleanup()

def reader_process(base_name, iterations=20):
    print(f"[Reader] Starting with base_name: {base_name}")
    # Give writer a moment to start
    time.sleep(0.5)
    
    reader = PosixIPCReader(base_name)
    
    count = 0
    while count < iterations:
        img = reader.get_image()
        depth = reader.get_depth()
        pc = reader.get_pointcloud()
        
        if img is not None:
            # print(f"[Reader] Got image {img.shape} val={img[0,0,0]}")
            pass
        else:
            print("[Reader] No image")
            
        if depth is not None:
             # print(f"[Reader] Got depth {depth.shape}")
             pass
             
        if pc is not None:
            # print(f"[Reader] Got PC {pc.shape}")
            pass

        count += 1
        time.sleep(0.05)
        
    print("[Reader] Done.")

if __name__ == "__main__":
    name = "test_cam"
    
    p1 = multiprocessing.Process(target=writer_process, args=(name,))
    p2 = multiprocessing.Process(target=reader_process, args=(name,))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    print("Test finished")
