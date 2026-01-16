import time
from ipc_transport import PosixIPCReader

def main():
    print("Python Reader: Waiting for 'test_cpp_write'...")
    reader = PosixIPCReader("test_cpp_write")
    
    if reader.wait_for_array("image", timeout=5.0):
        print("Python Reader: Connected!")
    else:
        print("Python Reader: Timeout!")
        return

    for i in range(10):
        img = reader.get_image()
        if img is not None:
             print(f"Read frame {i}, Shape: {img.shape}, Red pixel: {img[0,0,2]}")
        time.sleep(0.1)

if __name__ == "__main__":
    main()
