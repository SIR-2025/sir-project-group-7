# test_dual_cameras.py
from sic_framework.devices import Nao
from vision.camera_manager import CameraManager
import cv2
import time

NAO_IP = "10.0.0.241"

print("Testing dual camera setup...")

# Connect to NAO
print(f"\n1. Connecting to NAO at {NAO_IP}...")
nao = Nao(ip=NAO_IP)
print("   ✓ Connected!")

# Setup laptop camera
print("\n2. Setting up laptop camera...")
laptop_cam = CameraManager(use_local_camera=True, camera_index=0, use_threading=True)
if laptop_cam.is_available():
    print("   ✓ Laptop camera ready")
else:
    print("   ✗ Laptop camera failed!")
    exit(1)

# Setup NAO camera
print("\n3. Setting up NAO camera...")
nao_cam = CameraManager(nao=nao, use_local_camera=False, use_threading=True)
if nao_cam.is_available():
    print("   ✓ NAO camera ready")
else:
    print("   ✗ NAO camera failed!")
    exit(1)

# Create windows
print("\n4. Creating windows...")
cv2.namedWindow("Laptop Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("NAO Camera", cv2.WINDOW_NORMAL)

cv2.moveWindow("Laptop Camera", 50, 50)
cv2.resizeWindow("Laptop Camera", 640, 480)

cv2.moveWindow("NAO Camera", 700, 50)
cv2.resizeWindow("NAO Camera", 640, 480)

print("   ✓ Windows created")

# Display both cameras for 10 seconds
print("\n5. Displaying both cameras for 10 seconds...")
print("   Press 'q' to quit early\n")

start_time = time.time()
frame_count = 0

while (time.time() - start_time) < 10:
    # Laptop camera
    laptop_frame = laptop_cam.capture_frame()
    if laptop_frame is not None:
        cv2.putText(laptop_frame, "LAPTOP CAMERA",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(laptop_frame, f"Frame {frame_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Laptop Camera", laptop_frame)

    # NAO camera
    nao_frame = nao_cam.capture_frame()
    if nao_frame is not None:
        cv2.putText(nao_frame, "NAO CAMERA",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(nao_frame, f"Frame {frame_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("NAO Camera", nao_frame)

    frame_count += 1

    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n   Displayed {frame_count} frames")

# Cleanup
laptop_cam.cleanup()
nao_cam.cleanup()
cv2.destroyAllWindows()

print("\n✓ Test complete!")