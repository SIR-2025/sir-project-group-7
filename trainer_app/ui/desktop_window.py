import cv2

class DesktopWindow:
    def __init__(self, camera):
        self.camera = camera

    async def start(self):
        pass  # No GUI yet; can expand later

    def update_frame(self, frame):
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

    def flash_green(self):
        print("MOTIVATION FLASH")
