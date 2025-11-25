class RepCounter:
    def __init__(self):
        self.state = "down"
        self.count = 0

    def update(self, keypoints):
        if keypoints is None:
            return self.count

        # Example: use left shoulder (index 5) y coordinate
        left_shoulder_y = keypoints[0][5][1]  
        left_hip_y = keypoints[0][11][1]

        # Simple rule: shoulder moves above hip
        if left_shoulder_y < left_hip_y and self.state == "down":
            self.state = "up"
            self.count += 1

        if left_shoulder_y > left_hip_y:
            self.state = "down"

        return self.count
