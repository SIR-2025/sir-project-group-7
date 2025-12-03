import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene3(BaseScene):
    def run(self):
        print("SCENE 3: CIRCUIT INTRODUCTION")
        
        try:
            while True:
                frame = self.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = self.draw_scene_info(frame, "Scene 3: Circuit Introduction")
                cv2.imshow("Training System Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = time.time()
                
                # ACT 1: Jingle
                if self.scene_step == 0:
                    self.play_jingle()
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # ACT 2: Circuit explanation
                elif self.scene_step == 1 and current_time - self.step_start_time > 2:
                    self.nao_speak("Here's our training plan:",
                                  animation="animations/Stand/Gestures/Explain_1", wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                elif self.scene_step == 2 and current_time - self.step_start_time > 2:
                    self.nao_speak("Squats for the legs and hip mobility,", wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                elif self.scene_step == 3 and current_time - self.step_start_time > 3:
                    self.nao_speak("Push-ups for your upper body strength,", wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                elif self.scene_step == 4 and current_time - self.step_start_time > 3:
                    self.nao_speak("And a Plank to train your core and mental endurance.", wait=True)
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # ACT 3: Excitement check
                elif self.scene_step == 5 and current_time - self.step_start_time > 4:
                    self.nao_speak("So, are you excited already?",
                                  animation="animations/Stand/Gestures/Enthusiastic_1", wait=True)
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # ACT 4: Listen for response
                elif self.scene_step == 6 and current_time - self.step_start_time > 3:
                    self.start_listening()
                    self.scene_step = 61
                
                elif self.scene_step == 61 and self.is_listening_complete():
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                # ACT 5: Let's start
                elif self.scene_step == 7 and current_time - self.step_start_time > 1:
                    self.nao_speak("Let's start then!",
                                  animation="animations/Stand/Gestures/Yes_1", wait=True)
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                elif self.scene_step == 8 and current_time - self.step_start_time > 2:
                    self.scene_step = 9
                
                elif self.scene_step == 9:
                    print("END OF SCENE 3")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()