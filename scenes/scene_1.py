import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene1(BaseScene):
    def run(self):
        print("SCENE 1: GREETING & CALIBRATION")
        
        try:
            while True:
                frame = self.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = self.draw_scene_info(frame, "Scene 1: Greeting & Calibration")
                cv2.imshow("Training System Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = time.time()
                
                # ACT 1: First greeting - "Hey there you are, almost did not see you"
                if self.scene_step == 0:
                    self.nao_speak("Hey there you are, almost did not see you!",
                                  animation="animations/Stand/Gestures/Hey_1", wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # ACT 2: Listen for person's response - "Who let you in here?!"
                elif self.scene_step == 1 and current_time - self.step_start_time > 3:
                    self.start_listening()
                    self.scene_step = 11
                
                elif self.scene_step == 11 and self.is_listening_complete():
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # ACT 3: Robot ignores and introduces himself
                # "I'm Coach Nao, the smallest but smartest trainer in town."
                elif self.scene_step == 2 and current_time - self.step_start_time > 1:
                    self.nao_speak("I'm Coach Nao, the smallest but smartest trainer in town.", wait=True)
                    self.scene_step = 21
                    self.step_start_time = current_time
                
                # "I will be your trainer today and guide you through the most intense session you will ever be doing."
                elif self.scene_step == 21 and current_time - self.step_start_time > 4:
                    self.nao_speak("I will be your trainer today and guide you through the most intense session you will ever be doing.", 
                                  wait=True)
                    self.scene_step = 22
                    self.step_start_time = current_time
                
                # *Fist pump*
                elif self.scene_step == 22 and current_time - self.step_start_time > 5:
                    self.nao_animate("animations/Stand/Gestures/Yes_1")
                    self.scene_step = 23
                    self.step_start_time = current_time
                
                # (Awkward pause)
                elif self.scene_step == 23 and current_time - self.step_start_time > 2:
                    self.scene_step = 24
                    self.step_start_time = current_time
                
                # "Okay, let's make sure I can observe your movements. Please stand in front of me so I can detect your movements today."
                elif self.scene_step == 24 and current_time - self.step_start_time > 1:
                    self.nao_speak("Okay, let's make sure I can observe your movements. Please stand in front of me so I can detect your movements today.",
                                  wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # ACT 4: Listen for person's agreement
                # "Okay it's kind of weird that you got in here but I was just about to do my home workout so I guess you could help out."
                elif self.scene_step == 3 and current_time - self.step_start_time > 6:
                    self.start_listening()
                    self.scene_step = 31
                
                elif self.scene_step == 31 and self.is_listening_complete():
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # Wait a moment (person moves to mark)
                elif self.scene_step == 4 and current_time - self.step_start_time > 2:
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # ACT 5: "Perfect. Ready for round one?"
                elif self.scene_step == 5 and current_time - self.step_start_time > 1:
                    self.nao_speak("Perfect. Ready for round one?",
                                  animation="animations/Stand/Gestures/Enthusiastic_1", wait=True)
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # Listen for "Yeah, sure..."
                elif self.scene_step == 6 and current_time - self.step_start_time > 3:
                    self.start_listening()
                    self.scene_step = 61
                
                elif self.scene_step == 61 and self.is_listening_complete():
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                # Wait a moment before ending
                elif self.scene_step == 7 and current_time - self.step_start_time > 1:
                    self.scene_step = 8
                
                elif self.scene_step == 8:
                    print("END OF SCENE 1")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()