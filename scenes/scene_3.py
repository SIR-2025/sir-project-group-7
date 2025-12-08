import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene3(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=3)
    
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
                    intro = self.generate_speech(
                        "Announce the training plan. Be professional.",
                        fallback_text="Here's our training plan:"
                    )
                    self.nao_speak(intro, animation="animations/Stand/Gestures/Explain_1", wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                elif self.scene_step == 2 and current_time - self.step_start_time > 2:
                    squats = self.generate_speech(
                        "Explain squats are for legs and hip mobility.",
                        fallback_text="Squats for the legs and hip mobility,"
                    )
                    self.nao_speak(squats, wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                elif self.scene_step == 3 and current_time - self.step_start_time > 3:
                    pushups = self.generate_speech(
                        "Explain push-ups are for upper body strength.",
                        fallback_text="Push-ups for your upper body strength,"
                    )
                    self.nao_speak(pushups, wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                elif self.scene_step == 4 and current_time - self.step_start_time > 3:
                    plank = self.generate_speech(
                        "Explain plank trains core and mental endurance.",
                        fallback_text="And a Plank to train your core and mental endurance."
                    )
                    self.nao_speak(plank, wait=True)
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # ACT 3: Excitement check
                elif self.scene_step == 5 and current_time - self.step_start_time > 4:
                    excitement = self.generate_speech(
                        "Ask if they're excited. Be enthusiastic.",
                        fallback_text="So, are you excited already?"
                    )
                    self.nao_speak(excitement, animation="animations/Stand/Gestures/Enthusiastic_1", wait=True)
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # ACT 4: Listen for response
                elif self.scene_step == 6 and current_time - self.step_start_time > 3:
                    self.start_listening("Person responding about excitement for the workout.")
                    self.scene_step = 61
                
                # FIXED: Don't respond, just continue to "Let's start then!"
                elif self.scene_step == 61 and self.is_listening_complete():
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                # ACT 5: Let's start
                elif self.scene_step == 7 and current_time - self.step_start_time > 1:
                    start = self.generate_speech(
                        "Say let's start with enthusiasm.",
                        fallback_text="Let's start then!"
                    )
                    self.nao_speak(start, animation="animations/Stand/Gestures/Yes_1", wait=True)
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