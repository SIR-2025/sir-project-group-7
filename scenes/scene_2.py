import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene2(BaseScene):
    def run(self):
        print("SCENE 2: MIRROR WARM-UP")
        
        try:
            while True:
                frame = self.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = self.draw_scene_info(frame, "Scene 2: Mirror Warm-Up")
                cv2.imshow("Training System Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = time.time()
                
                # ACT 1: Jingle
                if self.scene_step == 0:
                    self.play_jingle()
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # ACT 2: Arm raises
                elif self.scene_step == 1 and current_time - self.step_start_time > 2:
                    self.nao_speak("First: Raise your arms like me... or higher if you have human shoulders.",
                                  animation="animations/Stand/Gestures/Arms_1", wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # ACT 3: Leg shakes
                elif self.scene_step == 2 and current_time - self.step_start_time > 5:
                    self.nao_speak("Shake your legs, or do whatever humans do when they loosen up. Mine just... vibrate a bit.",
                                  animation="animations/Stand/BodyTalk/BodyTalk_1", wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # ACT 4: Listen for sarcastic comment
                elif self.scene_step == 3 and current_time - self.step_start_time > 6:
                    self.start_listening()
                    self.scene_step = 31
                
                elif self.scene_step == 31 and self.is_listening_complete():
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # ACT 5: Stretch left and feeling warm
                elif self.scene_step == 4 and current_time - self.step_start_time > 1:
                    self.nao_speak("Let's stretch left! My body can only go... this far. Respect my angles. Feeling warm already? Because then we can start with the real work, or at least you. Ha-ha-ha!",
                                  animation="animations/Stand/Gestures/Explain_1", wait=True)
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # ACT 6: Listen for final comment
                elif self.scene_step == 5 and current_time - self.step_start_time > 8:
                    self.start_listening()
                    self.scene_step = 51
                
                elif self.scene_step == 51 and self.is_listening_complete():
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # ACT 7: Closing
                elif self.scene_step == 6 and current_time - self.step_start_time > 1:
                    self.nao_speak("Sounds good!", wait=False)
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                elif self.scene_step == 7 and current_time - self.step_start_time > 2:
                    self.scene_step = 8
                
                elif self.scene_step == 8:
                    print("END OF SCENE 2")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()