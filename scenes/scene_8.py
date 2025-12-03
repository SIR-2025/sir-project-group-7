import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene8(BaseScene):
    def run(self):
        print("SCENE 8: THE UNLIKELY PARTNERSHIP")
        
        try:
            while True:
                frame = self.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = self.draw_scene_info(frame, "Scene 8: Finale")
                cv2.imshow("Training System Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = time.time()
                
                # ACT 1: Human agrees to partnership
                # Listen for "You know what, Nao? For a lost little robot..."
                if self.scene_step == 0:
                    print("[Human stands up]")
                    self.start_listening()
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                elif self.scene_step == 1 and self.is_listening_complete():
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # ACT 2: NAO's happy response
                elif self.scene_step == 2 and current_time - self.step_start_time > 1:
                    print("[NAO does happy little dance with feet]")
                    if self.use_nao and self.nao:
                        self.nao_animate("animations/Stand/Gestures/Enthusiastic_1")
                    time.sleep(2)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                elif self.scene_step == 3 and current_time - self.step_start_time > 1:
                    self.nao_speak("Yes I understand! I will attempt to knock on your door! And I'll double-check the apartment number! Probably!",
                                  wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # Listen for "Probably?"
                elif self.scene_step == 4 and current_time - self.step_start_time > 5:
                    self.start_listening()
                    self.scene_step = 41
                
                elif self.scene_step == 41 and self.is_listening_complete():
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # ACT 3: Setting expectations
                elif self.scene_step == 5 and current_time - self.step_start_time > 1:
                    self.nao_speak("Let's not set unrealistic expectations. Same time next week, Lucas of 3B?",
                                  wait=True)
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # Listen for "Looking forward to it..."
                elif self.scene_step == 6 and current_time - self.step_start_time > 4:
                    self.start_listening()
                    self.scene_step = 61
                
                elif self.scene_step == 61 and self.is_listening_complete():
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                # ACT 4: Sign off
                elif self.scene_step == 7 and current_time - self.step_start_time > 1:
                    print("[NAO waves enthusiastically]")
                    self.nao_speak("Coach Nao, temporarily disoriented but permanently enthusiastic, signing off!",
                                  animation="animations/Stand/Gestures/Enthusiastic_1", wait=True)
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                # Pause at door
                elif self.scene_step == 8 and current_time - self.step_start_time > 5:
                    print("[NAO turns to leave, pauses at door]")
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                elif self.scene_step == 9 and current_time - self.step_start_time > 2:
                    self.nao_speak("It's... left out of here to the elevator, correct?",
                                  wait=True)
                    self.scene_step = 10
                    self.step_start_time = current_time
                
                # Listen for "Other way, Nao. Other way."
                elif self.scene_step == 10 and current_time - self.step_start_time > 3:
                    self.start_listening()
                    self.scene_step = 101
                
                elif self.scene_step == 101 and self.is_listening_complete():
                    self.scene_step = 11
                    self.step_start_time = current_time
                
                # Final exit
                elif self.scene_step == 11 and current_time - self.step_start_time > 1:
                    self.nao_speak("Right! Of course! I knew that!",
                                  wait=True)
                    self.scene_step = 12
                    self.step_start_time = current_time
                
                elif self.scene_step == 12 and current_time - self.step_start_time > 3:
                    print("[NAO turns and exits confidently in the wrong direction]")
                    self.scene_step = 13
                    self.step_start_time = current_time
                
                elif self.scene_step == 13 and current_time - self.step_start_time > 2:
                    self.scene_step = 14
                
                elif self.scene_step == 14:
                    print("END OF SCENE 8")
                    print("\n*** PERFORMANCE COMPLETE ***\n")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()