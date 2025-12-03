import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene7(BaseScene):
    def run(self):
        print("SCENE 7: COOL-DOWN & HONEST MOMENT")
        
        try:
            while True:
                frame = self.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = self.draw_scene_info(frame, "Scene 7: Cool-Down")
                cv2.imshow("Training System Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = time.time()
                
                # ACT 1: Session complete
                if self.scene_step == 0:
                    print("[NAO LEDs glow soft blue]")
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest
                        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
                        self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0, 0.5, 0))
                        self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0, 0.5, 0))
                    
                    self.nao_speak("Session complete. Statistically, you performed at 98% efficiency. A new record!",
                                  animation="animations/Stand/Gestures/Enthusiastic_1", wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # Human sits on mat with water
                elif self.scene_step == 1 and current_time - self.step_start_time > 5:
                    print("[Human grabs water, sits on mat]")
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # Listen for "So what now? Are you going to go find your actual trainee?"
                elif self.scene_step == 2 and current_time - self.step_start_time > 3:
                    self.start_listening()
                    self.scene_step = 21
                
                elif self.scene_step == 21 and self.is_listening_complete():
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # NAO's response
                elif self.scene_step == 3 and current_time - self.step_start_time > 1:
                    print("[NAO looks toward door, then back at human]")
                    self.scene_step = 31
                    self.step_start_time = current_time
                
                elif self.scene_step == 31 and current_time - self.step_start_time > 2:
                    self.nao_speak("I should immediately proceed to apartment 4B. But this was a great learning experience.",
                                  wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                elif self.scene_step == 4 and current_time - self.step_start_time > 5:
                    self.nao_speak("Would you be open to... irregularly scheduled training sessions?",
                                  wait=True)
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # Wait a moment before ending
                elif self.scene_step == 5 and current_time - self.step_start_time > 4:
                    self.scene_step = 6
                
                elif self.scene_step == 6:
                    print("END OF SCENE 7")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()