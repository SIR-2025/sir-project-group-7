import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene4(BaseScene):
    def run(self):
        print("SCENE 4: SQUATS - THE BONDING BEGINS")
        
        try:
            squat_count = 0
            target_squats = 3
            
            while True:
                frame = self.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = self.draw_scene_info(frame, f"Scene 4: Squats ({squat_count}/{target_squats})")
                cv2.imshow("Training System Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = time.time()
                
                # ACT 1: Introduction
                if self.scene_step == 0:
                    self.nao_speak("Excellent. Our first exercise is the biomechanical marvel known as the... squat.",
                                  animation="animations/Stand/Gestures/Explain_1", wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                elif self.scene_step == 1 and current_time - self.step_start_time > 5:
                    self.nao_speak("I will now demonstrate a poor representation of it because I'm not as flexible as you.",
                                  wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # ACT 2: NAO attempts squat (sitting motion)
                elif self.scene_step == 2 and current_time - self.step_start_time > 4:
                    # NAO sits down (squat attempt)
                    print("[NAO SITS DOWN - 'squat attempt']")
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
                        self.nao.motion.request(NaoPostureRequest("Sit", 0.8))
                    else:
                        time.sleep(2)
                    self.scene_step = 21
                    self.step_start_time = current_time
                
                elif self.scene_step == 21 and current_time - self.step_start_time > 3:
                    # NAO stands back up
                    print("[NAO STANDS BACK UP]")
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
                        self.nao.motion.request(NaoPostureRequest("Stand", 0.8))
                    else:
                        time.sleep(2)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # ACT 3: Request squats
                elif self.scene_step == 3 and current_time - self.step_start_time > 2:
                    self.nao_speak("So this is the best I can do but you can surely do it better, so please perform three squats.",
                                  wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # ACT 4: Wait for first squat
                # Person says "That was not a great squat but I appreciate the effort Mr. Nao. (performs squat) One!"
                elif self.scene_step == 4 and current_time - self.step_start_time > 5:
                    self.start_listening()
                    self.scene_step = 41
                
                elif self.scene_step == 41 and self.is_listening_complete():
                    response = self.get_listen_result()
                    # Listen for full response or just "one"
                    if response and ("one" in response.lower() or "1" in response):
                        squat_count = 1
                        self.nao_speak("Nice, good attempt!", wait=True)
                        self.scene_step = 5
                    self.step_start_time = current_time
                
                # ACT 5: Wait for second squat - "Two!"
                elif self.scene_step == 5 and current_time - self.step_start_time > 3:
                    self.start_listening()
                    self.scene_step = 51
                
                elif self.scene_step == 51 and self.is_listening_complete():
                    response = self.get_listen_result()
                    if response and ("two" in response.lower() or "2" in response):
                        squat_count = 2
                        self.nao_speak("Wow what an improvement compared to mine!",
                                      wait=True)
                        self.scene_step = 6
                    self.step_start_time = current_time
                
                # ACT 6: Wait for third squat - "Three!"
                elif self.scene_step == 6 and current_time - self.step_start_time > 3:
                    self.start_listening()
                    self.scene_step = 61
                
                elif self.scene_step == 61 and self.is_listening_complete():
                    response = self.get_listen_result()
                    if response and ("three" in response.lower() or "3" in response):
                        squat_count = 3
                        self.nao_speak("Great job! You look like you do this quite often. Sometimes I question why I do this.",
                                      wait=True)
                        self.scene_step = 7
                    self.step_start_time = current_time
                
                # ACT 7: Listen for trainee question
                elif self.scene_step == 7 and current_time - self.step_start_time > 5:
                    self.start_listening()
                    self.scene_step = 71
                
                elif self.scene_step == 71 and self.is_listening_complete():
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                # ACT 8: Move on to push-ups
                elif self.scene_step == 8 and current_time - self.step_start_time > 1:
                    self.nao_speak("Uhmmm... let's move on. How about we do some push-ups now, are you ready?",
                                  wait=True)
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                # Listen for agreement
                elif self.scene_step == 9 and current_time - self.step_start_time > 4:
                    self.start_listening()
                    self.scene_step = 91
                
                elif self.scene_step == 91 and self.is_listening_complete():
                    self.scene_step = 10
                
                elif self.scene_step == 10:
                    print("END OF SCENE 4")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()