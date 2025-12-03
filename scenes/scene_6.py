import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene6(BaseScene):
    def run(self):
        print("\n" + "="*70)
        print("SCENE 6: PLANK ENDURANCE")
        print("="*70)
        
        try:
            plank_timer = 30
            
            while True:
                frame = self.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Calculate remaining plank time
                if self.scene_step >= 2 and self.scene_step < 10:
                    elapsed = time.time() - self.step_start_time
                    remaining = max(0, plank_timer - int(elapsed))
                    frame = self.draw_scene_info(frame, f"Scene 6: Plank ({remaining}s)")
                else:
                    frame = self.draw_scene_info(frame, "Scene 6: Plank")
                
                cv2.imshow("Training System Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = time.time()
                
                # ACT 1: Introduction
                if self.scene_step == 0:
                    self.nao_speak("The final challenge! The plank! Get into position! I will count down from 30 seconds!",
                                  animation="animations/Stand/Gestures/Enthusiastic_1", wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # Wait for person to get into position
                elif self.scene_step == 1 and current_time - self.step_start_time > 5:
                    self.nao_speak("AND... HOLD! 30... 29... 28... Your form is perfect!",
                                  wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # ACT 2: Count down from 30 to 16 (15 seconds elapsed)
                elif self.scene_step == 2 and current_time - self.step_start_time > 5:
                    self.nao_speak("25... 24... 23... Keep it steady!",
                                  wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                elif self.scene_step == 3 and current_time - self.step_start_time > 4:
                    self.nao_speak("20... 19... 18... You're doing great!",
                                  wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # At 15 seconds - HUMAN SPEAKS FIRST: "So... does this happen often?"
                elif self.scene_step == 4 and current_time - self.step_start_time > 4:
                    self.start_listening()
                    self.scene_step = 41
                
                elif self.scene_step == 41 and self.is_listening_complete():
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # NAO responds with "15 SECONDS!"
                elif self.scene_step == 5 and current_time - self.step_start_time > 1:
                    self.nao_speak("15 SECONDS! Let's focus on your core, not my operational flaws!",
                                  wait=True)
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # Continue counting
                elif self.scene_step == 6 and current_time - self.step_start_time > 2:
                    self.nao_speak("14... 13... Though statistically, this is only the third time this week!",
                                  wait=True)
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                elif self.scene_step == 7 and current_time - self.step_start_time > 4:
                    self.nao_speak("12... 11...",
                                  wait=True)
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                # Listen for person's laugh about "Third time?!"
                elif self.scene_step == 8 and current_time - self.step_start_time > 2:
                    self.start_listening()
                    self.scene_step = 81
                
                elif self.scene_step == 81 and self.is_listening_complete():
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                # Final countdown
                elif self.scene_step == 9 and current_time - self.step_start_time > 1:
                    self.nao_speak("10... 9... 8... FOCUS Lucas of 3B! Let's not focus on my lack of spatial awareness!",
                                  wait=True)
                    self.scene_step = 10
                    self.step_start_time = current_time
                
                elif self.scene_step == 10 and current_time - self.step_start_time > 5:
                    self.nao_speak("7... 6... 5...",
                                  wait=True)
                    self.scene_step = 11
                    self.step_start_time = current_time
                
                elif self.scene_step == 11 and current_time - self.step_start_time > 3:
                    self.nao_speak("4... 3... 2... 1... TIME!",
                                  wait=True)
                    self.scene_step = 12
                    self.step_start_time = current_time
                
                # Conclusion
                elif self.scene_step == 12 and current_time - self.step_start_time > 3:
                    self.nao_speak("Excellent! You maintained form despite my distracting personal failures!",
                                  animation="animations/Stand/Gestures/Yes_1", wait=True)
                    self.scene_step = 13
                    self.step_start_time = current_time
                
                elif self.scene_step == 13 and current_time - self.step_start_time > 4:
                    self.scene_step = 14
                
                elif self.scene_step == 14:
                    print("\n" + "="*70)
                    print("END OF SCENE 6")
                    print("="*70 + "\n")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()