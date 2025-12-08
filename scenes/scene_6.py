import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene6(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=6)
    
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
                

                if self.scene_step == 0:
                    intro = self.generate_speech(
                        "Announce the final challenge: the plank! Tell them to get into position. You'll count down from 30 seconds.",
                        fallback_text="The final challenge! The plank! Get into position! I will count down from 30 seconds!"
                    )
                    self.nao_speak(intro, animation="animations/Stand/Gestures/Enthusiastic_1", wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                elif self.scene_step == 1 and current_time - self.step_start_time > 5:
                    start_count = self.generate_speech(
                        "Say 'AND HOLD!' Start counting: 30, 29, 28. Compliment their perfect form.",
                        fallback_text="AND... HOLD! 30... 29... 28... Your form is perfect!"
                    )
                    self.nao_speak(start_count, wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # ACT 2: Count down
                elif self.scene_step == 2 and current_time - self.step_start_time > 5:
                    count1 = self.generate_speech(
                        "Count: 25, 24, 23. Tell them to keep it steady.",
                        fallback_text="25... 24... 23... Keep it steady!"
                    )
                    self.nao_speak(count1, wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                elif self.scene_step == 3 and current_time - self.step_start_time > 4:
                    count2 = self.generate_speech(
                        "Count: 20, 19, 18. Encourage them - they're doing great.",
                        fallback_text="20... 19... 18... You're doing great!"
                    )
                    self.nao_speak(count2, wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                elif self.scene_step == 4 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will ask if getting lost happens often.")
                    self.scene_step = 41
                
                elif self.scene_step == 41 and self.is_listening_complete():
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                elif self.scene_step == 5 and current_time - self.step_start_time > 1:
                    deflect = self.generate_speech(
                        "Say '15 SECONDS!' Deflect - tell them to focus on core, not your operational flaws.",
                        fallback_text="15 SECONDS! Let's focus on your core, not my operational flaws!"
                    )
                    self.nao_speak(deflect, wait=True)
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                elif self.scene_step == 6 and current_time - self.step_start_time > 2:
                    confession = self.generate_speech(
                        "Count: 14, 13. Admit this is only the third time this week.",
                        fallback_text="14... 13... Though statistically, this is only the third time this week!"
                    )
                    self.nao_speak(confession, wait=True)
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                elif self.scene_step == 7 and current_time - self.step_start_time > 4:
                    count3 = self.generate_speech(
                        "Count: 12, 11.",
                        fallback_text="12... 11..."
                    )
                    self.nao_speak(count3, wait=True)
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                elif self.scene_step == 8 and current_time - self.step_start_time > 2:
                    self.start_listening("Person might laugh or react to the third time admission.")
                    self.scene_step = 81
                
                elif self.scene_step == 81 and self.is_listening_complete():
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                # Final countdown
                elif self.scene_step == 9 and current_time - self.step_start_time > 1:
                    final_count1 = self.generate_speech(
                        "Count: 10, 9, 8. Say 'FOCUS Lucas of 3B!' Deflect from your lack of spatial awareness.",
                        fallback_text="10... 9... 8... FOCUS Lucas of 3B! Let's not focus on my lack of spatial awareness!"
                    )
                    self.nao_speak(final_count1, wait=True)
                    self.scene_step = 10
                    self.step_start_time = current_time
                
                elif self.scene_step == 10 and current_time - self.step_start_time > 5:
                    final_count2 = self.generate_speech(
                        "Count: 7, 6, 5.",
                        fallback_text="7... 6... 5..."
                    )
                    self.nao_speak(final_count2, wait=True)
                    self.scene_step = 11
                    self.step_start_time = current_time
                
                elif self.scene_step == 11 and current_time - self.step_start_time > 3:
                    finish = self.generate_speech(
                        "Count: 4, 3, 2, 1. Say TIME!",
                        fallback_text="4... 3... 2... 1... TIME!"
                    )
                    self.nao_speak(finish, wait=True)
                    self.scene_step = 12
                    self.step_start_time = current_time
                
                # Conclusion
                elif self.scene_step == 12 and current_time - self.step_start_time > 3:
                    conclusion = self.generate_speech(
                        "Say excellent! They maintained form despite your distracting personal failures.",
                        fallback_text="Excellent! You maintained form despite my distracting personal failures!"
                    )
                    self.nao_speak(conclusion, animation="animations/Stand/Gestures/Yes_1", wait=True)
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