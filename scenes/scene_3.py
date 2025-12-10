import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
from NewMotions import CoachNaoMotions
import cv2
import time


class Scene3(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=3)
        self.motions = CoachNaoMotions(nao=self.nao if self.use_nao else None)
    
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
                    
                    # Professional, focused LED state
                    if self.use_nao and self.nao:
                        self.motions.set_led_emotion("focused")
                    
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # ACT 2: Circuit explanation with professional gestures
                elif self.scene_step == 1 and current_time - self.step_start_time > 2:
                    intro = self.generate_speech(
                        "Announce the training plan. Be professional.",
                        fallback_text="Here's our training plan:"
                    )
                    
                    # Detailed explanation gesture
                    if self.use_nao and self.nao:
                        self.motions.detailed_explanation()
                    else:
                        self.nao_speak(intro, animation="animations/Stand/Gestures/Explain_1", wait=True)
                    
                    self.nao_speak(intro, wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # Exercise 1: Squats with counting gesture
                elif self.scene_step == 2 and current_time - self.step_start_time > 2:
                    squats = self.generate_speech(
                        "Explain squats are for legs and hip mobility.",
                        fallback_text="Squats for the legs and hip mobility,"
                    )
                    
                    # Gesture showing "one" - first exercise
                    if self.use_nao and self.nao:
                        self.motions.count_gesture_sequence(1)
                    
                    self.nao_speak(squats, wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # Exercise 2: Push-ups with counting gesture
                elif self.scene_step == 3 and current_time - self.step_start_time > 3:
                    pushups = self.generate_speech(
                        "Explain push-ups are for upper body strength.",
                        fallback_text="Push-ups for your upper body strength,"
                    )
                    
                    # Gesture showing "two" - second exercise
                    if self.use_nao and self.nao:
                        self.motions.count_gesture_sequence(2)
                    
                    self.nao_speak(pushups, wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # Exercise 3: Plank with counting gesture
                elif self.scene_step == 4 and current_time - self.step_start_time > 3:
                    plank = self.generate_speech(
                        "Explain plank trains core and mental endurance.",
                        fallback_text="And a Plank to train your core and mental endurance."
                    )
                    
                    # Gesture showing "three" - third exercise
                    if self.use_nao and self.nao:
                        self.motions.count_gesture_sequence(3)
                    
                    self.nao_speak(plank, wait=True)
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # ACT 3: Excitement check with enthusiastic build-up
                elif self.scene_step == 5 and current_time - self.step_start_time > 4:
                    excitement = self.generate_speech(
                        "Ask if they're excited. Be enthusiastic.",
                        fallback_text="So, are you excited already?"
                    )
                    
                    # Build excitement with LED and gesture
                    if self.use_nao and self.nao:
                        self.motions.set_led_emotion("excited")
                        time.sleep(0.3)
                    
                    self.nao_speak(excitement, animation="animations/Stand/Gestures/Enthusiastic_5", wait=True)
                    
                    # Expectant pointing gesture
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.point_forward()
                    
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
                
                # ACT 5: Let's start - climactic buildup
                elif self.scene_step == 7 and current_time - self.step_start_time > 1:
                    start = self.generate_speech(
                        "Say let's start with enthusiasm.",
                        fallback_text="Let's start then!"
                    )
                    
                    # Maximum energy gesture
                    if self.use_nao and self.nao:
                        self.motions.celebration_gesture()
                    else:
                        self.nao_speak(start, animation="animations/Stand/Gestures/Yes_1", wait=True)
                    
                    self.nao_speak(start, wait=True)
                    
                    # Confident ready stance
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.confident_presentation()
                    
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