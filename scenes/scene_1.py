import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
from NewMotions import CoachNaoMotions
import cv2
import time


class Scene1(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=1)
        self.motions = CoachNaoMotions(nao=self.nao if self.use_nao else None)
    
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
                
                # OPENING - Enthusiastic entrance
                if self.scene_step == 0:
                    # Set excited LED state
                    self.motions.set_led_emotion("excited")
                    
                    # Wave and greeting
                    self.nao_speak("Hey there you are, almost did not see you!",
                                  animation="animations/Stand/Gestures/Hey_1", wait=True)
                    
                    # Add a small awkward pause after wave
                    time.sleep(0.5)
                    if self.use_nao and self.nao:
                        self.motions.awkward_wave()
                    
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # CONVERSATIONAL
                elif self.scene_step == 1 and current_time - self.step_start_time > 3:
                    self.start_listening("Person might ask who you are or who let you in.")
                    self.scene_step = 11
                
                # CONVERSATIONAL RESPONSE with gesture
                elif self.scene_step == 11 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        # Confident self-reference gesture
                        if self.use_nao and self.nao:
                            self.motions.self_reference()
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # NARRATIVE BEAT - Mission statement with confident presentation
                elif self.scene_step == 2 and current_time - self.step_start_time > 4:
                    mission = self.generate_speech(
                        "Tell them you'll guide them through the most intense training session they'll ever do.",
                        fallback_text="I will be your trainer today and guide you through the most intense session you will ever be doing."
                    )
                    
                    # Confident trainer stance
                    if self.use_nao and self.nao:
                        self.motions.confident_presentation()
                    
                    self.nao_speak(mission, wait=True)
                    self.scene_step = 22
                    self.step_start_time = current_time
                
                # ANIMATION - Fist pump / enthusiastic gesture
                elif self.scene_step == 22 and current_time - self.step_start_time > 5:
                    if self.use_nao and self.nao:
                        self.motions.celebration_gesture()
                    else:
                        self.nao_animate("animations/Stand/Gestures/Yes_1")
                    
                    self.scene_step = 23
                    self.step_start_time = current_time
                
                # NARRATIVE BEAT - Camera positioning with pointing
                elif self.scene_step == 23 and current_time - self.step_start_time > 2:
                    camera_instruction = self.generate_speech(
                        "Ask them to stand in front of you so you can observe their movements.",
                        fallback_text="Okay, let's make sure I can observe your movements. Please stand in front of me so I can detect your movements today."
                    )
                    
                    # Point forward to indicate where they should stand
                    if self.use_nao and self.nao:
                        self.motions.point_forward()
                    
                    self.nao_speak(camera_instruction, wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # CONVERSATIONAL - Person agrees
                elif self.scene_step == 3 and current_time - self.step_start_time > 6:
                    self.start_listening("Person is agreeing to train with you.")
                    self.scene_step = 31
                
                # CONVERSATIONAL RESPONSE with encouraging gesture
                elif self.scene_step == 31 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        # Encouraging nod
                        if self.use_nao and self.nao:
                            self.motions.encouraging_nod()
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # TRANSITION - Ready check with enthusiastic gesture
                elif self.scene_step == 4 and current_time - self.step_start_time > 3:
                    ready_check = self.generate_speech(
                        "Say 'Perfect' to acknowledge they're positioned. Then ask if they're ready for round one.",
                        fallback_text="Perfect. Ready for round one?"
                    )
                    
                    # Set focused LED state
                    if self.use_nao and self.nao:
                        self.motions.set_led_emotion("focused")
                    
                    self.nao_speak(ready_check,
                                  animation="animations/Stand/Gestures/Enthusiastic_4", 
                                  wait=True)
                    
                    # Add an expectant pointing gesture
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.point_forward()
                    
                    self.scene_step = 6
                    self.step_start_time = current_time
                 
                elif self.scene_step == 6 and current_time - self.step_start_time > 3:
                    self.start_listening("Person confirming they're ready.")
                    self.scene_step = 61
                
                elif self.scene_step == 61 and self.is_listening_complete():
                    # Final affirmative gesture
                    if self.use_nao and self.nao:
                        self.motions.thumbs_up_equivalent()
                    
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                elif self.scene_step == 7 and current_time - self.step_start_time > 1:
                    print("END OF SCENE 1")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()