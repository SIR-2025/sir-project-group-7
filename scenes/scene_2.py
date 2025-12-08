import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene2(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=2)
    
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
                    arms_instruction = self.generate_speech(
                        "Tell them to raise their arms and make a joke about your limited robot shoulder mobility.",
                        fallback_text="First: Raise your arms like me... or higher if you have human shoulders."
                    )
                    self.nao_speak(arms_instruction, animation="animations/Stand/Gestures/Arms_1", wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # ACT 3: Leg shakes
                elif self.scene_step == 2 and current_time - self.step_start_time > 5:
                    legs_instruction = self.generate_speech(
                        "Tell them to shake their legs. Joke about how your robot legs just vibrate.",
                        fallback_text="Shake your legs, or do whatever humans do when they loosen up. Mine just... vibrate a bit."
                    )
                    self.nao_speak(legs_instruction, animation="animations/Stand/BodyTalk/BodyTalk_1", wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # ACT 4: Listen for sarcastic comment
                elif self.scene_step == 3 and current_time - self.step_start_time > 6:
                    self.start_listening("Person might make sarcastic comment about your limitations. React with confidence.")
                    self.scene_step = 31
                
                elif self.scene_step == 31 and self.is_listening_complete():
                    user_said = self.get_user_input()
                    
                    if user_said and len(user_said) > 3:
                        gpt_response = self.get_gpt_response()
                        if gpt_response:
                            self.nao_speak(gpt_response, wait=True)
                            time.sleep(1)
                    
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # ACT 5: Stretch and feeling warm
                elif self.scene_step == 4 and current_time - self.step_start_time > 1:
                    stretch_instruction = self.generate_speech(
                        "Tell them to stretch left. Joke about your limited range of motion. Ask if they're feeling warm and ready for real work.",
                        fallback_text="Let's stretch left! My body can only go... this far. Respect my angles. Feeling warm already? Because then we can start with the real work, or at least you. Ha-ha-ha!"
                    )
                    self.nao_speak(stretch_instruction, animation="animations/Stand/Gestures/Explain_1", wait=True)
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # ACT 6: Listen for final comment
                elif self.scene_step == 5 and current_time - self.step_start_time > 8:
                    self.start_listening("Person responding about being ready for the workout.")
                    self.scene_step = 51
                
                elif self.scene_step == 51 and self.is_listening_complete():
                    user_said = self.get_user_input()
                    
                    if user_said:
                        gpt_response = self.get_gpt_response()
                        if gpt_response:
                            self.nao_speak(gpt_response, wait=False)
                    
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # ACT 7: Closing
                elif self.scene_step == 6 and current_time - self.step_start_time > 2:
                    self.scene_step = 7
                
                elif self.scene_step == 7:
                    print("END OF SCENE 2")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()