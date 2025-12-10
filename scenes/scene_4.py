import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
from NewMotions import CoachNaoMotions
import cv2
import time


class Scene4(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=4)
        self.motions = CoachNaoMotions(nao=self.nao if self.use_nao else None)
    
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
                
                # ACT 1: Introduction with professional explanation
                if self.scene_step == 0:
                    intro = self.generate_speech(
                        "Announce the first exercise is the squat. Call it a biomechanical marvel.",
                        fallback_text="Excellent. Our first exercise is the biomechanical marvel known as the... squat."
                    )
                    
                    # Detailed explanation gesture
                    if self.use_nao and self.nao:
                        self.motions.detailed_explanation()
                    else:
                        self.nao_speak(intro, animation="animations/Stand/Gestures/Explain_1", wait=True)
                    
                    self.nao_speak(intro, wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # Warning about demonstration with self-awareness
                elif self.scene_step == 1 and current_time - self.step_start_time > 5:
                    demo_warning = self.generate_speech(
                        "Tell them you'll demonstrate but you're not as flexible as them.",
                        fallback_text="I will now demonstrate a poor representation of it because I'm not as flexible as you."
                    )
                    
                    # Self-deprecating gesture
                    if self.use_nao and self.nao:
                        self.motions.self_reference()
                        time.sleep(0.5)
                        self.motions.embarrassed_look_away()
                    
                    self.nao_speak(demo_warning, wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # ACT 2: NAO attempts squat - use custom sequence
                elif self.scene_step == 2 and current_time - self.step_start_time > 4:
                    print("[NAO ATTEMPTS 'SQUAT' - SITS DOWN]")
                    
                    if self.use_nao and self.nao:
                        # Use the custom attempt_squat method
                        self.motions.attempt_squat()
                    else:
                        time.sleep(2)
                    
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # ACT 3: Request squats with pointing
                elif self.scene_step == 3 and current_time - self.step_start_time > 2:
                    request = self.generate_speech(
                        "Admit that's the best you can do but they can do better. Request three squats.",
                        fallback_text="So this is the best I can do but you can surely do it better, so please perform three squats."
                    )
                    
                    # Point at trainee expectantly
                    if self.use_nao and self.nao:
                        self.motions.point_forward()
                    
                    self.nao_speak(request, wait=True)
                    
                    # Show "three" with gesture
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.count_gesture_sequence(3)
                    
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # ACT 4: Wait for first squat
                elif self.scene_step == 4 and current_time - self.step_start_time > 5:
                    self.start_listening("Person might comment on your squat, then do their first squat and say 'One'.")
                    self.scene_step = 41
                
                elif self.scene_step == 41 and self.is_listening_complete():
                    response = self.get_user_input()
                    if response and ("one" in response.lower() or "1" in response or "won" in response.lower()):
                        squat_count = 1
                        print(f"[Squat #{squat_count} detected]")
                        
                        # Encouraging gesture
                        if self.use_nao and self.nao:
                            self.motions.encouraging_nod()
                        
                        self.nao_speak("Nice, good attempt!", wait=True)
                        self.scene_step = 5
                        self.step_start_time = current_time
                    else:
                        print(f"[Did not detect 'one' in: '{response}' - asking for repeat...]")
                        
                        # Confused gesture
                        if self.use_nao and self.nao:
                            self.motions.confused_shrug()
                        
                        missed = self.generate_speech(
                            "You didn't see the squat properly. Ask them to repeat squat number one and say 'one' when done. Be slightly awkward but professional.",
                            fallback_text="Sorry, I didn't quite catch that. Can you repeat squat one and say 'one' when you're done?"
                        )
                        self.nao_speak(missed, wait=True)
                        
                        self.scene_step = 4
                        self.step_start_time = current_time
                
                # ACT 5: Wait for second squat - "Two!"
                elif self.scene_step == 5 and current_time - self.step_start_time > 3:
                    self.start_listening("Person doing second squat, will say 'Two'.")
                    self.scene_step = 51
                
                elif self.scene_step == 51 and self.is_listening_complete():
                    response = self.get_user_input()
                    if response and ("two" in response.lower() or "2" in response or 
                                    "too" in response.lower() or "to" in response.lower() or 
                                    "do" in response.lower() or "tu" in response.lower()):
                        squat_count = 2
                        print(f"[Squat #{squat_count} detected]")
                        
                        # More enthusiastic gesture - they're improving!
                        if self.use_nao and self.nao:
                            self.motions.count_gesture_sequence(2)
                        
                        self.nao_speak("Wow what an improvement compared to mine!", wait=True)
                        
                        # Self-deprecating follow-up
                        if self.use_nao and self.nao:
                            time.sleep(0.3)
                            self.motions.defeated_slump()
                        
                        self.scene_step = 6
                        self.step_start_time = current_time
                    else:
                        print(f"[Did not detect 'two' in: '{response}' - asking for repeat...]")
                        
                        if self.use_nao and self.nao:
                            self.motions.embarrassed_look_away()
                        
                        missed = self.generate_speech(
                            "You didn't catch squat number two. Ask them to do it again and say 'two' when done. Sound slightly embarrassed about missing it.",
                            fallback_text="Oops, my sensors must have glitched. Can you do squat two again and say 'two' clearly?"
                        )
                        self.nao_speak(missed, wait=True)
                        
                        self.scene_step = 5
                        self.step_start_time = current_time
                
                # ACT 6: Wait for third squat - "Three!"
                elif self.scene_step == 6 and current_time - self.step_start_time > 3:
                    self.start_listening("Person doing third squat, will say 'Three'.")
                    self.scene_step = 61

                elif self.scene_step == 61 and self.is_listening_complete():
                    response = self.get_user_input()

                    if response and ("three" in response.lower() or "3" in response or 
                                    "tree" in response.lower() or "free" in response.lower() or 
                                    "thee" in response.lower()):
                        squat_count = 3
                        print(f"[Squat #{squat_count} detected]")
                        
                        # Big celebration!
                        if self.use_nao and self.nao:
                            self.motions.encouragement_sequence()
                        else:
                            self.nao_animate("animations/Stand/Negation/NAO/Center_Neutral_NEG_04")
                        
                        self.nao_speak("Great job! You look like you do this quite often. Sometimes I question why I do this.", 
                                    wait=True)
                        
                        # Reflective, self-aware pause
                        if self.use_nao and self.nao:
                            time.sleep(0.5)
                            self.motions.reflective_pause()
                        
                        self.scene_step = 7
                        self.step_start_time = current_time
                    else:
                        print(f"[Did not detect 'three' in: '{response}' - retrying...]")
                        
                        if self.use_nao and self.nao:
                            self.motions.pleading_gesture()
                        
                        missed = self.generate_speech(
                            "You missed the final squat. Ask them to repeat squat three and announce 'three' when finished. Act frustrated with your own sensors.",
                            fallback_text="My bad! I think I blinked. One more time - squat three, and shout 'three' when you finish!"
                        )
                        self.nao_speak(missed, wait=True)
                        
                        self.scene_step = 6
                        self.step_start_time = current_time
                
                # ACT 7: Listen for trainee question
                elif self.scene_step == 7 and current_time - self.step_start_time > 5:
                    self.start_listening("Person might ask a question about breaking in or training.")
                    self.scene_step = 71
                
                elif self.scene_step == 71 and self.is_listening_complete():
                    # Awkward response gesture
                    if self.use_nao and self.nao:
                        self.motions.embarrassed_look_away()
                    
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                # ACT 8: Move on to push-ups
                elif self.scene_step == 8 and current_time - self.step_start_time > 1:
                    # HARDCODED to ensure clean transition to Scene 5
                    self.nao_speak("Uhmmm... let's move on. How about we do some push-ups now, are you ready?", wait=True)
                    
                    # Redirect with pointing gesture
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.point_forward()
                        self.motions.set_led_emotion("focused")
                    
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                # Listen for agreement
                elif self.scene_step == 9 and current_time - self.step_start_time > 4:
                    self.start_listening("Person agreeing to push-ups.")
                    self.scene_step = 91
                
                # Don't respond
                elif self.scene_step == 91 and self.is_listening_complete():
                    # Final ready gesture
                    if self.use_nao and self.nao:
                        self.motions.thumbs_up_equivalent()
                    
                    self.scene_step = 10
                
                elif self.scene_step == 10:
                    print("END OF SCENE 4")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()