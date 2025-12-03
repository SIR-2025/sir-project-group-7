import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene5(BaseScene):
    """Scene 5: Push-Ups - The Wrong Apartment (~2 min)"""
    
    def run(self):
        print("\n" + "="*70)
        print("SCENE 5: PUSH-UPS - THE REVELATION")
        print("="*70)
        
        try:
            pushup_count = 0
            target_pushups = 10
            
            while True:
                frame = self.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = self.draw_scene_info(frame, f"Scene 5: Push-ups ({pushup_count}/{target_pushups})")
                cv2.imshow("Training System Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = time.time()
                
                # ACT 1: Introduction
                if self.scene_step == 0:
                    print("[NAO leans forward playfully]")
                    self.nao_speak("Okay. Get into push-up position. I don't think you need a demonstration this time.",
                                  animation="animations/Stand/Gestures/Explain_1", wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # Listen for person's question about demonstration
                elif self.scene_step == 1 and current_time - self.step_start_time > 2:
                    self.start_listening()
                    self.scene_step = 2
                
                elif self.scene_step == 2 and self.is_listening_complete():
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # ACT 2: Deflect and start exercise
                elif self.scene_step == 3 and current_time - self.step_start_time > 1:
                    self.nao_speak("Let's not talk about my physical limitations, you are the trainee, I am the trainer. Start with 10 pushups!",
                                  wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # Listen for agreement
                elif self.scene_step == 4 and current_time - self.step_start_time > 2:
                    self.start_listening()
                    self.scene_step = 5
                
                elif self.scene_step == 5 and self.is_listening_complete():
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # ACT 3: Count push-ups 1, 2, 3
                # Person does three pushups, after each push says the count
                elif self.scene_step == 6 and current_time - self.step_start_time > 2:
                    print("[Person does push-up 1...]")
                    self.set_leds_listening()
                    speech_detected = self.dialogue_manager.listen_for_any_speech(max_duration=5.0, silence_threshold=0.04)
                    self.set_leds_thinking()
                    if speech_detected:
                        pushup_count = 1
                        print(f"[Push-up #1 counted]")
                        self.scene_step = 8
                    else:
                        print("[No speech detected, retrying...]")
                        self.scene_step = 6
                    self.step_start_time = current_time
                
                # Push-up 2
                elif self.scene_step == 8 and current_time - self.step_start_time > 1:
                    print("[Person does push-up 2...]")
                    self.set_leds_listening()
                    speech_detected = self.dialogue_manager.listen_for_any_speech(max_duration=5.0, silence_threshold=0.04)
                    self.set_leds_thinking()
                    if speech_detected:
                        pushup_count = 2
                        print(f"[Push-up #2 counted]")
                        self.scene_step = 10
                    else:
                        print("[No speech detected, retrying...]")
                        self.scene_step = 8
                    self.step_start_time = current_time
                
                # Push-up 3
                elif self.scene_step == 10 and current_time - self.step_start_time > 1:
                    print("[Person does push-up 3...]")
                    self.set_leds_listening()
                    speech_detected = self.dialogue_manager.listen_for_any_speech(max_duration=5.0, silence_threshold=0.04)
                    self.set_leds_thinking()
                    if speech_detected:
                        pushup_count = 3
                        print(f"[Push-up #3 counted]")
                        # Robot responds: "Wow very smooth!"
                        self.nao_speak("Wow very smooth!", wait=True)
                        self.scene_step = 12
                    else:
                        print("[No speech detected, retrying...]")
                        self.scene_step = 10
                    self.step_start_time = current_time
                
                # ACT 4: Person keeps going with count until 9
                # Count push-ups 4, 5, 6, 7, 8, 9
                elif self.scene_step == 12 and current_time - self.step_start_time > 1:
                    print("[Person continues push-ups 4-9...]")
                    self.set_leds_listening()
                    speech_detected = self.dialogue_manager.listen_for_any_speech(max_duration=10.0, silence_threshold=0.04)
                    self.set_leds_thinking()
                    if speech_detected:
                        pushup_count = 9
                        print(f"[Push-ups 4-9 counted]")
                        # Robot: "Okay one more. This performance is unprecedented in my trainee history"
                        self.nao_speak("Okay one more. This performance is unprecedented in my trainee history.",
                                      wait=True)
                        self.scene_step = 14
                    else:
                        print("[No speech detected, retrying...]")
                        self.scene_step = 12
                    self.step_start_time = current_time
                
                # ACT 5: Final push-up (10)
                elif self.scene_step == 14 and current_time - self.step_start_time > 1:
                    print("[Person does final push-up 10...]")
                    self.set_leds_listening()
                    speech_detected = self.dialogue_manager.listen_for_any_speech(max_duration=5.0, silence_threshold=0.04)
                    self.set_leds_thinking()
                    if speech_detected:
                        pushup_count = 10
                        print(f"[Push-up #10 counted]")
                        # Robot: "You are doing splendid! My other trainees are not as smooth."
                        self.nao_speak("You are doing splendid! My other trainees are not as smooth.",
                                      wait=True)
                        self.scene_step = 16
                    else:
                        print("[No speech detected, retrying...]")
                        self.scene_step = 14
                    self.step_start_time = current_time
                
                # ACT 6: The revelation - "Do you train humans or robots?"
                elif self.scene_step == 16 and current_time - self.step_start_time > 4:
                    self.start_listening()
                    self.scene_step = 17
                
                elif self.scene_step == 17 and self.is_listening_complete():
                    self.scene_step = 18
                    self.step_start_time = current_time
                
                # "Robots, of course..."
                elif self.scene_step == 18 and current_time - self.step_start_time > 1:
                    self.nao_speak("Robots, of course. Wait, are you not PR-103A25? You do have some interesting mechanical attributes.",
                                  wait=True)
                    self.scene_step = 19
                    self.step_start_time = current_time
                
                # Listen for "Um, no... my name is Lucas"
                elif self.scene_step == 19 and current_time - self.step_start_time > 5:
                    self.start_listening()
                    self.scene_step = 20
                
                elif self.scene_step == 20 and self.is_listening_complete():
                    self.scene_step = 21
                    self.step_start_time = current_time
                
                # "Lucas? That's... not in my database..."
                elif self.scene_step == 21 and current_time - self.step_start_time > 1:
                    print("[Processing beep, LEDs flash yellow]")
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest
                        # Flash yellow
                        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
                        self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 1, 1, 0, 0))  # Yellow
                        self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 1, 1, 0, 0))   # Yellow
                    
                    self.nao_speak("Lucas? That's... not in my database. This is apartment 4B, correct?",
                                  wait=True)
                    self.scene_step = 22
                    self.step_start_time = current_time
                
                # Listen for "This is 3B. You're one floor off."
                elif self.scene_step == 22 and current_time - self.step_start_time > 4:
                    self.start_listening()
                    self.scene_step = 23
                
                elif self.scene_step == 23 and self.is_listening_complete():
                    self.scene_step = 24
                    self.step_start_time = current_time
                
                # Awkward realization
                elif self.scene_step == 24 and current_time - self.step_start_time > 1:
                    print("[Long awkward processing whir...]")
                    time.sleep(2)
                    self.nao_speak("...Oh. This explains why the door was unlocked. PR-103A25 always deadbolts.",
                                  wait=True)
                    self.scene_step = 25
                    self.step_start_time = current_time
                
                elif self.scene_step == 25 and current_time - self.step_start_time > 5:
                    self.nao_speak("Well... your form is excellent. Would you... like me to leave?",
                                  wait=True)
                    self.scene_step = 26
                    self.step_start_time = current_time
                
                # Listen for "You know what? You're already here..."
                elif self.scene_step == 26 and current_time - self.step_start_time > 4:
                    self.start_listening()
                    self.scene_step = 27
                
                elif self.scene_step == 27 and self.is_listening_complete():
                    self.scene_step = 28
                    self.step_start_time = current_time
                
                # Relief and continuation
                elif self.scene_step == 28 and current_time - self.step_start_time > 1:
                    print("[Relieved beep]")
                    self.nao_speak("Okay that sounds good Lucas of 3B. Let's move to the final challenge!",
                                  animation="animations/Stand/Gestures/Enthusiastic_1", wait=True)
                    self.scene_step = 29
                    self.step_start_time = current_time
                
                elif self.scene_step == 29 and current_time - self.step_start_time > 4:
                    self.scene_step = 30
                
                elif self.scene_step == 30:
                    print("\n" + "="*70)
                    print("END OF SCENE 5")
                    print("="*70 + "\n")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()