import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene5(BaseScene):
    """Scene 5: Push-Ups - The Wrong Apartment (~2 min)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=5)
    
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
                    self.start_listening("Person might ask about demonstration or joke about your limitations.")
                    self.scene_step = 2
                
                elif self.scene_step == 2 and self.is_listening_complete():
                    self.scene_step = 3
                    self.step_start_time = current_time
                

                elif self.scene_step == 3 and current_time - self.step_start_time > 1:
                    self.nao_speak("Let's not talk about my physical limitations, you are the trainee, I am the trainer. Start with 10 pushups!",
                                  wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # Listen for agreement
                elif self.scene_step == 4 and current_time - self.step_start_time > 2:
                    self.start_listening("Person agreeing to start push-ups.")
                    self.scene_step = 5
                
                elif self.scene_step == 5 and self.is_listening_complete():
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # ACT 3: Push-up 1
                elif self.scene_step == 6 and current_time - self.step_start_time > 2:
                    print("[Person does push-up 1...]")
                    self.start_listening("Person doing first push-up, will say 'one'.")
                    self.scene_step = 61
                
                elif self.scene_step == 61 and self.is_listening_complete():
                    response = self.get_user_input()
                    if response and ("one" in response.lower() or "1" in response or "won" in response.lower()):
                        pushup_count = 1
                        print(f"[✓ Push-up #{pushup_count} detected]")
                        self.scene_step = 8
                        self.step_start_time = current_time
                    else:
                        print(f"[Did not detect 'one' in: '{response}' - asking for repeat...]")
                        
                        missed = self.generate_speech(
                            "You didn't catch push-up one. Ask them to repeat and say 'one' clearly. Sound professional but slightly awkward.",
                            fallback_text="Sorry, I didn't catch that. Push-up one again - say 'one' when you're done!"
                        )
                        self.nao_speak(missed, wait=True)
                        
                        self.scene_step = 6
                        self.step_start_time = current_time
                
                # Push-up 2
                elif self.scene_step == 8 and current_time - self.step_start_time > 1:
                    print("[Person does push-up 2...]")
                    self.start_listening("Person doing second push-up, will say 'two'.")
                    self.scene_step = 81
                
                elif self.scene_step == 81 and self.is_listening_complete():
                    response = self.get_user_input()
                    if response and ("two" in response.lower() or "2" in response or 
                                    "too" in response.lower() or "to" in response.lower() or 
                                    "do" in response.lower() or "tu" in response.lower()):
                        pushup_count = 2
                        print(f"[Push-up #{pushup_count} detected]")
                        self.scene_step = 10
                        self.step_start_time = current_time
                    else:
                        print(f"[Did not detect 'two' in: '{response}' - asking for repeat...]")
                        
                        missed = self.generate_speech(
                            "You missed push-up two. Ask them to do it again and say 'two'. Blame your sensors humorously.",
                            fallback_text="Oops, my audio sensors glitched! Push-up two again - loud and clear!"
                        )
                        self.nao_speak(missed, wait=True)
                        
                        self.scene_step = 8
                        self.step_start_time = current_time
                
                # Push-up 3
                elif self.scene_step == 10 and current_time - self.step_start_time > 1:
                    print("[Person does push-up 3...]")
                    self.start_listening("Person doing third push-up, will say 'three'.")
                    self.scene_step = 101
                
                elif self.scene_step == 101 and self.is_listening_complete():
                    response = self.get_user_input()
                    if response and ("three" in response.lower() or "3" in response or 
                                    "tree" in response.lower() or "free" in response.lower() or 
                                    "thee" in response.lower()):
                        pushup_count = 3
                        print(f"[Push-up #{pushup_count} detected]")
                        
                        self.nao_speak("Wow very smooth!", wait=True)
                        self.scene_step = 12
                        self.step_start_time = current_time
                    else:
                        print(f"[Did not detect 'three' in: '{response}' - asking for repeat...]")
                        
                        missed = self.generate_speech(
                            "You missed the third push-up. Ask them to repeat and announce 'three'.",
                            fallback_text="Wait, did you do three? My camera lagged! One more time - say 'three'!"
                        )
                        self.nao_speak(missed, wait=True)
                        
                        self.scene_step = 10
                        self.step_start_time = current_time
                
                elif self.scene_step == 12 and current_time - self.step_start_time > 1:
                    print("[Person continues push-ups 4-9...]")
                    self.start_listening("Person will count four, five, six, seven, eight, nine in sequence.")
                    self.scene_step = 121
                
                elif self.scene_step == 121 and self.is_listening_complete():
                    response = self.get_user_input()

                    if response and (("nine" in response.lower() or "9" in response or "nein" in response.lower()) or
                                    (("four" in response.lower() or "4" in response) and 
                                     ("five" in response.lower() or "5" in response))):
                        pushup_count = 9
                        print(f"[Push-ups 4-9 detected]")
                        
                        self.nao_speak("Okay one more. This performance is unprecedented in my trainee history.",
                                      wait=True)
                        self.scene_step = 14
                        self.step_start_time = current_time
                    else:
                        print(f"[Did not detect sequence 4-9 in: '{response}' - asking for repeat...]")
                        
                        missed = self.generate_speech(
                            "You didn't catch the counting. Ask them to continue from four to nine and count out loud.",
                            fallback_text="Sorry, I lost count! Continue from four to nine - count them out loud!"
                        )
                        self.nao_speak(missed, wait=True)
                        
                        self.scene_step = 12
                        self.step_start_time = current_time
                
                # ACT 5: Final push-up
                elif self.scene_step == 14 and current_time - self.step_start_time > 1:
                    print("[Person does final push-up 10...]")
                    self.start_listening("Person doing final push-up, will say 'ten'.")
                    self.scene_step = 141
                
                elif self.scene_step == 141 and self.is_listening_complete():
                    response = self.get_user_input()
                    if response and ("ten" in response.lower() or "10" in response or "hen" in response.lower()):
                        pushup_count = 10
                        print(f"[Push-up #{pushup_count} detected - COMPLETE!]")
                        
                        self.nao_speak("You are doing splendid! My other trainees are not as smooth.",
                                      wait=True)
                        self.scene_step = 16
                        self.step_start_time = current_time
                    else:
                        print(f"[✗ Did not detect 'ten' in: '{response}' - asking for repeat...]")
                        
                        missed = self.generate_speech(
                            "You missed the final tenth push-up. Ask them to do number ten and say 'ten'.",
                            fallback_text="One more! Push-up ten - the grand finale! Say 'ten' when you finish!"
                        )
                        self.nao_speak(missed, wait=True)
                        
                        self.scene_step = 14
                        self.step_start_time = current_time
                
                # ACT 6: THE REVELATION
                elif self.scene_step == 16 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will ask if you train humans or robots.")
                    self.scene_step = 17
                
                elif self.scene_step == 17 and self.is_listening_complete():
                    self.scene_step = 18
                    self.step_start_time = current_time
                
                elif self.scene_step == 18 and current_time - self.step_start_time > 1:
                    self.nao_speak("Robots, of course. Wait, are you not PR-103A25? You do have some interesting mechanical attributes.",
                                  wait=True)
                    self.scene_step = 19
                    self.step_start_time = current_time
                
                elif self.scene_step == 19 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will reveal they're human with a name.")
                    self.scene_step = 20
                
                elif self.scene_step == 20 and self.is_listening_complete():
                    self.scene_step = 21
                    self.step_start_time = current_time
                
                elif self.scene_step == 21 and current_time - self.step_start_time > 1:
                    print("[Processing beep, LEDs flash yellow]")
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest
                        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
                        self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 1, 1, 0, 0))
                        self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 1, 1, 0, 0))
                        
                        self.nao_animate("animations/Stand/Gestures/IDontKnow_1")
                    
                    self.nao_speak("Lucas? That's... not in my database. This is apartment 4B, correct?",
                                wait=True)
                    self.scene_step = 22
                    self.step_start_time = current_time

                elif self.scene_step == 22 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will say this is apartment 3B, one floor off.")
                    self.scene_step = 23

                elif self.scene_step == 23 and self.is_listening_complete():
                    self.scene_step = 24
                    self.step_start_time = current_time

                elif self.scene_step == 24 and current_time - self.step_start_time > 1:
                    print("[Long awkward processing whir...]")
                    time.sleep(2)
                    self.nao_speak("...Oh. This explains why the door was unlocked. PR-103A25 always deadbolts.",
                                wait=True)
                    self.scene_step = 25
                    self.step_start_time = current_time

                elif self.scene_step == 25 and current_time - self.step_start_time > 5:
                    self.nao_speak("Well... your form is excellent. Would you... like me to leave?",
                                animation="animations/Stand/Gestures/Please_1",
                                wait=True)
                    self.scene_step = 26
                    self.step_start_time = current_time
                
                elif self.scene_step == 26 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will say you're already here, might as well finish.")
                    self.scene_step = 27
                
                elif self.scene_step == 27 and self.is_listening_complete():
                    self.scene_step = 28
                    self.step_start_time = current_time
                
                
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