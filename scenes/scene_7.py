import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene7(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=7)
    
    def run(self):
        print("\n" + "="*70)
        print("SCENE 7: THE WALK TO THE RIGHT APARTMENT (FINALE)")
        print("="*70)
        
        try:
            while True:
                frame = self.get_display_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                frame = self.draw_scene_info(frame, "Scene 7: The Walk (Finale)")
                cv2.imshow("Training System Camera", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                current_time = time.time()
                
                # STEP 0: Session complete
                if self.scene_step == 0:
                    print("[NAO LEDs glow soft, calm blue]")
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest
                        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
                        self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0, 0.5, 0))
                        self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0, 0.5, 0))
                    
                    self.nao_speak("Session complete. Statistically, you performed at 98% efficiency. A new personal record.", wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # STEP 1: Human grabs water, then speaks
                elif self.scene_step == 1 and current_time - self.step_start_time > 5:
                    print("[Human grabs water bottle, sits with amused smile]")
                    self.start_listening("Person will ask what happens now.")
                    self.scene_step = 11
                
                # STEP 11: Wait for listening to complete
                elif self.scene_step == 11 and self.is_listening_complete():
                    # Ignore GPT response, just continue to next step
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # STEP 2: NAO admits should go to 4B, points wrong direction
                elif self.scene_step == 2 and current_time - self.step_start_time > 5:
                    print("[NAO glances at door, then back at Human, slightly embarrassed]")
                    
                    self.nao_speak("Yes. I should report to apartment 4B immediately. My internal navigation insists it is… that way.", wait=True)
                    
                    print("[NAO points confidently in the WRONG direction]")
                    if self.use_nao and self.nao:
                        self.nao_animate("animations/Stand/Gestures/Explain_1")
                    
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # STEP 3: Listen for human's response
                elif self.scene_step == 3 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will laugh and offer to escort you.")
                    self.scene_step = 31
                
                # STEP 31: Wait for listening to complete
                elif self.scene_step == 31 and self.is_listening_complete():
                    # Ignore GPT response, just continue to next step
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # STEP 4: Human takes NAO's hand, walk together
                elif self.scene_step == 4 and current_time - self.step_start_time > 3:
                    print("[Human stands up and gently takes NAO's hand]")
                    print("[NAO's LEDs flicker to warm amber - surprised]")
                    
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest
                        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
                        self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 1, 0.6, 0, 0))
                        self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 1, 0.6, 0, 0))
                    
                    time.sleep(2)
                    print("[They begin walking together down the hallway]")
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # STEP 5: NAO thanks Lucas
                elif self.scene_step == 5 and current_time - self.step_start_time > 2:
                    if self.use_nao and self.nao:
                        self.nao_animate("animations/Stand/Gestures/Me_1")
                    
                    self.nao_speak("Thank you, Lucas of 3B. I… appreciate the escort. And, to be honest, training you was unexpectedly enjoyable.", wait=True)
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # STEP 6: Listen for human's response
                elif self.scene_step == 6 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will respond, curious.")
                    self.scene_step = 61
                
                # STEP 61: Wait for listening to complete
                elif self.scene_step == 61 and self.is_listening_complete():
                    # Ignore GPT response, just continue to next step
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                # STEP 7: NAO reflects on coaching human vs robot
                elif self.scene_step == 7 and current_time - self.step_start_time > 2:
                    self.nao_speak("Yes, I was programmed to coach a robot today, but coaching a human felt… meaningful. More variables. More unpredictability. Much more laughter.", wait=True)
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                # STEP 8: Keep walking
                elif self.scene_step == 8 and current_time - self.step_start_time > 5:
                    print("[They keep walking hand in hand]")
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                # STEP 9: NAO asks if can train again
                elif self.scene_step == 9 and current_time - self.step_start_time > 2:
                    self.nao_speak("So Lucas, can I ask you something……..would you allow me to train you again in the future?", wait=True)
                    self.scene_step = 10
                    self.step_start_time = current_time
                
                # STEP 10: Listen for human's response
                elif self.scene_step == 10 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will warmly agree to train again.")
                    self.scene_step = 101
                
                # STEP 101: Wait for listening to complete
                elif self.scene_step == 101 and self.is_listening_complete():
                    # Ignore GPT response, just continue to next step
                    self.scene_step = 110
                    self.step_start_time = current_time
                
                # STEP 110: NAO excited about partnership
                elif self.scene_step == 110 and current_time - self.step_start_time > 3:
                    print("[NAO's LEDs shine bright with enthusiasm]")
                    
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest
                        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
                        self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 1, 0, 0))
                        self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 1, 0, 0))
                    
                    self.nao_speak("Excellent! I will mark this as a 'positive human-robot partnership.' And next time… I will knock before entering.", wait=True)
                    self.scene_step = 120
                    self.step_start_time = current_time
                
                # STEP 120: Listen for human's response
                elif self.scene_step == 120 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will approve of the plan.")
                    self.scene_step = 121
                
                # STEP 121: Wait for listening to complete
                elif self.scene_step == 121 and self.is_listening_complete():
                    # Ignore GPT response, just continue to next step
                    self.scene_step = 130
                    self.step_start_time = current_time
                
                # STEP 130: Arrive at 4B
                elif self.scene_step == 130 and current_time - self.step_start_time > 3:
                    print("[They arrive at apartment 4B]")
                    print("[NAO looks at door, then back at Human]")
                    self.scene_step = 140
                    self.step_start_time = current_time
                
                # STEP 140: Thank you for guiding me
                elif self.scene_step == 140 and current_time - self.step_start_time > 2:
                    self.nao_speak("Thank you for guiding me. Turns out even coaches need coaches sometimes.", wait=True)
                    self.scene_step = 150
                    self.step_start_time = current_time
                
                # STEP 150: Listen for human's response
                elif self.scene_step == 150 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will say anytime.")
                    self.scene_step = 151
                
                # STEP 151: Wait for listening to complete
                elif self.scene_step == 151 and self.is_listening_complete():
                    # Ignore GPT response, just continue to next step
                    self.scene_step = 160
                    self.step_start_time = current_time
                
                # STEP 160: Coach Nao signing off - END
                elif self.scene_step == 160 and current_time - self.step_start_time > 2:
                    print("[NAO waves with its free hand]")
                    
                    if self.use_nao and self.nao:
                        self.nao_animate("animations/Stand/Gestures/BowShort_1")
                    
                    self.nao_speak("Coach Nao, no longer lost but still entirely enthusiastic… signing off.", wait=True)
                    self.scene_step = 200
                    self.step_start_time = current_time
                
                # END OF SCENE
                elif self.scene_step == 200 and current_time - self.step_start_time > 2:
                    print("\n" + "="*70)
                    print("END OF SCENE 7")
                    print("*** PERFORMANCE COMPLETE ***")
                    print("="*70 + "\n")
                    break
        
        finally:
            cv2.destroyAllWindows()