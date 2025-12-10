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
                
                if self.scene_step == 0:
                    print("[NAO LEDs glow soft, calm blue]")
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest
                        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
                        self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0, 0.5, 0))
                        self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0, 0.5, 0))
                    
                    complete = self.generate_speech(
                        "Announce session complete. Say they performed at 98% efficiency - a new personal record.",
                        fallback_text="Session complete. Statistically, you performed at 98% efficiency. A new personal record."
                    )
                    self.nao_speak(complete, wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                elif self.scene_step == 1 and current_time - self.step_start_time > 5:
                    print("[Human grabs water bottle, sits with amused smile]")
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                elif self.scene_step == 2 and current_time - self.step_start_time > 3:
                    self.start_listening("Person will ask what happens now and offer to walk you to the actual apartment.")
                    self.scene_step = 21
                
                elif self.scene_step == 21 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                elif self.scene_step == 3 and current_time - self.step_start_time > 3:
                    print("[NAO glances at door, then back at Human, slightly embarrassed]")
                    
                    admit = self.generate_speech(
                        "Say you should report to apartment 4B immediately. Your internal navigation insists it's 'that way'. Point confidently in completely wrong direction.",
                        fallback_text="Yes. I should report to apartment 4B immediately. My internal navigation insists it is… that way."
                    )
                    self.nao_speak(admit, wait=True)
                    
                    print("[NAO points confidently in the WRONG direction]")
                    if self.use_nao and self.nao:
                        self.nao_animate("animations/Stand/Gestures/Explain_1")
                    
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                elif self.scene_step == 4 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will laugh and offer to escort you there.")
                    self.scene_step = 41
                
                elif self.scene_step == 41 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                elif self.scene_step == 5 and current_time - self.step_start_time > 3:
                    print("[Human stands up and gently takes NAO's hand]")
                    print("[NAO's LEDs flicker to warm amber - surprised]")
                    
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest
                        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
                        self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 1, 0.6, 0, 0))
                        self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 1, 0.6, 0, 0))
                    
                    time.sleep(2)
                    print("[They begin walking together down the hallway]")
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                elif self.scene_step == 6 and current_time - self.step_start_time > 2:
                    if self.use_nao and self.nao:
                        self.nao_animate("animations/Stand/Gestures/Me_1")
                    
                    self.nao_speak("Thank you, Lucas of 3B. I… appreciate the escort. And, to be honest, training you was unexpectedly enjoyable.",
                                  wait=True)
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                elif self.scene_step == 7 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will respond, curious.")
                    self.scene_step = 71
                
                elif self.scene_step == 71 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                elif self.scene_step == 8 and current_time - self.step_start_time > 3:
                    reflection = self.generate_speech(
                        "Explain you were programmed to coach a robot today, but coaching a human felt meaningful. More variables, more unpredictability, much more laughter.",
                        fallback_text="Yes, I was programmed to coach a robot today, but coaching a human felt… meaningful. More variables. More unpredictability. Much more laughter."
                    )
                    self.nao_speak(reflection, wait=True)
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                elif self.scene_step == 9 and current_time - self.step_start_time > 5:
                    print("[They keep walking hand in hand]")
                    self.scene_step = 10
                    self.step_start_time = current_time
                
                elif self.scene_step == 10 and current_time - self.step_start_time > 2:
                    ask = self.generate_speech(
                        "Say 'So Lucas, can I ask you something'. Pause. Ask if they would allow you to train them again in the future.",
                        fallback_text="So Lucas, can I ask you something……..would you allow me to train you again in the future?"
                    )
                    self.nao_speak(ask, wait=True)
                    self.scene_step = 11
                    self.step_start_time = current_time
                
                elif self.scene_step == 11 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will warmly agree to train again.")
                    self.scene_step = 111
                
                elif self.scene_step == 111 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 12
                    self.step_start_time = current_time
                
                elif self.scene_step == 12 and current_time - self.step_start_time > 3:
                    print("[NAO's LEDs shine bright with enthusiasm]")
                    
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest
                        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
                        self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 1, 0, 0))
                        self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 1, 0, 0))
                    
                    excited = self.generate_speech(
                        "Say 'Excellent!' Mark this as a positive human-robot partnership. Promise to knock before entering next time.",
                        fallback_text="Excellent! I will mark this as a 'positive human-robot partnership.' And next time… I will knock before entering."
                    )
                    self.nao_speak(excited, wait=True)
                    self.scene_step = 13
                    self.step_start_time = current_time
                
                elif self.scene_step == 13 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will approve of the plan.")
                    self.scene_step = 131
                
                elif self.scene_step == 131 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 14
                    self.step_start_time = current_time
                
                elif self.scene_step == 14 and current_time - self.step_start_time > 3:
                    print("[They arrive at apartment 4B]")
                    print("[NAO looks at door, then back at Human]")
                    self.scene_step = 15
                    self.step_start_time = current_time
                
                elif self.scene_step == 15 and current_time - self.step_start_time > 2:
                    thanks = self.generate_speech(
                        "Thank them for guiding you. Say even coaches need coaches sometimes.",
                        fallback_text="Thank you for guiding me. Turns out even coaches need coaches sometimes."
                    )
                    self.nao_speak(thanks, wait=True)
                    self.scene_step = 16
                    self.step_start_time = current_time
                
                elif self.scene_step == 16 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will say anytime.")
                    self.scene_step = 161
                
                elif self.scene_step == 161 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 17
                    self.step_start_time = current_time
                
                elif self.scene_step == 17 and current_time - self.step_start_time > 3:
                    print("[NAO waves with its free hand]")
                    
                    if self.use_nao and self.nao:
                        self.nao_animate("animations/Stand/Gestures/BowShort_1")
                    
                    self.nao_speak("Coach Nao, no longer lost but still entirely enthusiastic… signing off.",
                                  wait=True)
                    self.scene_step = 18
                    self.step_start_time = current_time
                
                elif self.scene_step == 18 and current_time - self.step_start_time > 3:
                    confused = self.generate_speech(
                        "Ask uncertainly if the elevator is to the left. Sound confused about directions.",
                        fallback_text="It's... left out of here to the elevator, correct?"
                    )
                    self.nao_speak(confused, wait=True)
                    self.scene_step = 181
                    self.step_start_time = current_time
                
                elif self.scene_step == 181 and current_time - self.step_start_time > 3:
                    self.start_listening("Person will correct your direction.")
                    self.scene_step = 182
                
                elif self.scene_step == 182 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 19
                    self.step_start_time = current_time
                
                elif self.scene_step == 19 and current_time - self.step_start_time > 2:
                    wrong = self.generate_speech(
                        "Say 'Right! Of course! I knew that!' confidently but clearly didn't know.",
                        fallback_text="Right! Of course! I knew that!"
                    )
                    self.nao_speak(wrong, wait=True)
                    self.scene_step = 20
                    self.step_start_time = current_time
                
                elif self.scene_step == 20 and current_time - self.step_start_time > 2:
                    print("[NAO turns and starts walking confidently in the WRONG direction]")
                    print("[Human smiles, shakes head affectionately]")
                    self.scene_step = 100
                    self.step_start_time = current_time
                
                elif self.scene_step == 100 and current_time - self.step_start_time > 3:
                    self.scene_step = 101
                
                elif self.scene_step == 101:
                    print("\n" + "="*70)
                    print("END OF SCENE 7")
                    print("*** PERFORMANCE COMPLETE ***")
                    print("="*70 + "\n")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()