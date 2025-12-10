import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
from NewMotions import CoachNaoMotions
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest
import cv2
import time


class Scene7(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=7)
        self.motions = CoachNaoMotions(nao=self.nao if self.use_nao else None)
    
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
                
                # ACT 1: Session complete - warm, calm energy
                if self.scene_step == 0:
                    print("[NAO LEDs glow soft, calm blue]")
                    if self.use_nao and self.nao:
                        self.motions.set_led_emotion("calm")
                    
                    complete = self.generate_speech(
                        "Announce session complete. Say they performed at 98% efficiency - a new personal record.",
                        fallback_text="Session complete. Statistically, you performed at 98% efficiency. A new personal record."
                    )
                    
                    # Professional concluding gesture
                    if self.use_nao and self.nao:
                        self.motions.polite_bow()
                    
                    self.nao_speak(complete, wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # Pause for human to get water
                elif self.scene_step == 1 and current_time - self.step_start_time > 5:
                    print("[Human grabs water bottle, sits with amused smile]")
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # Listen for offer to walk to 4B
                elif self.scene_step == 2 and current_time - self.step_start_time > 3:
                    self.start_listening("Person will ask what happens now and offer to walk you to the actual apartment.")
                    self.scene_step = 21
                
                # Respond gratefully
                elif self.scene_step == 21 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        # Grateful gesture
                        if self.use_nao and self.nao:
                            self.motions.grateful_farewell()
                        
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # Admit navigation error with embarrassment
                elif self.scene_step == 3 and current_time - self.step_start_time > 3:
                    print("[NAO glances at door, then back at Human, slightly embarrassed]")
                    
                    if self.use_nao and self.nao:
                        self.motions.look_around_confused()
                        time.sleep(0.5)
                        self.motions.set_led_emotion("embarrassed")
                    
                    admit = self.generate_speech(
                        "Say you should report to apartment 4B immediately. Your internal navigation insists it's 'that way'. Point confidently in completely wrong direction.",
                        fallback_text="Yes. I should report to apartment 4B immediately. My internal navigation insists it is... that way."
                    )
                    self.nao_speak(admit, wait=True)
                    
                    print("[NAO points confidently in the WRONG direction]")
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.point_confidently_wrong()
                    else:
                        self.nao_animate("animations/Stand/Gestures/Explain_1")
                    
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # Listen for correction
                elif self.scene_step == 4 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will laugh and offer to escort you there.")
                    self.scene_step = 41
                
                # Accept escort
                elif self.scene_step == 41 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        # Grateful but embarrassed
                        if self.use_nao and self.nao:
                            self.motions.defeated_slump()
                            time.sleep(0.3)
                            self.motions.polite_bow()
                        
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # Walking together - warm moment
                elif self.scene_step == 5 and current_time - self.step_start_time > 3:
                    print("[Human stands up and gently takes NAO's hand]")
                    print("[NAO's LEDs flicker to warm amber - surprised]")
                    
                    if self.use_nao and self.nao:
                        self.motions.set_led_emotion("embarrassed")  # Warm amber
                        time.sleep(1)
                        # Gentle, touched gesture
                        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_11"))
                    
                    time.sleep(2)
                    print("[They begin walking together down the hallway]")
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # Grateful reflection
                elif self.scene_step == 6 and current_time - self.step_start_time > 2:
                    if self.use_nao and self.nao:
                        self.motions.self_reference()
                    
                    gratitude = "Thank you, Lucas of 3B. I... appreciate the escort. And, to be honest, training you was unexpectedly enjoyable."
                    self.nao_speak(gratitude, wait=True)
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                # Listen for "Yeah?"
                elif self.scene_step == 7 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will respond, curious.")
                    self.scene_step = 71
                
                # Respond warmly
                elif self.scene_step == 71 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        if self.use_nao and self.nao:
                            self.motions.reflective_pause()
                        
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                # Meaningful explanation
                elif self.scene_step == 8 and current_time - self.step_start_time > 3:
                    reflection = self.generate_speech(
                        "Explain you were programmed to coach a robot today, but coaching a human felt meaningful. More variables, more unpredictability, much more laughter.",
                        fallback_text="Yes, I was programmed to coach a robot today, but coaching a human felt... meaningful. More variables. More unpredictability. Much more laughter."
                    )
                    
                    # Thoughtful, sincere gesture
                    if self.use_nao and self.nao:
                        self.motions.detailed_explanation()
                        time.sleep(0.5)
                        self.motions.set_led_emotion("calm")
                    
                    self.nao_speak(reflection, wait=True)
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                # Walking pause
                elif self.scene_step == 9 and current_time - self.step_start_time > 5:
                    print("[They keep walking hand in hand]")
                    self.scene_step = 10
                    self.step_start_time = current_time
                
                # Ask to train again - vulnerable moment
                elif self.scene_step == 10 and current_time - self.step_start_time > 2:
                    ask = self.generate_speech(
                        "Say 'So Lucas, can I ask you something'. Pause. Ask if they would allow you to train them again in the future.",
                        fallback_text="So Lucas, can I ask you something......would you allow me to train you again in the future?"
                    )
                    
                    # Hopeful, vulnerable gesture
                    if self.use_nao and self.nao:
                        self.motions.hopeful_reach()
                    
                    self.nao_speak(ask, wait=True)
                    self.scene_step = 11
                    self.step_start_time = current_time
                
                # Listen for agreement
                elif self.scene_step == 11 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will warmly agree to train again.")
                    self.scene_step = 111
                
                # Respond to agreement
                elif self.scene_step == 111 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        if self.use_nao and self.nao:
                            self.motions.polite_bow()
                        
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 12
                    self.step_start_time = current_time
                
                # Enthusiastic joy!
                elif self.scene_step == 12 and current_time - self.step_start_time > 3:
                    print("[NAO's LEDs shine bright with enthusiasm]")
                    
                    if self.use_nao and self.nao:
                        self.motions.set_led_emotion("excited")
                        self.motions.celebration_gesture()
                    
                    excited = self.generate_speech(
                        "Say 'Excellent!' Mark this as a positive human-robot partnership. Promise to knock before entering next time.",
                        fallback_text="Excellent! I will mark this as a 'positive human-robot partnership.' And next time... I will knock before entering."
                    )
                    self.nao_speak(excited, wait=True)
                    
                    # Self-aware follow-up
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.self_reference()
                    
                    self.scene_step = 13
                    self.step_start_time = current_time
                
                # Listen for approval
                elif self.scene_step == 13 and current_time - self.step_start_time > 5:
                    self.start_listening("Person will approve of the plan.")
                    self.scene_step = 131
                
                # Acknowledge approval
                elif self.scene_step == 131 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        if self.use_nao and self.nao:
                            self.motions.encouraging_nod()
                        
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 14
                    self.step_start_time = current_time
                
                # Arrive at 4B
                elif self.scene_step == 14 and current_time - self.step_start_time > 3:
                    print("[They arrive at apartment 4B]")
                    print("[NAO looks at door, then back at Human]")
                    
                    if self.use_nao and self.nao:
                        self.motions.look_around_confused()  # Looking between door and human
                    
                    self.scene_step = 15
                    self.step_start_time = current_time
                
                # Final thank you
                elif self.scene_step == 15 and current_time - self.step_start_time > 2:
                    thanks = self.generate_speech(
                        "Thank them for guiding you. Say even coaches need coaches sometimes.",
                        fallback_text="Thank you for guiding me. Turns out even coaches need coaches sometimes."
                    )
                    
                    # Deep, sincere bow
                    if self.use_nao and self.nao:
                        self.motions.bonding_moment_sequence()
                    
                    self.nao_speak(thanks, wait=True)
                    self.scene_step = 16
                    self.step_start_time = current_time
                
                # Listen for farewell
                elif self.scene_step == 16 and current_time - self.step_start_time > 4:
                    self.start_listening("Person will say anytime.")
                    self.scene_step = 161
                
                # Respond to farewell
                elif self.scene_step == 161 and self.is_listening_complete():
                    gpt_response = self.get_gpt_response()
                    if gpt_response:
                        if self.use_nao and self.nao:
                            self.motions.grateful_farewell()
                        
                        self.nao_speak(gpt_response, wait=True)
                    self.scene_step = 17
                    self.step_start_time = current_time
                
                # Sign-off
                elif self.scene_step == 17 and current_time - self.step_start_time > 3:
                    print("[NAO waves with its free hand]")
                    
                    if self.use_nao and self.nao:
                        self.nao_animate("animations/Stand/Gestures/Hey_1")
                    
                    signoff = "Coach Nao, no longer lost but still entirely enthusiastic... signing off."
                    self.nao_speak(signoff, wait=True)
                    self.scene_step = 18
                    self.step_start_time = current_time
                
                # COMEDY CALLBACK - still lost!
                elif self.scene_step == 18 and current_time - self.step_start_time > 3:
                    # Person corrects direction
                    print("[Human: 'It's left out of here to the elevator, correct?']")
                    print("[NAO: 'Other way, Nao. Other way.']")
                    time.sleep(2)
                    
                    wrong = self.generate_speech(
                        "Say 'Right! Of course! I knew that!' Sound confident but clearly didn't know.",
                        fallback_text="Right! Of course! I knew that!"
                    )
                    
                    # Overconfident gesture
                    if self.use_nao and self.nao:
                        self.motions.point_confidently_wrong()
                        self.motions.set_led_emotion("embarrassed")
                    
                    self.nao_speak(wrong, wait=True)
                    self.scene_step = 19
                    self.step_start_time = current_time
                
                # Final comedic exit
                elif self.scene_step == 19 and current_time - self.step_start_time > 2:
                    print("[NAO turns and starts walking confidently in the WRONG direction]")
                    print("[Human smiles, shakes head affectionately]")
                    
                    # One last confused look back
                    if self.use_nao and self.nao:
                        time.sleep(1)
                        self.motions.confused_shrug()
                        self.motions.awkward_wave()
                    
                    self.scene_step = 100
                    self.step_start_time = current_time
                
                # Fade out
                elif self.scene_step == 100 and current_time - self.step_start_time > 3:
                    # LEDs fade to off
                    if self.use_nao and self.nao:
                        self.set_leds_off()
                    
                    self.scene_step = 101
                
                # End
                elif self.scene_step == 101:
                    print("\n" + "="*70)
                    print("END OF SCENE 7")
                    print("*** PERFORMANCE COMPLETE ***")
                    print("="*70 + "\n")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()