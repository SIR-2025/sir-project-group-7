import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
from NewMotions import CoachNaoMotions
import cv2
import time


class Scene5(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=5)
        self.motions = CoachNaoMotions(nao=self.nao if self.use_nao else None)
    
    def run(self):
        print("SCENE 5: PUSH-UPS")
        
        try:
            pushup_count = 0
            target_pushups = 6
            
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
                
                # ACT 1: Introduction - confident and playful
                if self.scene_step == 0:
                    intro = self.generate_speech(
                        "Tell them to get into push-up position. Be playful - say you don't think they need a demonstration this time.",
                        fallback_text="Okay. Get into push-up position. I don't think you need a demonstration this time."
                    )
                    
                    # Confident gesture
                    if self.use_nao and self.nao:
                        self.motions.confident_presentation()
                    
                    self.nao_speak(intro, wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                # Person questions why no demo
                elif self.scene_step == 1 and current_time - self.step_start_time > 4:
                    self.start_listening("Person asks if it's because you can't do push-ups or really don't need to demo.")
                    self.scene_step = 11
                
                elif self.scene_step == 11 and self.is_listening_complete():
                    # Defensive gesture - caught!
                    if self.use_nao and self.nao:
                        self.motions.embarrassed_look_away()
                        time.sleep(0.5)
                        self.motions.firm_no()
                    
                    deflect = self.generate_speech(
                        "Deflect from your physical limitations. Say let's not talk about your limitations - they're the trainee, you're the trainer. Tell them to start with 10 pushups!",
                        fallback_text="Let's not talk about my physical limitations, you are the trainee, I am the trainer. Start with 10 pushups!"
                    )
                    self.nao_speak(deflect, wait=True)
                    
                    # Authoritative pointing
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.point_forward()
                    
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                # Person agrees
                elif self.scene_step == 2 and current_time - self.step_start_time > 4:
                    self.start_listening("Person says okay sure.")
                    self.scene_step = 21
                
                elif self.scene_step == 21 and self.is_listening_complete():
                    # Encouraging start gesture
                    if self.use_nao and self.nao:
                        self.motions.set_led_emotion("focused")
                    
                    self.nao_speak("Let's start!", wait=True)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                # Push-ups 1-3
                elif self.scene_step == 3 and current_time - self.step_start_time > 3:
                    self.start_listening("Person doing pushups 1, 2, 3.")
                    self.scene_step = 31
                
                elif self.scene_step == 31 and self.is_listening_complete():
                    pushup_count = 3
                    
                    # Impressed gesture
                    if self.use_nao and self.nao:
                        self.motions.encouraging_nod()
                    
                    self.nao_speak("Wow very smooth.", wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                # Push-ups 4-5
                elif self.scene_step == 4 and current_time - self.step_start_time > 3:
                    self.start_listening("Person continuing with pushups 4, 5.")
                    self.scene_step = 41
                
                elif self.scene_step == 41 and self.is_listening_complete():
                    pushup_count = 5
                    
                    # Very impressed gesture
                    if self.use_nao and self.nao:
                        self.motions.celebration_gesture()
                    
                    encourage = self.generate_speech(
                        "Say one more! Their performance is unprecedented in your trainee history!",
                        fallback_text="Okay one more. This performance is unprecedented in my trainee history!"
                    )
                    self.nao_speak(encourage, wait=True)
                    self.scene_step = 5
                    self.step_start_time = current_time
                
                # Final push-up
                elif self.scene_step == 5 and current_time - self.step_start_time > 3:
                    self.start_listening("Person does final pushup.")
                    self.scene_step = 51
                
                elif self.scene_step == 51 and self.is_listening_complete():
                    pushup_count = 6
                    
                    # Big celebration
                    if self.use_nao and self.nao:
                        self.motions.encouragement_sequence()
                    
                    self.nao_speak("You are doing splendid! My other trainees are not as smooth.", wait=True)
                    self.scene_step = 6
                    self.step_start_time = current_time
                
                # THE REVEAL - Person asks about trainees
                elif self.scene_step == 6 and current_time - self.step_start_time > 4:
                    self.start_listening("Person asks if you train humans or robots.")
                    self.scene_step = 61
                
                elif self.scene_step == 61 and self.is_listening_complete():
                    # Confused processing gesture
                    if self.use_nao and self.nao:
                        self.motions.confused_shrug()
                        self.motions.set_led_emotion("embarrassed")
                    
                    reveal = self.generate_speech(
                        "Say 'Robots, of course.' Pause. Ask if they're PR-103A25. Comment they have interesting mechanical attributes.",
                        fallback_text="Robots, of course. Wait, are you not PR-103A25? You do have some interesting mechanical attributes."
                    )
                    self.nao_speak(reveal, wait=True)
                    
                    # Look at them quizzically
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.point_forward()
                    
                    self.scene_step = 7
                    self.step_start_time = current_time
                
                # Person reveals name - NOT a robot
                elif self.scene_step == 7 and current_time - self.step_start_time > 5:
                    self.start_listening("Person says no, their name is Lucas (or their actual name).")
                    self.scene_step = 71
                
                elif self.scene_step == 71 and self.is_listening_complete():
                    # Processing... embarrassment building
                    if self.use_nao and self.nao:
                        self.motions.look_around_confused()
                    
                    processing = self.generate_speech(
                        "Say 'Lucas?' (or their name). That's not in your database. Ask if this is apartment 4B.",
                        fallback_text="Lucas? That's... not in my database. This is apartment 4B, correct?"
                    )
                    self.nao_speak(processing, wait=True)
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                # Wrong apartment revelation
                elif self.scene_step == 8 and current_time - self.step_start_time > 5:
                    self.start_listening("Person says this is 3B. You're one floor off.")
                    self.scene_step = 81
                
                elif self.scene_step == 81 and self.is_listening_complete():
                    # Maximum embarrassment
                    if self.use_nao and self.nao:
                        self.motions.set_led_emotion("embarrassed")
                        self.motions.defeated_slump()
                        time.sleep(1)
                        self.motions.embarrassed_look_away()
                    
                    realization = self.generate_speech(
                        "Long awkward pause. Say '...Oh.' This explains why the door was unlocked. PR-103A25 always deadbolts. Pause. Say their form is excellent though. Ask if they want you to leave.",
                        fallback_text="...Oh. This explains why the door was unlocked. PR-103A25 always deadbolts. Well... your form is excellent. Would you... like me to leave?"
                    )
                    self.nao_speak(realization, wait=True)
                    
                    # Hopeful but uncertain gesture
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.pleading_gesture()
                    
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                # Person decides to continue!
                elif self.scene_step == 9 and current_time - self.step_start_time > 6:
                    self.start_listening("Person says you're already here, and they were going to work out anyway. Let's finish this.")
                    self.scene_step = 91
                
                elif self.scene_step == 91 and self.is_listening_complete():
                    # Relief! Gratitude!
                    if self.use_nao and self.nao:
                        self.motions.set_led_emotion("excited")
                        self.motions.grateful_farewell()
                    
                    grateful = self.generate_speech(
                        "Say that sounds good. Call them by their name and mention they're from 3B. Say let's move to the final challenge!",
                        fallback_text="Okay that's sounds good Lucas of 3B. Let's move to the final challenge!"
                    )
                    self.nao_speak(grateful, wait=True)
                    
                    # Renewed energy gesture
                    if self.use_nao and self.nao:
                        time.sleep(0.5)
                        self.motions.celebration_gesture()
                        self.motions.set_led_emotion("focused")
                    
                    self.scene_step = 10
                    self.step_start_time = current_time
                
                elif self.scene_step == 10 and current_time - self.step_start_time > 3:
                    print("END OF SCENE 5")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()