import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from base_scene import BaseScene
import cv2
import time


class Scene4(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_scene_context(scene_number=4)
    
    def run(self):
        print("SCENE 4: SQUATS - THE BONDING BEGINS")
        
        try:
            squat_count = 0
            target_squats = 3
            
            squat_1_retries = 0
            squat_2_retries = 0
            squat_3_retries = 0
            max_retries = 2
            
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
                
                if self.scene_step == 0:
                    intro = self.generate_speech(
                        "Announce the first exercise is the squat. Call it a biomechanical marvel.",
                        fallback_text="Excellent. Our first exercise is the biomechanical marvel known as the... squat."
                    )
                    self.nao_speak(intro, animation="animations/Stand/Gestures/Explain_1", wait=True)
                    self.scene_step = 1
                    self.step_start_time = current_time
                
                elif self.scene_step == 1 and current_time - self.step_start_time > 5:
                    demo_warning = self.generate_speech(
                        "Tell them you'll demonstrate but you're not as flexible as them.",
                        fallback_text="I will now demonstrate a poor representation of it because I'm not as flexible as you."
                    )
                    self.nao_speak(demo_warning, wait=True)
                    self.scene_step = 2
                    self.step_start_time = current_time
                
                elif self.scene_step == 2 and current_time - self.step_start_time > 4:
                    print("[NAO SITS DOWN - 'squat attempt']")
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
                        self.nao.motion.request(NaoPostureRequest("Sit", 0.8))
                    else:
                        time.sleep(2)
                    self.scene_step = 21
                    self.step_start_time = current_time
                
                elif self.scene_step == 21 and current_time - self.step_start_time > 3:
                    print("[NAO STANDS BACK UP]")
                    if self.use_nao and self.nao:
                        from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest
                        self.nao.motion.request(NaoPostureRequest("Stand", 0.8))
                    else:
                        time.sleep(2)
                    self.scene_step = 3
                    self.step_start_time = current_time
                
                elif self.scene_step == 3 and current_time - self.step_start_time > 2:
                    request = self.generate_speech(
                        "Admit that's the best you can do but they can do better. Request three squats.",
                        fallback_text="So this is the best I can do but you can surely do it better, so please perform three squats."
                    )
                    self.nao_speak(request, wait=True)
                    self.scene_step = 4
                    self.step_start_time = current_time
                
                elif self.scene_step == 4 and current_time - self.step_start_time > 5:
                    self.start_listening("Person might comment on your squat, then do their first squat and say 'One'.")
                    self.scene_step = 41
                
                elif self.scene_step == 41 and self.is_listening_complete():
                    response = self.get_user_input()
                    if response and ("one" in response.lower() or "1" in response or "won" in response.lower()):
                        squat_count = 1
                        squat_1_retries = 0
                        print(f"[Squat #{squat_count} detected]")
                        
                        feedback = self._get_squat_feedback()
                        self.nao_speak(feedback, wait=True)
                        time.sleep(0.5)
                        self.nao_speak("Now do the second one!", wait=True)
                        
                        self.scene_step = 5
                        self.step_start_time = current_time
                    else:
                        squat_1_retries += 1
                        print(f"[Did not detect 'one' - retry {squat_1_retries}/{max_retries}]")
                        
                        if squat_1_retries >= max_retries:
                            print("[Max retries for squat 1 - moving on]")
                            squat_count = 1
                            self.nao_speak("Let's count that as one! Moving on...", wait=True)
                            self.scene_step = 5
                        else:
                            missed = self.generate_speech(
                                "You didn't see the squat properly. Ask them to repeat squat number one and say 'one' when done.",
                                fallback_text="Sorry, I didn't catch that. Can you do squat one again and say 'one'?"
                            )
                            self.nao_speak(missed, wait=True)
                            self.scene_step = 4
                        
                        self.step_start_time = current_time
                
                elif self.scene_step == 5 and current_time - self.step_start_time > 3:
                    self.start_listening("Person doing second squat, will say 'Two'.")
                    self.scene_step = 51
                
                elif self.scene_step == 51 and self.is_listening_complete():
                    response = self.get_user_input()
                    if response and ("two" in response.lower() or "2" in response or 
                                    "too" in response.lower() or "to" in response.lower() or 
                                    "do" in response.lower() or "tu" in response.lower()):
                        squat_count = 2
                        squat_2_retries = 0
                        print(f"[Squat #{squat_count} detected]")
                        
                        feedback = self._get_squat_feedback()
                        self.nao_speak(feedback, wait=True)
                        time.sleep(0.5)
                        self.nao_speak("One more to go!", wait=True)
                        
                        self.scene_step = 6
                        self.step_start_time = current_time
                    else:
                        squat_2_retries += 1
                        print(f"[Did not detect 'two' - retry {squat_2_retries}/{max_retries}]")
                        
                        if squat_2_retries >= max_retries:
                            print("[Max retries for squat 2 - moving on]")
                            squat_count = 2
                            self.nao_speak("Alright, two done! One more to go!", wait=True)
                            self.scene_step = 6
                        else:
                            missed = self.generate_speech(
                                "You didn't catch squat number two. Ask them to do it again and say 'two' when done.",
                                fallback_text="Oops, my sensors glitched. Can you do squat two again and say 'two'?"
                            )
                            self.nao_speak(missed, wait=True)
                            self.scene_step = 5
                        
                        self.step_start_time = current_time
                
                elif self.scene_step == 6 and current_time - self.step_start_time > 3:
                    self.start_listening("Person doing third squat, will say 'Three'.")
                    self.scene_step = 61

                elif self.scene_step == 61 and self.is_listening_complete():
                    response = self.get_user_input()

                    if response and ("three" in response.lower() or "3" in response or 
                                    "tree" in response.lower() or "free" in response.lower() or 
                                    "thee" in response.lower()):
                        squat_count = 3
                        squat_3_retries = 0
                        print(f"[Squat #{squat_count} detected]")
                        
                        if self.use_nao and self.nao:
                            self.nao_animate("animations/Stand/Negation/NAO/Center_Neutral_NEG_04")
                        
                        feedback = self._get_squat_feedback()
                        self.nao_speak(feedback, wait=True)
                        time.sleep(0.5)
                        self.nao_speak("That was the last one!", wait=True)
                        
                        time.sleep(1)
                        tips = self._get_squat_tips()
                        self.nao_speak(tips, wait=True)
                        
                        time.sleep(1)
                        self.nao_speak("Great job! You look like you do this quite often. Sometimes I question why I do this.", wait=True)
                        
                        self.scene_step = 7
                        self.step_start_time = current_time
                    else:
                        squat_3_retries += 1
                        print(f"[Did not detect 'three' - retry {squat_3_retries}/{max_retries}]")
                        
                        if squat_3_retries >= max_retries:
                            print("[Max retries for squat 3 - moving on]")
                            squat_count = 3
                            self.nao_speak("Perfect! Three squats complete!", wait=True)
                            self.scene_step = 7
                        else:
                            missed = self.generate_speech(
                                "You missed the final squat. Ask them to repeat squat three and announce 'three' when finished.",
                                fallback_text="My bad! One more time - squat three, and shout 'three' when you finish!"
                            )
                            self.nao_speak(missed, wait=True)
                            self.scene_step = 6
                        
                        self.step_start_time = current_time
                
                elif self.scene_step == 7 and current_time - self.step_start_time > 5:
                    self.start_listening("Person might ask a question about breaking in or training.")
                    self.scene_step = 71
                
                elif self.scene_step == 71 and self.is_listening_complete():
                    self.scene_step = 8
                    self.step_start_time = current_time
                
                elif self.scene_step == 8 and current_time - self.step_start_time > 1:
                    self.nao_speak("Uhmmm... let's move on. How about we do some push-ups now, are you ready?", wait=True)
                    self.scene_step = 9
                    self.step_start_time = current_time
                
                elif self.scene_step == 9 and current_time - self.step_start_time > 4:
                    self.start_listening("Person agreeing to push-ups.")
                    self.scene_step = 91
                
                elif self.scene_step == 91 and self.is_listening_complete():
                    self.scene_step = 10
                
                elif self.scene_step == 10:
                    print("END OF SCENE 4")
                    break
        
        finally:
            self.set_leds_off()
            cv2.destroyAllWindows()

    def _get_squat_tips(self):
            if not self.pose_analyzer:
                return "Remember: knees over toes, back straight!"
            
            try:
                angles, _ = self.pose_analyzer.capture_and_analyze()
                
                if angles is None:
                    return "Keep practicing those squats!"
                
                analysis = self.pose_analyzer.check_squat_form(angles)
                
                issues = []
                for joint, data in analysis['joints'].items():
                    if data['status'] != 'good':
                        if 'knee' in joint:
                            issues.append("knee alignment")
                        elif 'hip' in joint:
                            issues.append("hip depth")
                        elif 'back' in joint:
                            issues.append("back posture")
                
                depth_status = analysis.get('squat_depth', {}).get('status')
                if depth_status == 'too_shallow':
                    issues.append("squat depth")
                
                if issues:
                    tip = issues[0].replace('_', ' ')
                    return f"Pro tip: watch your {tip} next time!"
                else:
                    return "Your form is solid! Keep it up!"
            
            except Exception as e:
                print(f"[Tips error: {e}]")
                return "Remember to breathe during squats!"
    
    def _get_squat_feedback(self):
        if not self.pose_analyzer:
            print("[No pose analyzer - using fallback]")
            return "Nice form!"
        
        try:
            angles, annotated_frame = self.pose_analyzer.capture_and_analyze()
            
            if angles is None:
                print("[Could not detect pose - using fallback]")
                return "Good effort!"
            
            analysis = self.pose_analyzer.check_squat_form(angles)
            accuracy = analysis.get('overall_accuracy', 0)
            
            print(f"[Pose Analysis: {accuracy:.1f}% accuracy]")
            
            if accuracy >= 85:
                return "Excellent form!"
            elif accuracy >= 70:
                return "Good squat!"
            else:
                return "Nice try!"
        
        except Exception as e:
            print(f"[Pose analysis error: {e}]")
            return "Looking good!"