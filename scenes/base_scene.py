import cv2
import time
import threading
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest
from sic_framework.devices.common_naoqi.naoqi_leds import NaoFadeRGBRequest, NaoLEDRequest


class BaseScene:
    """Base class for all performance scenes"""
    
    def __init__(self, nao=None, dialogue_manager=None, camera_manager=None, 
                 pose_analyzer=None, use_nao=True):
        self.nao = nao
        self.dialogue_manager = dialogue_manager
        self.camera_manager = camera_manager
        self.pose_analyzer = pose_analyzer
        self.use_nao = use_nao
        
        self.scene_step = 0
        self.step_start_time = time.time()
        self.frame_count = 0
        
        self.listening_active = False
        self.listen_thread = None
        self.listen_result = [None]
    
    def set_leds_listening(self):
        """Set face LEDs to blue (listening)"""
        if self.use_nao and self.nao:
            self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
            self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0, 1, 0))  # Right blue
            self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0, 1, 0))   # Left blue
    
    def set_leds_thinking(self):
        """Set face LEDs to red (thinking/processing)"""
        if self.use_nao and self.nao:
            self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
            self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 1, 0, 0, 0))  # Right red
            self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 1, 0, 0, 0))   # Left red
    
    def set_leds_off(self):
        """Turn off face LEDs"""
        if self.use_nao and self.nao:
            self.nao.leds.request(NaoLEDRequest("FaceLeds", False))
    
    def nao_speak(self, text, animation=None, wait=True):
        cleaned_text = text.replace('...', ',')
        
        print(f"[NAO SPEAKS]: {text}")
        if animation:
            print(f"[NAO ANIMATES]: {animation.split('/')[-1]}")
        
        if self.use_nao and self.nao:
            self.nao.tts.request(NaoqiTextToSpeechRequest(cleaned_text), block=wait)
            if animation:
                self.nao.motion.request(NaoqiAnimationRequest(animation), block=False)
        elif wait:
            time.sleep(len(text.split()) * 0.4)
    
    def nao_animate(self, animation):
        print(f"[NAO ANIMATES]: {animation.split('/')[-1]}")
        
        if self.use_nao and self.nao:
            self.nao.motion.request(NaoqiAnimationRequest(animation), block=False)
        else:
            time.sleep(1)
    
    def play_jingle(self):
        print("[JINGLE PLAYS]")
        time.sleep(1)
    
    def start_listening(self):
        if self.listening_active:
            return
        
        print("[LISTENING FOR RESPONSE...]")
        self.listening_active = True
        self.listen_result[0] = None
        
        self.set_leds_listening()
        
        def listen_async():
            result = self.dialogue_manager.listen_and_respond_auto(
                max_duration=30.0,
                silence_threshold=0.04,
                silence_duration=2
            )
            
            self.set_leds_thinking()
            
            if result and 'user_input' in result:
                self.listen_result[0] = result['user_input']
                print(f"[PERSON SAID]: {self.listen_result[0]}")
            else:
                print("[NO SPEECH DETECTED]")
            
        
        self.listen_thread = threading.Thread(target=listen_async, daemon=True)
        self.listen_thread.start()
    
    def is_listening_complete(self):
        if self.listen_thread and not self.listen_thread.is_alive():
            self.listening_active = False
            return True
        return False
    
    def get_listen_result(self):
        return self.listen_result[0]
    
    def draw_scene_info(self, frame, scene_name):
        cv2.putText(frame, scene_name, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frame {self.frame_count}", 
                   (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.listening_active:
            cv2.putText(frame, "LISTENING...", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def get_display_frame(self):
        frame = self.camera_manager.capture_frame()
        if frame is None:
            return None
        
        self.frame_count += 1
        angles, annotated_frame = self.pose_analyzer.analyze_frame(frame)
        return annotated_frame if annotated_frame is not None else frame
    
    def run(self):
        raise NotImplementedError("Subclasses must implement run()")