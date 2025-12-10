"""
NewMotions.py - Custom Motion Library for Coach Nao Performance

This module contains custom motion sequences and choreography utilities
that extend NAO's built-in animations for theatrical performance.
"""

from sic_framework.devices.common_naoqi.naoqi_motion import (
    NaoPostureRequest,
    NaoqiAnimationRequest,
)
from sic_framework.devices.common_naoqi.naoqi_leds import (
    NaoFadeRGBRequest,
    NaoLEDRequest,
)
import time


class CoachNaoMotions:
    """
    Custom motion sequences for the Coach Nao performance.
    All motions are designed to be theatrical, expressive, and comedic.
    """
    
    def __init__(self, nao=None):
        self.nao = nao
    
    # ==================== GREETING & WELCOMING ====================
    
    def enthusiastic_entrance(self):
        """
        Energetic entrance with wave and fist pump
        """
        if not self.nao:
            return
        
        # Wave hello
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Hey_1"))
        time.sleep(2)
        
        # Enthusiastic gesture
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Enthusiastic_4"))
        time.sleep(1.5)
    
    def awkward_wave(self):
        """
        Slightly delayed, awkward wave - good for shy/uncertain moments
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Hey_6"))
        time.sleep(0.5)
        # Add a small pause - awkwardness!
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_3"))
    
    # ==================== EXPLAINING & DEMONSTRATING ====================
    
    def detailed_explanation(self):
        """
        Animated explanation with multiple gestures - perfect for workout instructions
        """
        if not self.nao:
            return
        
        # Use varied explain animations
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Explain_2"))
        time.sleep(1)
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Explain_7"))
    
    def point_forward(self):
        """
        Points forward at the trainee - used for direct instructions
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/You_1"))
        time.sleep(1.5)
    
    def confident_presentation(self):
        """
        Confident trainer stance with sweeping arm gesture
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Explain_10"))
        time.sleep(1)
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Explain_11"))
    
    # ==================== ENCOURAGEMENT & MOTIVATION ====================
    
    def celebration_gesture(self):
        """
        Celebratory arms up - for successful exercise completion
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Enthusiastic_5"))
        time.sleep(1)
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Yes_1"))
    
    def encouraging_nod(self):
        """
        Approving nod with affirmative gesture
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Yes_2"))
        time.sleep(1)
    
    def thumbs_up_equivalent(self):
        """
        NAO's version of thumbs up - encouraging gesture
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Yes_3"))
        time.sleep(0.8)
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Enthusiastic_4"))
    
    # ==================== SELF-AWARE & COMEDIC ====================
    
    def self_reference(self):
        """
        Points to self - for self-deprecating humor
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Me_1"))
        time.sleep(1.5)
    
    def confused_shrug(self):
        """
        "I don't know" gesture - perfect for spatial awareness jokes
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/IDontKnow_1"))
        time.sleep(1.5)
    
    def defeated_slump(self):
        """
        Slight slump in posture - acknowledging limitations
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_12"))
        time.sleep(1)
    
    def embarrassed_look_away(self):
        """
        Turns head slightly away - embarrassment
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_6"))
        time.sleep(1.2)
    
    # ==================== NEGATIVE/CORRECTIVE ====================
    
    def firm_no(self):
        """
        Strong negative gesture - for incorrect form
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/No_3"))
        time.sleep(1.2)
    
    def gentle_correction(self):
        """
        Softer negative with explanation - correcting without harsh judgment
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/No_8"))
        time.sleep(0.8)
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Explain_4"))
    
    # ==================== PHYSICAL DEMONSTRATIONS ====================
    
    def attempt_squat(self):
        """
        NAO's awkward squat attempt - sits down slowly
        """
        if not self.nao:
            return
        
        # Dramatic preparation
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_14"))
        time.sleep(0.5)
        
        # Actually sit
        self.nao.motion.request(NaoPostureRequest("Sit", 0.8))
        time.sleep(2)
        
        # Stand back up
        self.nao.motion.request(NaoPostureRequest("Stand", 0.8))
    
    def arm_stretch_demo(self):
        """
        Demonstrates arm stretches - limited range of motion
        """
        if not self.nao:
            return
        
        # Raise arms
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_15"))
        time.sleep(1)
        
        # Side stretch
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_17"))
    
    def robotic_warmup(self):
        """
        Stiff, mechanical warmup movements - comedic
        """
        if not self.nao:
            return
        
        # Mechanical body movements
        animations = [
            "animations/Stand/BodyTalk/BodyTalk_1",
            "animations/Stand/BodyTalk/BodyTalk_8",
            "animations/Stand/BodyTalk/BodyTalk_13"
        ]
        
        for anim in animations:
            self.nao.motion.request(NaoqiAnimationRequest(anim))
            time.sleep(1.2)
    
    # ==================== POLITE & FORMAL ====================
    
    def polite_bow(self):
        """
        Short respectful bow
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/BowShort_1"))
        time.sleep(1.5)
    
    def pleading_gesture(self):
        """
        Asking/pleading gesture - begging for understanding
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Please_1"))
        time.sleep(1.5)
    
    # ==================== COUNTING & TRACKING ====================
    
    def count_gesture_sequence(self, count):
        """
        Gestures while counting - varies based on count
        """
        if not self.nao:
            return
        
        # Different gestures for different counts
        if count == 1:
            self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/You_1"))
        elif count == 2:
            self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Enthusiastic_4"))
        elif count == 3:
            self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Yes_1"))
        else:
            self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_10"))
    
    # ==================== FINALE & EMOTIONAL ====================
    
    def grateful_farewell(self):
        """
        Warm, grateful goodbye gesture
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/BowShort_1"))
        time.sleep(1)
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Hey_1"))
    
    def reflective_pause(self):
        """
        Thoughtful, contemplative body language
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_11"))
        time.sleep(1.5)
    
    def hopeful_reach(self):
        """
        Reaching out gesture - asking question or offering connection
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_19"))
        time.sleep(1)
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/You_4"))
    
    # ==================== DIRECTIONAL/SPATIAL ====================
    
    def point_confidently_wrong(self):
        """
        Points in a direction with confidence (even if wrong!) - comedic
        """
        if not self.nao:
            return
        
        # Strong pointing gesture
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Explain_1"))
        time.sleep(1)
        # Follow up with "I'm sure!" body language
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Yes_1"))
    
    def look_around_confused(self):
        """
        Looking around as if lost or confused
        """
        if not self.nao:
            return
        
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/BodyTalk/BodyTalk_9"))
        time.sleep(1)
        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/IDontKnow_2"))
    
    # ==================== LED EMOTION STATES ====================
    
    def set_led_emotion(self, emotion="neutral"):
        """
        Set LED colors to match emotional state
        
        Args:
            emotion: "excited", "calm", "embarrassed", "focused", "neutral"
        """
        if not self.nao:
            return
        
        self.nao.leds.request(NaoLEDRequest("FaceLeds", True))
        
        if emotion == "excited":
            # Bright green
            self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 1, 0, 0))
            self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 1, 0, 0))
        
        elif emotion == "calm":
            # Soft blue
            self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0, 0.5, 0))
            self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0, 0.5, 0))
        
        elif emotion == "embarrassed":
            # Warm amber/orange
            self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 1, 0.6, 0, 0))
            self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 1, 0.6, 0, 0))
        
        elif emotion == "focused":
            # White-ish (equal RGB)
            self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0.7, 0.7, 0.7, 0))
            self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0.7, 0.7, 0.7, 0))
        
        elif emotion == "disappointed":
            # Dim red
            self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0.5, 0, 0, 0))
            self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0.5, 0, 0, 0))
        
        else:  # neutral
            # Default cyan
            self.nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", 0, 0.5, 0.5, 0))
            self.nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", 0, 0.5, 0.5, 0))
    
    # ==================== COMBINATION SEQUENCES ====================
    
    def trainer_intro_sequence(self):
        """
        Complete introduction sequence for a fitness trainer
        """
        if not self.nao:
            return
        
        self.set_led_emotion("excited")
        self.enthusiastic_entrance()
        time.sleep(0.5)
        self.confident_presentation()
        self.set_led_emotion("focused")
    
    def exercise_demo_sequence(self):
        """
        Full exercise demonstration sequence with self-awareness
        """
        if not self.nao:
            return
        
        self.set_led_emotion("focused")
        self.detailed_explanation()
        time.sleep(0.5)
        self.self_reference()  # Acknowledging limitations
        time.sleep(0.3)
        self.embarrassed_look_away()
        self.set_led_emotion("neutral")
    
    def encouragement_sequence(self):
        """
        Full encouragement routine - celebrate trainee success
        """
        if not self.nao:
            return
        
        self.set_led_emotion("excited")
        self.celebration_gesture()
        time.sleep(0.3)
        self.thumbs_up_equivalent()
        time.sleep(0.5)
        self.point_forward()  # "You did it!"
    
    def confused_navigation_sequence(self):
        """
        Lost robot sequence - looking around confused
        """
        if not self.nao:
            return
        
        self.set_led_emotion("embarrassed")
        self.look_around_confused()
        time.sleep(0.5)
        self.confused_shrug()
        time.sleep(0.5)
        self.point_confidently_wrong()  # Points wrong way confidently!
    
    def bonding_moment_sequence(self):
        """
        Warm, connecting gesture sequence for emotional scenes
        """
        if not self.nao:
            return
        
        self.set_led_emotion("calm")
        self.reflective_pause()
        time.sleep(0.5)
        self.hopeful_reach()
        time.sleep(0.5)
        self.polite_bow()


# ==================== STANDALONE HELPER FUNCTIONS ====================

def quick_gesture(nao, gesture_type="explain"):
    """
    Quick one-off gesture without instantiating the full class
    
    Args:
        nao: NAO robot instance
        gesture_type: Type of gesture to perform
    """
    if not nao:
        return
    
    gestures = {
        "explain": "animations/Stand/Gestures/Explain_2",
        "yes": "animations/Stand/Gestures/Yes_1",
        "no": "animations/Stand/Gestures/No_3",
        "hey": "animations/Stand/Gestures/Hey_1",
        "enthusiastic": "animations/Stand/Gestures/Enthusiastic_4",
        "me": "animations/Stand/Gestures/Me_1",
        "you": "animations/Stand/Gestures/You_1",
        "confused": "animations/Stand/Gestures/IDontKnow_1",
        "bow": "animations/Stand/Gestures/BowShort_1"
    }
    
    animation = gestures.get(gesture_type, "animations/Stand/BodyTalk/BodyTalk_1")
    nao.motion.request(NaoqiAnimationRequest(animation))


def set_simple_led(nao, color="blue"):
    """
    Quick LED color change
    
    Args:
        nao: NAO robot instance
        color: "red", "green", "blue", "white", "off"
    """
    if not nao:
        return
    
    nao.leds.request(NaoLEDRequest("FaceLeds", True))
    
    colors = {
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "white": (1, 1, 1),
        "cyan": (0, 1, 1),
        "yellow": (1, 1, 0),
        "magenta": (1, 0, 1),
        "off": (0, 0, 0)
    }
    
    r, g, b = colors.get(color, (0, 0, 1))
    nao.leds.request(NaoFadeRGBRequest("RightFaceLeds", r, g, b, 0))
    nao.leds.request(NaoFadeRGBRequest("LeftFaceLeds", r, g, b, 0))