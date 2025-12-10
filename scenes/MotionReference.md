# Quick Reference: Coach Nao Motion Integration

## Import and Setup

```python
from NewMotions import CoachNaoMotions

class YourScene(BaseScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motions = CoachNaoMotions(nao=self.nao if self.use_nao else None)
```

## Common Motion Patterns

### Basic Gesture
```python
if self.use_nao and self.nao:
    self.motions.point_forward()
```

### Gesture + LED
```python
if self.use_nao and self.nao:
    self.motions.set_led_emotion("excited")
    self.motions.celebration_gesture()
```

### Before Speech
```python
if self.use_nao and self.nao:
    self.motions.detailed_explanation()

self.nao_speak("Here's the plan...", wait=True)
```

### After Speech
```python
self.nao_speak("Great job!", wait=True)

if self.use_nao and self.nao:
    time.sleep(0.5)
    self.motions.thumbs_up_equivalent()
```

## Quick Motion Selector

**Need enthusiasm?** → `celebration_gesture()` or `encouragement_sequence()`

**Need encouragement?** → `encouraging_nod()` or `thumbs_up_equivalent()`

**Need comedy?** → `confused_shrug()`, `defeated_slump()`, `point_confidently_wrong()`

**Need to point/direct?** → `point_forward()` or `confident_presentation()`

**Need professionalism?** → `detailed_explanation()` or `polite_bow()`

**Need self-awareness?** → `self_reference()`, `embarrassed_look_away()`

**Need emotion?** → `set_led_emotion("excited"/"calm"/"embarrassed"/"focused")`

**Need bonding?** → `reflective_pause()`, `hopeful_reach()`, `bonding_moment_sequence()`

## LED Colors

```python
"excited"      # Bright green - celebration, joy
"calm"         # Soft blue - reflection, peace
"embarrassed"  # Warm amber - caught, sheepish
"focused"      # White-ish - professional, attentive
"disappointed" # Dim red - correction, concern
"neutral"      # Cyan - default state
```

## Pre-Built Sequences

**Full sequences that combine multiple gestures:**

- `trainer_intro_sequence()` - Energetic entrance
- `exercise_demo_sequence()` - Demo with self-awareness
- `encouragement_sequence()` - Celebrate success
- `confused_navigation_sequence()` - Lost robot comedy
- `bonding_moment_sequence()` - Emotional connection

## Timing Tips

```python
# Quick gesture
self.motions.point_forward()  # ~1 second

# Medium sequence  
self.motions.celebration_gesture()  # ~2 seconds

# Full sequence
self.motions.encouragement_sequence()  # ~3-4 seconds

# Add pauses between gestures
self.motions.first_gesture()
time.sleep(0.5)  # Breathing room
self.motions.second_gesture()
```

## Available NAO Animations (Built-in)

**Greetings:**
- `animations/Stand/Gestures/Hey_1` - Wave
- `animations/Stand/Gestures/Hey_6` - Alternative wave

**Affirmative:**
- `animations/Stand/Gestures/Yes_1/2/3` - Nod variations

**Negative:**
- `animations/Stand/Gestures/No_3/8/9` - Shake variations

**Explaining:**
- `animations/Stand/Gestures/Explain_1` through `Explain_11` - Various explanatory gestures

**Enthusiasm:**
- `animations/Stand/Gestures/Enthusiastic_4/5` - Excited gestures

**Self-reference:**
- `animations/Stand/Gestures/Me_1/2` - Point to self

**Pointing:**
- `animations/Stand/Gestures/You_1/4` - Point at others

**Confusion:**
- `animations/Stand/Gestures/IDontKnow_1/2` - Shrug

**Body Language:**
- `animations/Stand/BodyTalk/BodyTalk_1` through `BodyTalk_22` - Expressive movements

**Polite:**
- `animations/Stand/Gestures/BowShort_1` - Short bow
- `animations/Stand/Gestures/Please_1` - Pleading

**Postures:**
- `NaoPostureRequest("Stand", 0.8)` - Stand up
- `NaoPostureRequest("Sit", 0.8)` - Sit down

## Common Combinations

**Enthusiastic Agreement:**
```python
self.motions.set_led_emotion("excited")
self.motions.encouraging_nod()
time.sleep(0.3)
self.motions.celebration_gesture()
```

**Embarrassed Admission:**
```python
self.motions.set_led_emotion("embarrassed")
self.motions.embarrassed_look_away()
time.sleep(0.5)
self.motions.self_reference()
```

**Professional Explanation:**
```python
self.motions.set_led_emotion("focused")
self.motions.detailed_explanation()
```

**Comedic Lost Moment:**
```python
self.motions.set_led_emotion("embarrassed")
self.motions.look_around_confused()
time.sleep(0.5)
self.motions.point_confidently_wrong()
```

**Grateful Farewell:**
```python
self.motions.set_led_emotion("calm")
self.motions.polite_bow()
time.sleep(0.5)
self.motions.grateful_farewell()
```

## Scene-Specific Highlights

### Scene 1 (Greeting)
- `enthusiastic_entrance()` - Opening
- `point_forward()` - "Stand here"
- `thumbs_up_equivalent()` - Ready confirmation

### Scene 2 (Warmup)
- `arm_stretch_demo()` - Physical demo
- `robotic_warmup()` - Comedic movements
- `self_reference()` - Defensive

### Scene 3 (Introduction)
- `count_gesture_sequence(1/2/3)` - Each exercise
- `celebration_gesture()` - "Let's start!"

### Scene 4 (Squats)
- `attempt_squat()` - Full sit/stand demo
- `encouragement_sequence()` - After completion
- `reflective_pause()` - "Why do I do this?"

### Scene 5 (Push-ups)
- `confused_shrug()` - Identity confusion
- `look_around_confused()` - Wrong apartment
- `pleading_gesture()` - "Can I stay?"

### Scene 6 (Plank)
- `firm_no()` - "Focus!"
- `defeated_slump()` - Admitting fault
- `celebration_gesture()` - Finish

### Scene 7 (Finale)
- `point_confidently_wrong()` - Navigation fail
- `bonding_moment_sequence()` - Emotional beat
- `hopeful_reach()` - "Train again?"
- `confused_shrug()` - Final exit

## Troubleshooting

**Gesture not showing?**
- Check `if self.use_nao and self.nao:` guard
- Ensure NAO is in Stand posture
- Wait for previous animation to complete

**LED not changing?**
- Verify `FaceLeds` are turned on
- Check RGB values are 0-1 range
- May need brief delay to be visible

**Timing feels off?**
- Add `time.sleep()` between gestures
- Let animations fully complete before next action
- Consider speech duration in timing

**Motion interrupting speech?**
- Use `wait=True` on speech
- Or place gesture after speech completes
- Avoid animations during critical dialogue

## File Locations

**Motion Library:** `/path/to/NewMotions.py`  
**Scene Files:** `scene_X_enhanced.py` (X = 1-7)  
**Documentation:** `MOVEMENT_SYSTEM_DOCUMENTATION.md`

---

**Remember:** Less is more! Choose gestures that truly enhance the moment rather than adding motion for motion's sake. The best performances balance movement and stillness.