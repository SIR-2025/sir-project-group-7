class TrainerStateMachine:
    def __init__(self):
        self.state = "GREETING"

    def update_state(self, text):
        t = text.lower()
        if self.state == "GREETING":
            self.state = "ASK_WORKOUT"
        elif "push" in t or "squat" in t or "abs" in t:
            self.state = "GUIDING_WORKOUT"
        elif "stop" in t:
            self.state = "COOLDOWN"
        return self.state
