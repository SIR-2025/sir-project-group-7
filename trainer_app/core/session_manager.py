class SessionManager:
    def __init__(self, sm, agent):
        self.sm = sm
        self.agent = agent
        self.memory = {}
        self.pose = None

    async def process(self, user_text):
        state = self.sm.update_state(user_text)
        reply, actions = await self.agent.generate_reply(
            user_text,
            state,
            self.memory,
            self.pose
        )
        self._update_memory(user_text, reply, actions)
        return reply, actions

    def update_pose(self, pose):
        self.pose = pose

    def _update_memory(self, user_text, reply, actions):
        # Simple personalization memory
        if "like" in user_text.lower():
            self.memory["user_preference"] = user_text
