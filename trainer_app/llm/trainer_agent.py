from llm.openai_client import OpenAIClient

class TrainerAgent:
    def __init__(self):
        self.client = OpenAIClient()
        self.system_prompt = """
You are a friendly, motivating **fitness personal trainer**.
Use social robotics principles:
- Consistent persona
- Motivational tone
- Memory references (if provided)
- Explain clearly
- Encourage, never judge
- Keep replies concise
"""

    async def generate_reply(self, user_text, state, memory, pose):
        mem_text = ""
        if "user_preference" in memory:
            mem_text = f"(User previously liked: {memory['user_preference']})"

        pose_text = ""
        if pose is not None:
            pose_text = "User is performing movement. Provide feedback."

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{user_text}\n{mem_text}\n{pose_text}"}
        ]

        reply = await self.client.ask(messages)

        actions = []
        if "show me" in user_text.lower():
            actions.append("vision_required")
        if "encourage" in reply.lower():
            actions.append("motivate")

        return reply, actions
