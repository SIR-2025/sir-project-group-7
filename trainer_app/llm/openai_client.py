from openai import OpenAI
from config import OPENAI_API_KEY, MODEL_NAME

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    async def ask(self, messages):
        completion = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        
        # Correct attribute access for new API
        return completion.choices[0].message.content
