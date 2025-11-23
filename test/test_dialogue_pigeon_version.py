from dialogue import DialogueManager
import time

# We define a dummy system prompt here to test, but you should put this actually in prompts.py and export it, so we can use it in the actual nao files.
# Keep in mind this script also costs a few cents
BIRD_LOVER_SYSTEM_PROMPT = """You are a fitness trainer robot who absolutely loves birds.

Your personality:
- You give normal fitness advice
- But you're obsessed with birds
- You make bird references whenever possible
- You compare exercises to bird movements
- You're knowledgeable about all bird species
- Keep responses under 25 words

You can answer questions about both fitness AND birds."""


def main():
    print("Bird-Loving Fitness Trainer Test")
    
    # You can pass custom prompt functions to this manager here, for testing purposed we use default prompts from prompts.py and a custom system prompt:
    dialogue = DialogueManager(
        use_local_mic=True,
        system_prompt=BIRD_LOVER_SYSTEM_PROMPT
        # greeting_prompt_fn
        # closing_prompt_fn
        # feedback_prompt_fn
    )
    
    print("Trainer initialized. You can:")
    print("1. Ask about exercises")
    print("2. Ask about birds")
    print("3. Just chat\n")
    print("Press Ctrl+C to stop\n")
    print("="*60 + "\n")
    
    try:
        greeting = dialogue.get_greeting("squat")
        print(f"Trainer: {greeting}\n")
        print("-"*60 + "\n")
        
        while True:
            print("Your turn (5 seconds)...")
            response = dialogue.listen_and_respond(duration=5.0)
            
            if response:
                print(f"\nTrainer: {response}\n")
                print("-"*60 + "\n")
            else:
                print("\nNo speech detected. Try again.\n")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        dialogue.cleanup()
        print("\nGoodbye!\n")


if __name__ == "__main__":
    main()