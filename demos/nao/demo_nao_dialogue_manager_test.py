from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest, NaoPostureRequest

from dialogue import DialogueManager

"""
This file might not work and still needs to be tested with an actual robot during a lab session:
"""
class NaoFitnessTrainerDemo(SICApplication):
    
    def __init__(self):
        super(NaoFitnessTrainerDemo, self).__init__()
        
        self.nao_ip = "10.0.0.241"
        self.nao = None
        self.dialogue_manager = None
        
        self.set_log_level(sic_logging.INFO)
        self.setup()
    
    def setup(self):
        self.logger.info("Initializing NAO robot...")
        
        self.nao = Nao(ip=self.nao_ip, dev_test=True)
        
        self.logger.info("Initializing DialogueManager...")
        self.dialogue_manager = DialogueManager(nao=self.nao)
        
        self.logger.info("Setup complete")
    
    def run(self):
        try:
            self.nao.tts.request(NaoqiTextToSpeechRequest("Hello, I am Nao, your fitness trainer!"))
            self.logger.info("Ready")
            
            while not self.shutdown_event.is_set():
                self.logger.info("Your turn to talk")
                
                user_input = self.dialogue_manager.listen_and_transcribe(duration=5.0)
                
                if user_input:
                    self.logger.info(f"User said: {user_input}")
                    
                    response = self.dialogue_manager._get_response(user_input)
                    
                    if response:
                        self.logger.info(f"NAO reply: {response}")
                        self.nao.tts.request(NaoqiTextToSpeechRequest(response))
                        
                        if "hello" in user_input.lower() or "hi" in user_input.lower():
                            self.logger.info("Greeting detected - performing wave gesture")
                            self.nao.motion.request(NaoPostureRequest("Stand", 0.5), block=False)
                            self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Hey_1"), block=False)
                else:
                    self.logger.info("No speech detected")
                    
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error(f"Exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()


if __name__ == "__main__":
    demo = NaoFitnessTrainerDemo()
    demo.run()