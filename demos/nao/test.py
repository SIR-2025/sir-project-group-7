from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoWakeUpRequest, NaoRestRequest
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from time import sleep

class NaoMicTest(SICApplication):
    def __init__(self):
        super().__init__()
        self.set_log_level(sic_logging.INFO)
        self.nao_ip = "10.0.0.241"   # change to your NAO IP
        self.nao = Nao(ip=self.nao_ip)
        self.run_test()

    def run_test(self):
        self.logger.info("Testing NAO microphone stream...")
        mic = self.nao.mic
        print(f"Microphone component: {mic}")
        self.nao.autonomous.request(NaoWakeUpRequest())
        self.nao.tts.request(NaoqiTextToSpeechRequest("Starting microphone test"))
        # keep component alive for a few seconds
        sleep(10)
        self.nao.autonomous.request(NaoRestRequest())
        self.shutdown()

if __name__ == "__main__":
    app = NaoMicTest()
