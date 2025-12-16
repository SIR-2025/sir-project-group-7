import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sic_framework.core.sic_application import SICApplication
from sic_framework.devices import Nao
from sic_framework.devices.common_naoqi.naoqi_autonomous import NaoWakeUpRequest, NaoRestRequest
from dialogue.dialogue_manager import DialogueManager
from vision.camera_manager import CameraManager
from vision.pose_analyzer import PoseAnalyzer
import sounddevice as sd
import time


class SceneManager(SICApplication):
    def __init__(self, use_nao=True, nao_ip="10.0.0.241", use_nao_mic=True, use_nao_camera=False):
        super(SceneManager, self).__init__()
        self.use_nao = use_nao
        self.nao_ip = nao_ip
        self.use_nao_mic = use_nao_mic  # Whether to use NAO's mic or laptop mic
        self.use_nao_camera = use_nao_camera  # Whether to use NAO's camera or local camera
        
        self.nao = None
        self.dialogue_manager = None
        self.camera_manager = None
        self.pose_analyzer = None
        self.scenes = []
        
        self.setup_resources()
    
    def setup_resources(self):
        if self.use_nao:
            self.nao = Nao(ip=self.nao_ip)
            self.nao.autonomous.request(NaoWakeUpRequest())
        
        self.dialogue_manager = DialogueManager(
            nao=self.nao if self.use_nao else None,
            use_local_mic=(args.mic == 'laptop'),  # False = NAO mic, True = laptop mic
            mic_device_index=args.mic_device, 
        )
        
        if self.use_nao_camera and self.nao:
            self.camera_manager = CameraManager(
                use_local_camera=False,
                nao=self.nao,
                use_threading=True
            )
        else:
            self.camera_manager = CameraManager(
                use_local_camera=True, 
                camera_index=0,  # 1 = iPhone via Continuity Camera
                use_threading=True
            )
            
            # Fallback to laptop camera if iPhone not available
            if not self.camera_manager.is_available():
                self.camera_manager.cleanup()
                self.camera_manager = CameraManager(
                    use_local_camera=True,
                    camera_index=0,
                    use_threading=True
                )
        
        self.pose_analyzer = PoseAnalyzer(camera_manager=self.camera_manager)
    
    def register_scene(self, scene_class):
        self.scenes.append(scene_class)
    
    def run_all_scenes(self):
        """Run all scenes in sequence"""
        for i, scene_class in enumerate(self.scenes, 1):
            print(f"\n>>> Starting Scene {i}/{len(self.scenes)}")
            
            try:
                # Create and run the scene
                scene = scene_class(
                    nao=self.nao,
                    dialogue_manager=self.dialogue_manager,
                    camera_manager=self.camera_manager,
                    pose_analyzer=self.pose_analyzer,
                    use_nao=self.use_nao
                )
                
                scene.run()
                
                print(f"<<< Scene {i} Complete\n")
                
            except Exception as e:
                print(f"Error in scene {i}: {e}")
                import traceback
                traceback.print_exc()
                
            finally:
                if 'scene' in locals():
                    scene.cleanup()
                time.sleep(1)  # Brief pause between scenes
    
    def cleanup_resources(self):
        if self.dialogue_manager:
            self.dialogue_manager.cleanup()
        if self.camera_manager:
            self.camera_manager.cleanup()
        if self.pose_analyzer:
            self.pose_analyzer.cleanup()
        if self.use_nao and self.nao:
            self.nao.autonomous.request(NaoRestRequest())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NAO Fitness Trainer Performance")
    parser.add_argument("--mode", choices=["nao", "laptop"], default="laptop",
                       help="Mode: 'nao' for NAO robot, 'laptop' for laptop-only testing")
    parser.add_argument("--nao-ip", default="10.0.0.241",
                       help="NAO robot IP address")
    parser.add_argument("--mic", choices=["nao", "laptop"], default="nao",
                       help="Microphone: 'nao' for NAO mic, 'laptop' for laptop mic (default: nao)")
    parser.add_argument('--mic-device', type=int, default=None,
                        help='Audio device index for external microphone (use list-devices to see options)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument("--camera", choices=["nao", "local"], default="local",
                       help="Camera: 'nao' for NAO camera, 'local' for laptop/iPhone (default: local)")
    parser.add_argument("--scenes", nargs="+", default=["1", "2", "3", "4", "5", "6", "7"],
                       help="Which scenes to run (e.g., --scenes 1 2 3)")
    
    args = parser.parse_args()

    if args.list_devices:
        print("Available Audio Devices:")
        print("=" * 50) 
        print(sd.query_devices())
        sys.exit(0)

    use_nao = (args.mode == "nao")
    use_nao_mic = (args.mic == "nao")
    use_nao_camera = (args.camera == "nao")
    
    print("NAO FITNESS TRAINER - CONFIGURATION")
    print(f"Robot Mode: {'NAO' if use_nao else 'Laptop (no robot)'}")
    print(f"Speech/TTS: {'NAO' if use_nao else 'Text only'}")
    print(f"Microphone: {'NAO' if use_nao_mic else 'Laptop'}")
    print(f"Camera: {'NAO' if use_nao_camera else 'Local (iPhone/Laptop)'}")
    if use_nao:
        print(f"NAO IP: {args.nao_ip}")
    print(f"Scenes: {', '.join(args.scenes)}")
    
    from scene_1 import Scene1
    from scene_2 import Scene2
    from scene_3 import Scene3
    from scene_4 import Scene4
    from scene_5 import Scene5
    from scene_6 import Scene6
    from scene_7 import Scene7
    
    manager = SceneManager(use_nao=use_nao, nao_ip=args.nao_ip, 
                          use_nao_mic=use_nao_mic, use_nao_camera=use_nao_camera)
    
    scene_map = {"1": Scene1, "2": Scene2, "3": Scene3, "4":Scene4, "5":Scene5, "6":Scene6, "7":Scene7}
    
    for scene_num in args.scenes:
        if scene_num in scene_map:
            manager.register_scene(scene_map[scene_num])
    
    manager.run_all_scenes()