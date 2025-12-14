# NAO Fitness Trainer

Theater performance where a NAO robot acts as a fitness coach. Runs in laptop-only mode for testing or with the NAO robot for performance.

## Modules

### dialogue/dialogue_manager.py
Handles conversation and audio I/O.
- Records audio from NAO mic or laptop mic
- Transcribes speech using Whisper
- Generates responses using GPT
- Detects silence to know when user stops talking
- Sends responses to NAO for text-to-speech
- Analyzes camera images with vision models

### dialogue/prompts.py
Character definition and scene contexts.
- System prompt defining Coach NAO's personality
- Context for each of the 7 scenes
- Helper functions for greetings, instructions, feedback

### scene_manager.py
Main application orchestrator.
- Initializes NAO, dialogue, camera, and pose detection
- Runs scenes in sequence
- Handles command-line arguments
- Manages cleanup

### vision/camera_manager.py
Camera input handler.
- Captures frames from laptop webcam, iPhone (Continuity Camera), or NAO camera
- Background thread for continuous capture
- Encodes frames to base64 for vision API
- Lists available cameras

### vision/pose_analyzer.py
Body pose detection using MediaPipe.
- Detects body landmarks in frames
- Calculates joint angles (knees, hips, back)
- Checks squat form and provides accuracy scores
- Draws skeleton overlay on frames

### utils.py
Configuration loader.
- Reads settings from conf/.env
- Validates configuration with Pydantic
- Provides type-safe access to API keys and parameters

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp conf/.env.example conf/.env
```

Edit `conf/.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Download MediaPipe model
Since this is committed to the repository, this is not needed.

## Running

### List audio devices
```bash
python scene_manager.py --list-devices
```

### Laptop only
```bash
python scene_manager.py --mode laptop --mic laptop --camera local
```

### NAO with laptop mic and iPhone camera
```bash
python scene_manager.py --mode nao --mic laptop --camera local
```

### NAO with NAO mic and iPhone camera
```bash
python scene_manager.py --mode nao --mic nao --camera local
```

### Full NAO mode
```bash
python scene_manager.py --mode nao --mic nao --camera nao
```

### Run specific scenes
```bash
python scene_manager.py --mode laptop --scenes 1 2 3
```

### Use external microphone
```bash
# List devices first
python scene_manager.py --list-devices

# Use device index
python scene_manager.py --mode laptop --mic laptop --mic-device 2
```

## Command-Line Options

```
--mode {nao,laptop}      Robot mode or laptop testing
--nao-ip IP              NAO IP address (default: 10.0.0.241)
--mic {nao,laptop}       Microphone source
--mic-device INDEX       External mic device index
--camera {nao,local}     Camera source
--scenes N [N ...]       Which scenes to run (default: all)
--list-devices           List audio devices and exit
```

## Troubleshooting

**OPENAI_API_KEY not found**
- Verify `conf/.env` exists with your API key

**Camera not working**
- List cameras: `python -c "from vision.camera_manager import CameraManager; CameraManager.list_cameras()"`
- Try different camera index

**Microphone not working**
- Run `--list-devices` to see available inputs
- Try different `--mic-device` index

**MediaPipe model missing**
- Check `pose_landmarkers/pose_landmarker_full.task` exists
- Re-download if needed

**NAO connection issues**
- Verify IP: `ping 10.0.0.241`
- Check NAO is powered on and on same network