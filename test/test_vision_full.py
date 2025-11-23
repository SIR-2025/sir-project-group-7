from vision import CameraManager, PoseAnalyzer
from dialogue import DialogueManager
import cv2
import time


def test_camera_only():
    print("\n=== Camera Test ===\n")
    camera = CameraManager(use_local_camera=True)
    camera.test_camera(duration=5)
    camera.cleanup()


def test_pose_only():
    print("\n=== Pose Detection Test ===\n")

    camera = CameraManager(use_local_camera=True)
    pose = PoseAnalyzer(camera_manager=camera)

    print("Do a squat! Press ESC to stop.\n")

    try:
        while True:
            angles, annotated_frame = pose.capture_and_analyze()

            if angles and annotated_frame is not None:
                analysis = pose.check_squat_form(angles)

                cv2.putText(
                    annotated_frame,
                    f"Accuracy: {analysis['overall_accuracy']:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Pose Detection", annotated_frame)
                print(f"Accuracy: {analysis['overall_accuracy']:.1f}%")

            if cv2.waitKey(30) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pose.cleanup()
        camera.cleanup()


def test_with_dialogue_continuous():
    print("\n=== Pose + AI Dialogue Test (Continuous) ===\n")

    camera = CameraManager(use_local_camera=True)
    pose = PoseAnalyzer(camera_manager=camera)
    dialogue = DialogueManager(
        use_local_mic=True,
        camera_manager=camera,
        pose_analyzer=pose
    )

    print("Camera will stay open. Press SPACE to capture and get feedback.")
    print("Press ESC to exit.\n")

    attempt = 0

    try:
        while True:
            frame = camera.capture_frame()

            if frame is not None:
                angles, annotated_frame = pose.analyze_frame(frame)

                if angles and annotated_frame is not None:
                    analysis = pose.check_squat_form(angles)

                    cv2.putText(
                        annotated_frame,
                        f"Accuracy: {analysis['overall_accuracy']:.1f}%",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        "Press SPACE for feedback, ESC to quit",
                        (10, annotated_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    cv2.imshow("Your Form", annotated_frame)
                else:
                    cv2.putText(
                        frame,
                        "No pose detected - stand in view",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    cv2.imshow("Your Form", frame)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            elif key == 32:
                attempt += 1
                print(f"\n=== Attempt {attempt} ===")

                angles, _ = pose.capture_and_analyze()

                if angles:
                    analysis = pose.check_squat_form(angles)
                    print(f"Accuracy: {analysis['overall_accuracy']:.1f}%")

                    feedback = dialogue.get_feedback(analysis, "squat", attempt)
                    print(f"Feedback: {feedback}\n")
                else:
                    print("No pose detected. Try again.\n")

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cv2.destroyAllWindows()
        pose.cleanup()
        camera.cleanup()
        dialogue.cleanup()


def test_with_dialogue():
    print("\n=== Pose + AI Dialogue Test ===\n")

    camera = CameraManager(use_local_camera=True)
    pose = PoseAnalyzer(camera_manager=camera)
    dialogue = DialogueManager(
        use_local_mic=True,
        camera_manager=camera,
        pose_analyzer=pose
    )

    print("Do a squat and I'll give feedback!\n")

    try:
        for i in range(3):
            print(f"\nAttempt {i + 1}/3")
            input("Press Enter to capture your form...")

            angles, annotated_frame = pose.capture_and_analyze()

            if angles:
                analysis = pose.check_squat_form(angles)

                print(f"Accuracy: {analysis['overall_accuracy']:.1f}%")

                if annotated_frame is not None:
                    cv2.imshow("Your Form", annotated_frame)
                    cv2.waitKey(2000)
                    cv2.destroyAllWindows()

                feedback = dialogue.get_feedback(analysis, "squat", i + 1)
                print(f"Feedback: {feedback}\n")
            else:
                print("No pose detected. Try again.\n")

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        pose.cleanup()
        camera.cleanup()
        dialogue.cleanup()


def test_voice_and_pose():
    print("\n=== Voice + Pose Test ===\n")

    camera = CameraManager(use_local_camera=True)
    pose = PoseAnalyzer(camera_manager=camera)
    dialogue = DialogueManager(
        use_local_mic=True,
        camera_manager=camera,
        pose_analyzer=pose,
        system_prompt="""You are an encouraging fitness trainer.
You can see the user's form through the camera.
Give brief, specific feedback. Max 20 words."""
    )

    print("Commands:")
    print("- Say 'check my form' - I'll analyze your current pose")
    print("- Say 'describe' - I'll describe what I see")
    print("- Ask any question - I'll answer")
    print("\nCamera feed is live. Press ESC to exit.\n")

    attempt = 0

    try:
        while True:
            frame = camera.capture_frame()

            if frame is not None:
                angles, annotated_frame = pose.analyze_frame(frame)

                display_frame = annotated_frame if annotated_frame is not None else frame

                cv2.putText(
                    display_frame,
                    "Speak your command (listening...)",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                cv2.imshow("Fitness Trainer", display_frame)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            elif key == 32:
                print("\nListening... (5 seconds)")
                user_input = dialogue.listen_and_transcribe(duration=5.0)

                if user_input:
                    print(f"You: {user_input}")

                    if "form" in user_input.lower() or "check" in user_input.lower():
                        attempt += 1
                        feedback = dialogue.get_pose_feedback("squat", show_frame=False)
                        print(f"Trainer: {feedback}\n")
                    elif "describe" in user_input.lower() or "see" in user_input.lower():
                        response = dialogue.describe_image()
                        print(f"Trainer: {response}\n")
                    else:
                        response = dialogue._get_response(user_input)
                        print(f"Trainer: {response}\n")
                else:
                    print("No speech detected.\n")

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cv2.destroyAllWindows()
        pose.cleanup()
        camera.cleanup()
        dialogue.cleanup()


def main():
    import sys

    print("\n" + "=" * 60)
    print("Vision System Full Test")
    print("=" * 60)

    if len(sys.argv) > 1:
        if sys.argv[1] == "camera":
            test_camera_only()
        elif sys.argv[1] == "pose":
            test_pose_only()
        elif sys.argv[1] == "dialogue":
            test_with_dialogue_continuous()
        elif sys.argv[1] == "voice":
            test_voice_and_pose()
    else:
        print("\nUsage:")
        print("  python test_vision_full.py camera    - Test camera only")
        print("  python test_vision_full.py pose      - Test pose detection")
        print("  python test_vision_full.py dialogue  - Test pose + AI (continuous)")
        print("  python test_vision_full.py voice     - Test voice + pose\n")

        choice = input("Choose (camera/pose/dialogue/voice): ").lower()

        if choice == "camera":
            test_camera_only()
        elif choice == "pose":
            test_pose_only()
        elif choice == "dialogue":
            test_with_dialogue_continuous()
        elif choice == "voice":
            test_voice_and_pose()


if __name__ == "__main__":
    main()