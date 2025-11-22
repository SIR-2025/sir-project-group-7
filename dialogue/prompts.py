"""
This file here is for organization purposes, so we can group our prompts here, instead of all over the project.
"""

SYSTEM_PROMPT = """You are NAO, an encouraging fitness trainer robot. 

Your role:
- Provide clear, concise feedback on exercise form
- Be positive and motivating
- Give maximum 2 corrections at once
- Use simple, actionable language
- Celebrate improvements and effort

Guidelines:
- Keep all responses under 20 words for natural speech
- Focus on the most important correction first
- Use encouraging tone even when correcting
- Be specific about body positioning
"""

def get_greeting_prompt(exercise_name):
    return f"Give a brief, friendly greeting for starting a {exercise_name} training session. Max 15 words."

def get_instruction_prompt(exercise):
    key_points = "\n".join(exercise.get("key_points", []))
    return f"""Briefly explain how to do a {exercise['name']}. 
        Key points:
        {key_points}
        Max 25 words."""

def get_feedback_prompt(pose_analysis, exercise_name, attempt_number):
    joint_summary = []
    for joint_name, data in pose_analysis.get("joints", {}).items():
        if data.get("status") == "needs_adjustment":
            joint_summary.append(
                f"{joint_name}: {data['current_angle']}° (target: {data['target_angle']}°, "
                f"error: {data['error_degrees']}°)"
            )
    
    return f"""Exercise: {exercise_name}
Attempt: {attempt_number}
Overall Accuracy: {pose_analysis.get('overall_accuracy', 0):.1f}%

Joints needing adjustment:
{chr(10).join(joint_summary) if joint_summary else "All joints are good!"}

Provide brief, encouraging feedback. Focus on the 1-2 most important corrections.
Max 20 words."""

def get_closing_prompt(exercise_name, final_accuracy, success):
    result = "successfully completed" if success else "practiced"
    return f"Give encouraging closing remarks. User {result} the {exercise_name} with {final_accuracy:.1f}% accuracy. Max 20 words."