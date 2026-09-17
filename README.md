# AI Squat Virtual Assistant

## Project Overview

The AI Squat Virtual Assistant is a real-time exercise monitoring system that uses computer vision and artificial intelligence techniques to detect squat poses through a webcam.

The system uses MediaPipe Pose Landmarker to identify body landmarks and calculates joint angles to determine squat posture. It provides real-time corrective feedback through voice prompts using the pyttsx3 text-to-speech library.

## Aim

To develop an AI-based Virtual Assistant system that detects exercise poses in real-time and provides corrective feedback through voice prompts.

## Problem Analysis

Incorrect exercise form can lead to ineffective workouts and may increase the risk of injury.

With the rise of home workouts and online fitness programs, there is a growing need for accessible tools that can provide real-time feedback on exercise form.

This project provides an AI-based solution that detects squat posture and gives immediate feedback through visual and voice prompts.

## Features

- Real-time squat pose detection
- Webcam-based exercise monitoring
- MediaPipe pose landmark detection
- Knee and hip angle calculation
- Automatic squat repetition counting
- Workout timer
- Voice-based corrective feedback
- Real-time visual feedback
- Reset option for repetitions and timer

## Technologies Used

- Python
- OpenCV
- MediaPipe
- pyttsx3
- NumPy

## Working

The system works according to the following steps:

1. Access the webcam.
2. Capture the user's video frame.
3. Detect the human body using MediaPipe.
4. Extract shoulder, hip, knee and ankle landmarks.
5. Calculate knee and hip angles.
6. Analyze the squat posture.
7. Provide corrective feedback.
8. Count completed squat repetitions.
9. Track workout time.
10. Provide feedback using voice prompts.

## Voice Feedback

The system provides voice feedback such as:

- "Lower slowly into your squat."
- "Good depth. Push through your heels."
- "Keep your knees in line with your toes."
- "Keep your chest more upright."
- "Great squat. Rep completed."
- "Counter and timer reset."

## Controls

| Key | Function |
|---|---|
| `q` | Quit the application |
| `r` | Reset repetitions and timer |

## Installation

Clone the repository:

```bash
git clone https://github.com/zenamghevariya-2006/AI-Squat-Virtual-Assistant.git