import cv2
import numpy as np
import pyaudio
import wave
import time
import logging
import threading
import json
import os
import base64
import hashlib
import re
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import face_recognition
from flask import Flask, request, jsonify, render_template_string, session
import secrets
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import PyPDF2
from google import genai
from dotenv import load_dotenv
import speech_recognition as sr
import torch
import numpy as np
import wave
import threading
import queue
import time
from urllib.request import urlretrieve
import os



# Add these global variables after your existing global variables
interview_model = None
gemini_client = None
# Import required libraries for monitoring
try:
    import mediapipe as mp
    from ultralytics import YOLO
except ImportError as e:
    logging.error(f"Required libraries not installed: {e}")
    print("Please install required packages: pip install mediapipe ultralytics opencv-python pyaudio numpy face-recognition pillow flask")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("exam_monitoring.log"),
        logging.StreamHandler()
    ]
)

# Global stop event for terminating exam
stop_event = threading.Event()

class DataManager:
    """Handles all JSON data operations for users and monitoring data"""
    
    def __init__(self):
        self.users_file = "users_data.json"
        self.monitoring_file = "monitoring_data.json"
        self.cheating_events_file = "cheating_events.json"
        self.ensure_files_exist()
    
    def ensure_files_exist(self):
        """Create JSON files if they don't exist"""
        files = [self.users_file, self.monitoring_file, self.cheating_events_file]
        for file in files:
            if not os.path.exists(file):
                with open(file, 'w') as f:
                    json.dump([], f)
    
    def load_users(self):
        """Load users from JSON file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def save_users(self, users):
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def add_user(self, user_data):
        """Add a new user"""
        users = self.load_users()
        users.append(user_data)
        self.save_users(users)
    
    def get_user_by_email(self, email):
        """Get user by email"""
        users = self.load_users()
        for user in users:
            if user['email'] == email:
                return user
        return None
    
    def save_monitoring_session(self, session_data):
        """Save monitoring session data"""
        try:
            with open(self.monitoring_file, 'r') as f:
                data = json.load(f)
        except:
            data = []
        
        data.append(session_data)
        
        with open(self.monitoring_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_user_sessions(self, user_id):
        """Get all sessions for a specific user"""
        try:
            with open(self.monitoring_file, 'r') as f:
                data = json.load(f)
            return [session for session in data if session.get('user_id') == user_id]
        except:
            return []
    
    def load_cheating_events(self):
        """Load cheating events from JSON file"""
        try:
            with open(self.cheating_events_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def save_cheating_events(self, events):
        """Save cheating events to JSON file"""
        with open(self.cheating_events_file, 'w') as f:
            json.dump(events, f, indent=2)
    
    def get_or_create_cheating_event(self, student_id, event_type):
        """Get or create a cheating event for a student"""
        events = self.load_cheating_events()
        
        # Find existing event
        for event in events:
            if (event.get('student_id') == student_id and 
                event.get('event_type') == event_type):
                return event, False
        
        # Create new event
        new_event = {
            'id': len(events) + 1,
            'student_id': student_id,
            'event_type': event_type,
            'cheating_flag': False,
            'tab_switch_count': 0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        events.append(new_event)
        self.save_cheating_events(events)
        return new_event, True
    
    def update_cheating_event(self, event_id, updates):
        """Update a cheating event"""
        events = self.load_cheating_events()
        
        for i, event in enumerate(events):
            if event.get('id') == event_id:
                event.update(updates)
                event['updated_at'] = datetime.now().isoformat()
                events[i] = event
                break
        
        self.save_cheating_events(events)
        return event

class AuthenticationSystem:
    """Handles user registration and login with face recognition"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def hash_password(self, password):
        """Hash password for secure storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def get_face_encoding(self, image):
        """Extract face encoding from image"""
        try:
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return None
            return face_recognition.face_encodings(image, face_locations)[0]
        except:
            return None
    
    def register_user(self, name, email, password, photo_path):
        """Register a new user with face encoding"""
        # Check if user already exists
        if self.data_manager.get_user_by_email(email):
            return False, "Email already registered"
        
        # Load and process photo
        try:
            image = cv2.imread(photo_path)
            if image is None:
                return False, "Could not load photo"
            
            face_encoding = self.get_face_encoding(image)
            if face_encoding is None:
                return False, "No face detected in photo"
            
            # Convert image to base64 for storage
            with open(photo_path, "rb") as img_file:
                photo_base64 = base64.b64encode(img_file.read()).decode()
            
            # Create user data
            user_data = {
                'id': len(self.data_manager.load_users()) + 1,
                'name': name,
                'email': email,
                'password': self.hash_password(password),
                'photo': photo_base64,
                'face_encoding': face_encoding.tolist(),
                'registered_at': datetime.now().isoformat()
            }
            
            self.data_manager.add_user(user_data)
            return True, "Registration successful"
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, email, password, captured_image):
        """Login user with email, password, and face verification"""
        user = self.data_manager.get_user_by_email(email)
        if not user:
            return False, "User not found"
        
        # Check password
        if user['password'] != self.hash_password(password):
            return False, "Invalid password"
        
        # Check face recognition
        captured_encoding = self.get_face_encoding(captured_image)
        if captured_encoding is None:
            return False, "No face detected in captured image"
        
        stored_encoding = np.array(user['face_encoding'])
        
        # Compare face encodings
        matches = face_recognition.compare_faces([stored_encoding], captured_encoding, tolerance=0.6)
        if not matches[0]:
            return False, "Face does not match registered photo"
        
        return True, user

class TabSwitchTracker:
    """Handles tab switch detection and tracking"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.max_tab_switches = 5
    
    def record_tab_switch(self, student_id):
        """Record a tab switch event for a student"""
        try:
            # Get or create cheating event
            cheating_event, created = self.data_manager.get_or_create_cheating_event(
                student_id, 'tab_switch'
            )
            
            logging.info(f"Cheating Event: {cheating_event}, Created: {created}")
            
            # Increment tab switch count
            tab_switch_count = cheating_event.get('tab_switch_count', 0) + 1
            
            # Update cheating event
            updates = {
                'tab_switch_count': tab_switch_count,
                'cheating_flag': tab_switch_count >= 1
            }
            
            updated_event = self.data_manager.update_cheating_event(
                cheating_event['id'], updates
            )
            
            logging.info(f"Updated Tab Switch Count: {tab_switch_count}")
            logging.info(f"Cheating Flag: {updates['cheating_flag']}")
            
            # Check if limit exceeded
            if tab_switch_count > self.max_tab_switches:
                stop_event.set()
                logging.info("Tab switches exceeded limit, exam terminated")
                return {
                    "status": "terminated",
                    "message": f"You have exceeded the allowed tab switches ({self.max_tab_switches}). Your exam is terminated.",
                    "count": tab_switch_count,
                    "cheating_flag": True
                }
            
            return {
                "status": "updated",
                "count": tab_switch_count,
                "cheating_flag": updates['cheating_flag'],
                "message": f"Tab switch detected! Total switches: {tab_switch_count}"
            }
            
        except Exception as e:
            logging.error(f"Error recording tab switch: {e}")
            return {
                "status": "error",
                "message": f"Error recording tab switch: {str(e)}"
            }

class AudioInterface:
    """Handles all audio input/output operations using Silero VAD and PyAudio"""
    
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 3
        
        # Initialize Silero VAD
        self.init_silero_vad()
        
        # Recording control
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def init_silero_vad(self):
        """Initialize Silero VAD model"""
        try:
            # Check if model exists, if not download it
            model_path = "silero_vad.jit"
            if not os.path.exists(model_path):
                print("Downloading Silero VAD model...")
                urlretrieve(
                    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.jit",
                    model_path
                )
            
            # Load the model
            self.vad_model = torch.jit.load(model_path)
            self.vad_model.eval()
            print("Silero VAD model loaded successfully")
            
        except Exception as e:
            print(f"Error initializing Silero VAD: {e}")
            self.vad_model = None
            
    def is_speech(self, audio_chunk):
        """Detect speech using Silero VAD"""
        try:
            if self.vad_model is None:
                return True  # Default to true if model isn't loaded
                
            # Convert audio chunk to tensor
            audio_tensor = torch.FloatTensor(audio_chunk)
            
            # Get speech probability
            speech_prob = self.vad_model(audio_tensor, self.RATE)
            
            return speech_prob > 0.5
            
        except Exception as e:
            print(f"Error in speech detection: {e}")
            return True
            
    def start_recording(self):
        """Start audio recording with VAD"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        print("Recording started...")
        
    def _record_audio(self):
        """Internal method to handle continuous audio recording"""
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            print("Listening... Speak now!")
            
            frames = []
            silence_counter = 0
            speech_detected = False
            
            while self.is_recording:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    
                    # Check for speech
                    is_speech = self.is_speech(audio_chunk)
                    
                    if is_speech:
                        if not speech_detected:
                            print("Speech detected!")
                        speech_detected = True
                        silence_counter = 0
                        frames.append(data)
                    elif speech_detected:
                        silence_counter += 1
                        frames.append(data)
                        
                        # After 1 second of silence, save the recording
                        if silence_counter > int(self.RATE / self.CHUNK):
                            if len(frames) > 0:
                                self.save_recording(frames)
                            frames = []
                            speech_detected = False
                            silence_counter = 0
                            
                except Exception as e:
                    print(f"Error reading audio stream: {e}")
                    break
                    
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Error in audio recording: {e}")
            
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=1)
            
    def save_recording(self, frames):
        """Save recorded audio to WAV file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recorded_audio_{timestamp}.wav"
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paFloat32))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
                
            print(f"Recording saved as {filename}")
            
            # Attempt speech recognition
            text = self.transcribe_audio(filename)
            if text:
                print(f"Transcribed text: {text}")
                return text
                
        except Exception as e:
            print(f"Error saving recording: {e}")
            
    def transcribe_audio(self, audio_file):
        """Convert speech to text using Google Speech Recognition"""
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                return text
                
        except sr.UnknownValueError:
            print("Speech Recognition could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Speech Recognition service: {e}")
            return None
            
    def play_audio(self, audio_file):
        """Play audio file using PyAudio"""
        try:
            wf = wave.open(audio_file, 'rb')
            
            stream = self.audio.open(
                format=self.audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            data = wf.readframes(self.CHUNK)
            while data:
                stream.write(data)
                data = wf.readframes(self.CHUNK)
                
            stream.stop_stream()
            stream.close()
            wf.close()
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            
    def cleanup(self):
        """Cleanup audio resources"""
        try:
            self.stop_recording()
            self.audio.terminate()
        except Exception as e:
            print(f"Error in cleanup: {e}")
            
    def start_listening(self):
        """Start listening to audio input"""
        self.start_recording()
        
    def stop_listening(self):
        """Stop listening to audio input"""
        self.stop_recording()
        
    def speak_text(self, text):
        """Convert text to speech (TTS)"""
        try:
            # This is a placeholder. For actual implementation,
            # you would need to integrate a TTS library like gTTS or pyttsx3
            print(f"Text-to-Speech: {text}")
            # For now, we'll just print the text that would be spoken
            return True
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return False

class InterviewSystem:
    """Real-time AI interview system with speech capabilities"""
    
    def __init__(self, audio_interface, user_data, gemini_client):
        self.audio_interface = audio_interface
        self.user_data = user_data
        self.gemini_client = gemini_client
        self.interview_active = False
        self.current_qa = []
        self.interview_log = []
        
    def start_interview(self, resume_text, job_description):
        """Start the AI interview with speech interaction"""
        try:
            # Generate interview questions
            self.questions = generate_interview_questions(resume_text, job_description)
            self.current_question_idx = 0
            self.interview_active = True
            
            print("\n=== AI Interview Started ===")
            print("The system will ask questions verbally and listen for your responses.")
            print("Speak clearly into your microphone to answer.")
            print("Press 'q' to quit, 'r' to repeat question\n")
            
            # Start the interview loop
            self.conduct_interview()
            
        except Exception as e:
            print(f"Error starting interview: {e}")
            
    def conduct_interview(self):
        """Main interview loop with speech interaction"""
        try:
            while self.interview_active and self.current_question_idx < len(self.questions):
                current_question = self.questions[self.current_question_idx]
                
                # Speak the question
                print(f"\nQuestion {self.current_question_idx + 1}:")
                print(current_question)
                self.audio_interface.speak_text(current_question)
                
                # Record and process answer
                print("\nListening for your answer... (Speak clearly)")
                audio_file = self.record_answer()
                
                if audio_file:
                    # Transcribe the answer
                    answer_text = self.audio_interface.transcribe_audio(audio_file)
                    
                    if answer_text:
                        print(f"\nYour answer: {answer_text}")
                        
                        # Save Q&A pair
                        self.current_qa.append({
                            'question': current_question,
                            'answer': answer_text,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Move to next question
                        self.current_question_idx += 1
                    else:
                        print("\nCould not understand the answer. Please try again.")
                        self.audio_interface.speak_text("I couldn't understand your answer. Please try again.")
                
                # Check for interview completion
                if self.current_question_idx >= len(self.questions):
                    self.complete_interview()
                    
        except KeyboardInterrupt:
            print("\nInterview interrupted by user.")
            self.complete_interview()
            
    def record_answer(self):
        """Record interview answer with voice activity detection"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interview_answer_{timestamp}.wav"
            
            # Configure audio parameters for interview
            CHUNK = 1024
            FORMAT = pyaudio.paFloat32
            CHANNELS = 1
            RATE = 16000
            
            print("Recording... (Speak now)")
            
            # Start recording with voice detection
            frames = []
            silence_counter = 0
            speech_detected = False
            
            stream = self.audio_interface.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            # Record with VAD
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # Check for speech using Silero VAD
                is_speech = self.audio_interface.is_speech(audio_chunk)
                
                if is_speech:
                    if not speech_detected:
                        print("Speech detected!")
                    speech_detected = True
                    silence_counter = 0
                    frames.append(data)
                elif speech_detected:
                    silence_counter += 1
                    frames.append(data)
                    
                    # Stop after 2 seconds of silence
                    if silence_counter > int(RATE / CHUNK * 2):
                        break
                        
                # Maximum answer length: 2 minutes
                if len(frames) > int(RATE / CHUNK * 120):
                    break
                    
            stream.stop_stream()
            stream.close()
            
            # Save the recording
            if len(frames) > 0:
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(self.audio_interface.audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                return filename
                
            return None
            
        except Exception as e:
            print(f"Error recording answer: {e}")
            return None
            
    def complete_interview(self):
        """Complete the interview and provide feedback"""
        self.interview_active = False
        print("\n=== Interview Completed ===")
        
        # Evaluate answers
        score = evaluate_qa_pairs(self.current_qa)
        
        # Generate feedback using Gemini AI
        feedback = self.generate_interview_feedback()
        
        # Save interview results
        self.save_interview_results(score, feedback)
        
        # Speak the feedback
        print("\nInterview Feedback:")
        print(f"Score: {score}/100")
        print("\nDetailed Feedback:")
        print(feedback)
        
        self.audio_interface.speak_text(f"You scored {score} out of 100 points. Here's your feedback: {feedback}")
        
    def generate_interview_feedback(self):
        """Generate detailed interview feedback using Gemini AI"""
        try:
            if not self.gemini_client:
                return "AI feedback generation not available. Please check your API configuration."
                
            # Prepare the prompt
            prompt = f"""
            Analyze this interview and provide constructive feedback:
            
            Candidate: {self.user_data['name']}
            Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Interview Q&A:
            """
            
            for qa in self.current_qa:
                prompt += f"\nQ: {qa['question']}\nA: {qa['answer']}\n"
                
            prompt += """
            Please provide:
            1. Overall performance assessment
            2. Communication skills evaluation
            3. Specific strengths
            4. Areas for improvement
            5. Recommendations for future interviews
            
            Keep the feedback constructive and actionable.
            """
            
            response = self.gemini_client.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return "Error generating AI feedback. Please check the logs."
            
    def save_interview_results(self, score, feedback):
        """Save interview results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interview_results_{timestamp}.json"
            
            results = {
                'candidate': {
                    'name': self.user_data['name'],
                    'email': self.user_data['email']
                },
                'interview_date': datetime.now().isoformat(),
                'qa_pairs': self.current_qa,
                'score': score,
                'feedback': feedback
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"\nInterview results saved to {filename}")
            
        except Exception as e:
            print(f"Error saving interview results: {e}")

class ExamMonitoringSystem:
    """Complete exam monitoring system with all features"""
    
    def __init__(self, user_data, data_manager):
        self.user_data = user_data
        self.data_manager = data_manager
        self.tab_tracker = TabSwitchTracker(data_manager)
        self.audio_interface = AudioInterface()
        self.audio_thread = None
        
        # Audio parameters
        self.AUDIO_THRESHOLD = 2000
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 48000
        self.SOUND_END_DELAY = 4
        
        # Object detection parameters
        self.CONFIDENCE_THRESHOLD = 0.5
        self.RESIZE_WIDTH = 640
        
        # Initialize components
        self.init_audio()
        self.init_computer_vision()
        self.init_object_detection()
        # Initialize audio interface
        self.audio_interface = AudioInterface()
        # Control variables
        self.monitoring = False
        self.audio_thread = None
        
        # Session data
        self.session_data = {
            'user_id': user_data['id'],
            'user_name': user_data['name'],
            'session_start': datetime.now().isoformat(),
            'alerts': {
                'multiple_faces': 0,
                'multiple_persons': 0,
                'cell_phone': 0,
                'book': 0,
                'gaze_away': 0,
                'audio_detected': 0,
                'tab_switches': 0
            },
            'events': []
        }
    
    def init_audio(self):
        """Initialize audio detection system"""
        try:
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            logging.info("Audio system initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize audio system: {e}")
            self.audio = None
            self.audio_stream = None
    
    def init_computer_vision(self):
        """Initialize MediaPipe components"""
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_face_mesh = mp.solutions.face_mesh
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=0.5
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            logging.info("Computer vision components initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize computer vision components: {e}")
    
    def init_object_detection(self):
        """Initialize YOLO model"""
        try:
            self.yolo_model = YOLO("yolo11s.pt")
            logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
    
    def log_event(self, event_type, description, severity="medium"):
        """Log monitoring event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'description': description,
            'severity': severity
        }
        self.session_data['events'].append(event)
        logging.info(f"Event logged: {event_type} - {description}")
    
    def record_tab_switch(self):
        """Record tab switch for current user"""
        result = self.tab_tracker.record_tab_switch(self.user_data['id'])
        
        # Update session data
        self.session_data['alerts']['tab_switches'] = result.get('count', 0)
        
        # Log the event
        self.log_event(
            'tab_switch',
            f"Tab switch detected. Count: {result.get('count', 0)}",
            'high' if result.get('cheating_flag') else 'medium'
        )
        
        return result
    
    def detect_faces(self, frame):
        """Detect faces and return count with annotated frame"""
        if not hasattr(self, 'face_detection'):
            return 0, frame
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_count = 0
            annotated_frame = frame.copy()
            
            detection_results = self.face_detection.process(rgb_frame)
            
            if detection_results.detections:
                face_count = len(detection_results.detections)
                
                for detection in detection_results.detections:
                    self.mp_drawing.draw_detection(annotated_frame, detection)
            
            if face_count > 1:
                cv2.putText(annotated_frame, 'ALERT: Multiple Faces!', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.session_data['alerts']['multiple_faces'] += 1
                self.log_event('multiple_faces', f'{face_count} faces detected', 'high')
            
            return face_count, annotated_frame
            
        except Exception as e:
            logging.error(f"Error in face detection: {e}")
            return 0, frame
    
    def track_gaze(self, frame):
        """Track gaze direction"""
        if not hasattr(self, 'face_mesh'):
            return 'center'
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    left_eye = [landmarks.landmark[33], landmarks.landmark[159]]
                    right_eye = [landmarks.landmark[362], landmarks.landmark[386]]
                    
                    left_eye_center = np.mean([(p.x, p.y) for p in left_eye], axis=0)
                    right_eye_center = np.mean([(p.x, p.y) for p in right_eye], axis=0)
                    
                    gaze_direction = "center"
                    if left_eye_center[0] < 0.4:
                        gaze_direction = "left"
                        self.session_data['alerts']['gaze_away'] += 1
                        self.log_event('gaze_away', 'Looking left', 'medium')
                    elif right_eye_center[0] > 0.6:
                        gaze_direction = "right"
                        self.session_data['alerts']['gaze_away'] += 1
                        self.log_event('gaze_away', 'Looking right', 'medium')
                    
                    return gaze_direction
            
            return "center"
            
        except Exception as e:
            logging.error(f"Error in gaze tracking: {e}")
            return "center"
    
    def detect_objects(self, frame):
        """Detect objects focusing on suspicious items"""
        if self.yolo_model is None:
            return [], frame, 0, []
        
        try:
            labels_detected = []
            suspicious_objects = []
            person_count = 0
            
            height, width = frame.shape[:2]
            if width > self.RESIZE_WIDTH:
                aspect_ratio = height / width
                frame = cv2.resize(frame, (self.RESIZE_WIDTH, int(self.RESIZE_WIDTH * aspect_ratio)))
            
            results = self.yolo_model(frame)
            
            for result in results:
                for box in result.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, score, class_id = box
                    
                    if score > self.CONFIDENCE_THRESHOLD:
                        label = self.yolo_model.names[int(class_id)]
                        labels_detected.append((label, float(score)))
                        
                        if label.lower() == "person":
                            person_count += 1
                            if person_count > 1:
                                suspicious_objects.append("multiple_persons")
                                self.session_data['alerts']['multiple_persons'] += 1
                                self.log_event('multiple_persons', f'{person_count} persons detected', 'high')
                        elif label.lower() == "cell phone":
                            suspicious_objects.append("cell_phone")
                            self.session_data['alerts']['cell_phone'] += 1
                            self.log_event('cell_phone', 'Cell phone detected', 'high')
                        elif label.lower() == "book":
                            suspicious_objects.append("book")
                            self.session_data['alerts']['book'] += 1
                            self.log_event('book', 'Book detected', 'medium')
                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display alerts on frame
            alert_y = 50
            if person_count >= 2:
                cv2.putText(frame, "ALERT: Multiple persons!", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert_y += 30
                
            if "cell_phone" in [obj.replace("_", " ") for obj in suspicious_objects]:
                cv2.putText(frame, "ALERT: Cell phone detected!", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert_y += 30
                
            if "book" in suspicious_objects:
                cv2.putText(frame, "ALERT: Book detected!", (10, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return labels_detected, frame, person_count, suspicious_objects
            
        except Exception as e:
            logging.error(f"Error in object detection: {e}")
            return [], frame, 0, []
    
    def audio_detection_thread(self):
        """Audio monitoring in separate thread"""
        if self.audio_stream is None:
            return
        
        logging.info("Starting audio monitoring...")
        sound_detected = False
        last_sound_time = 0
        
        while self.monitoring and not stop_event.is_set():
            try:
                data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                if np.max(np.abs(audio_data)) > self.AUDIO_THRESHOLD:
                    if not sound_detected:
                        sound_detected = True
                        self.session_data['alerts']['audio_detected'] += 1
                        self.log_event('audio_detected', 'Suspicious audio detected', 'medium')
                    last_sound_time = time.time()
                
                if sound_detected and (time.time() - last_sound_time > self.SOUND_END_DELAY):
                    sound_detected = False
                    
            except Exception as e:
                logging.error(f"Error in audio detection: {e}")
                break
    
    def start_monitoring(self):
        """Start complete monitoring system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logging.error("Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.monitoring = True
        stop_event.clear()  # Clear the stop event
        # Start audio interface
        self.audio_thread = threading.Thread(target=self.audio_detection_thread)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        # Start audio recording
        self.audio_interface.start_recording()
        # Start audio monitoring
        if self.audio_stream is not None:
            self.audio_thread = threading.Thread(target=self.audio_detection_thread)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        
        logging.info("Exam monitoring started. Press 'q' to quit, 's' for stats, 't' to simulate tab switch")
        
        try:
            while self.monitoring and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Perform all detections
                face_count, frame = self.detect_faces(frame)
                gaze_direction = self.track_gaze(frame)
                labels, frame, person_count, suspicious_objects = self.detect_objects(frame)
                
                # Display information on frame
                cv2.putText(frame, f"User: {self.user_data['name']}", (10, frame.shape[0] - 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Gaze: {gaze_direction}", (10, frame.shape[0] - 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Faces: {face_count}", (10, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Tab Switches: {self.session_data['alerts']['tab_switches']}", 
                           (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Check if exam should be terminated
                if stop_event.is_set():
                    cv2.putText(frame, "EXAM TERMINATED - Too many violations!", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Exam Monitoring System', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.show_statistics()
                elif key == ord('t'):  # Simulate tab switch
                    result = self.record_tab_switch()
                    print(f"Tab Switch Result: {result}")
                    
        except KeyboardInterrupt:
            logging.info("Monitoring interrupted")
        finally:
            self.stop_monitoring()
            cap.release()
            cv2.destroyAllWindows()
    
    def show_statistics(self):
        """Display current statistics"""
        print("\n" + "="*50)
        print("EXAM MONITORING STATISTICS")
        print("="*50)
        for alert_type, count in self.session_data['alerts'].items():
            print(f"{alert_type.replace('_', ' ').title()}: {count}")
        print(f"Total Events: {len(self.session_data['events'])}")
        print("="*50)
    
    def stop_monitoring(self):
        """Stop monitoring and save session data"""
        self.monitoring = False
        # Stop audio interface
        if self.audio_interface:
            self.audio_interface.stop_listening()
        
        if self.audio_thread:
            self.audio_thread.join(timeout=1)
        # Stop audio
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
        
        if self.audio is not None:
            try:
                self.audio.terminate()
            except:
                pass
        
        # Save session data
        self.session_data['session_end'] = datetime.now().isoformat()
        self.session_data['terminated_by_violations'] = stop_event.is_set()
        self.data_manager.save_monitoring_session(self.session_data)
        if hasattr(self, 'audio_interface'):
            self.audio_interface.cleanup()
        logging.info("Session data saved successfully")
       
# AI Interview Functions - Add these BEFORE the ExamMonitoringGUI class
def init_interview_components():
    """Initialize AI components for interview system"""
    global gemini_client
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logging.error("Google/Gemini API key not found in environment variables")
            return False
        
        # Configure the AI client
        genai.configure(api_key=api_key)
        gemini_client = genai.GenerativeModel('gemini-pro')
        logging.info("Gemini AI client initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize AI components: {e}")
        return False

# Add placeholder functions for missing implementations
def extract_text_from_file(file_path):
    """Extract text from various file formats"""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        elif file_extension == '.pdf':
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
            
        elif file_extension in ['.docx', '.doc']:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
            
        else:
            logging.error(f"Unsupported file format: {file_extension}")
            return None
            
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return None

def generate_interview_questions(client, resume: str, jd: str, n: int = 6) -> list[str]:
    """Use Gemini to generate n interview questions based on resume and JD."""
    prompt = (
        f"Given the following résumé:\n{resume_text}\n\n"
        f"And the following job description:\n{job_description}\n\n"
        f"Your task is to generate the top {n} most relevant interview questions tailored to the candidate’s skill and experience level, and aligned with the job requirements.\n\n"
        "Instructions:\n"
        "1. Carefully analyze the résumé to determine the candidate's level (e.g., fresher, intermediate, experienced).\n"
        "2. Match the question difficulty and content appropriately.\n"
        f"3. Return only a cleanly numbered list (e.g., 1. ..., 2. ..., etc) of {n} concise, high-quality questions. No introduction or explanation."
    )

    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = resp.text.strip()

    # Extract numbered questions
    questions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and "." in line:
            parts = line.split('.', 1)
            if len(parts) > 1:
                q = parts[1].strip()
                questions.append(q)
        if len(questions) >= n:
            break

    return questions

def evaluate_qa_pairs(qa_pairs):
    """Evaluate interview answers using AI and return a score"""
    global gemini_client
    
    if not gemini_client:
        logging.error("Gemini client not initialized, using basic evaluation")
        return basic_evaluation(qa_pairs)
    
    try:
        # Prepare the evaluation prompt
        qa_text = ""
        for i, qa in enumerate(qa_pairs, 1):
            qa_text += f"Question {i}: {qa['question']}\nAnswer {i}: {qa['answer']}\n\n"
        
        prompt = f"""
Evaluate this interview based on the questions and answers provided. Give a score from 0-100.

Evaluation Criteria:
- Answer relevance and completeness (30%)
- Communication clarity (25%) 
- Technical knowledge shown (20%)
- Professional attitude (15%)
- Examples and specifics provided (10%)

Interview Content:
{qa_text[:2500]}

Provide only a numeric score from 0-100. Consider that some answers might be "Skipped".
"""

        response = gemini_client.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract numeric score
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            score = min(100, max(0, int(numbers[0])))
            logging.info(f"AI evaluation completed with score: {score}")
            return score
        else:
            raise Exception("No score found in AI response")
        
    except Exception as e:
        logging.error(f"Error in AI evaluation: {e}")
        return basic_evaluation(qa_pairs)

def basic_evaluation(qa_pairs):
    """Basic evaluation fallback when AI is not available"""
    total_questions = len(qa_pairs)
    if total_questions == 0:
        return 0
    
    answered_questions = 0
    total_score = 0
    
    for qa in qa_pairs:
        answer = qa['answer'].strip()
        if answer and answer.lower() != 'skipped':
            answered_questions += 1
            # Basic scoring based on answer length and quality indicators
            answer_length = len(answer)
            if answer_length > 100:
                total_score += 85
            elif answer_length > 50:
                total_score += 70
            elif answer_length > 20:
                total_score += 55
            else:
                total_score += 40
        # Skipped questions get 0 points
    
    if answered_questions == 0:
        return 25  # Minimum score for attempting the interview
    
    average_score = total_score / answered_questions
    # Adjust for completion rate
    completion_bonus = (answered_questions / total_questions) * 15
    final_score = min(100, average_score + completion_bonus)
    
    logging.info(f"Basic evaluation completed with score: {int(final_score)}")
    return int(final_score)

class ExamMonitoringGUI:
    """GUI application for the exam monitoring system"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Exam Monitoring System")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        self.data_manager = DataManager()
        self.auth_system = AuthenticationSystem(self.data_manager)
        self.current_user = None
        self.monitoring_system = None
        
        # Initialize AI interview components
        if init_interview_components():
            logging.info("AI Interview system initialized successfully")
        else:
            logging.warning("AI Interview system initialization failed - check your API key in .env file")
        
        self.create_login_screen()
    
    def create_login_screen(self):
        """Create login interface"""
        self.clear_screen()
        
        # Title
        title_label = tk.Label(self.root, text="Exam Monitoring System", 
                              font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=30)
        
        # Login frame
        login_frame = tk.Frame(self.root, bg='white', padx=40, pady=30)
        login_frame.pack(pady=20)
        
        tk.Label(login_frame, text="Login", font=("Arial", 18, "bold"), 
                bg='white', fg='#333').pack(pady=10)
        
        # Email
        tk.Label(login_frame, text="Email:", font=("Arial", 12), 
                bg='white').pack(anchor='w')
        self.login_email = tk.Entry(login_frame, font=("Arial", 12), width=30)
        self.login_email.pack(pady=5)
        
        # Password
        tk.Label(login_frame, text="Password:", font=("Arial", 12), 
                bg='white').pack(anchor='w')
        self.login_password = tk.Entry(login_frame, font=("Arial", 12), 
                                      width=30, show="*")
        self.login_password.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(login_frame, bg='white')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Capture Photo & Login", 
                 command=self.capture_and_login, bg='#4CAF50', fg='white',
                 font=("Arial", 12), padx=20).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="Register", 
                 command=self.create_register_screen, bg='#2196F3', fg='white',
                 font=("Arial", 12), padx=20).pack(side='left', padx=5)
    
    def create_register_screen(self):
        """Create registration interface"""
        self.clear_screen()
        
        # Title
        title_label = tk.Label(self.root, text="User Registration", 
                              font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=30)
        
        # Registration frame
        reg_frame = tk.Frame(self.root, bg='white', padx=40, pady=30)
        reg_frame.pack(pady=20)
        
        # Name
        tk.Label(reg_frame, text="Full Name:", font=("Arial", 12), 
                bg='white').pack(anchor='w')
        self.reg_name = tk.Entry(reg_frame, font=("Arial", 12), width=30)
        self.reg_name.pack(pady=5)
        
        # Email
        tk.Label(reg_frame, text="Email:", font=("Arial", 12), 
                bg='white').pack(anchor='w')
        self.reg_email = tk.Entry(reg_frame, font=("Arial", 12), width=30)
        self.reg_email.pack(pady=5)
        
        # Password
        tk.Label(reg_frame, text="Password:", font=("Arial", 12), 
                bg='white').pack(anchor='w')
        self.reg_password = tk.Entry(reg_frame, font=("Arial", 12), 
                                    width=30, show="*")
        self.reg_password.pack(pady=5)
        
        # Photo selection
        self.photo_path = tk.StringVar()
        tk.Label(reg_frame, text="Profile Photo:", font=("Arial", 12), 
                bg='white').pack(anchor='w')
        
        photo_frame = tk.Frame(reg_frame, bg='white')
        photo_frame.pack(pady=5, fill='x')
        
        tk.Entry(photo_frame, textvariable=self.photo_path, 
                font=("Arial", 10), width=25).pack(side='left', padx=(0, 5))
        tk.Button(photo_frame, text="Browse", 
                 command=self.browse_photo).pack(side='left')
        
        # Buttons
        button_frame = tk.Frame(reg_frame, bg='white')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Register", 
                 command=self.register_user, bg='#4CAF50', fg='white',
                 font=("Arial", 12), padx=20).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="Back to Login", 
                 command=self.create_login_screen, bg='#9E9E9E', fg='white',
                 font=("Arial", 12), padx=20).pack(side='left', padx=5)
    
    def browse_photo(self):
        """Browse for photo file"""
        filename = filedialog.askopenfilename(
            title="Select Profile Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filename:
            self.photo_path.set(filename)
    
    def register_user(self):
        """Handle user registration"""
        name = self.reg_name.get().strip()
        email = self.reg_email.get().strip()
        password = self.reg_password.get().strip()
        photo = self.photo_path.get().strip()
        
        if not all([name, email, password, photo]):
            messagebox.showerror("Error", "Please fill all fields and select a photo")
            return
        
        success, message = self.auth_system.register_user(name, email, password, photo)
        
        if success:
            messagebox.showinfo("Success", message)
            self.create_login_screen()
        else:
            messagebox.showerror("Error", message)
    
    def capture_and_login(self):
        """Capture photo and attempt login"""
        email = self.login_email.get().strip()
        password = self.login_password.get().strip()
        
        if not email or not password:
            messagebox.showerror("Error", "Please enter email and password")
            return
        
        # Capture photo from webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access camera")
            return
        
        messagebox.showinfo("Info", "Camera will open. Position your face and press SPACE to capture, ESC to cancel")
        
        captured_image = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Capture Login Photo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                captured_image = frame.copy()
                break
            elif key == 27:  # ESC to cancel
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_image is None:
            messagebox.showinfo("Info", "Photo capture cancelled")
            return
        
        # Attempt login
        success, result = self.auth_system.login_user(email, password, captured_image)
        
        if success:
            self.current_user = result
            messagebox.showinfo("Success", f"Welcome, {result['name']}!")
            self.create_dashboard()
        else:
            messagebox.showerror("Login Failed", result)

    def create_dashboard(self):
        """Create user dashboard"""
        self.clear_screen()
        
        # Title
        title_label = tk.Label(self.root, text=f"Welcome, {self.current_user['name']}", 
                              font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=30)
        
        # Dashboard frame
        dash_frame = tk.Frame(self.root, bg='white', padx=40, pady=30)
        dash_frame.pack(pady=20)
        
        tk.Label(dash_frame, text="Exam Dashboard", font=("Arial", 18, "bold"), 
                bg='white', fg='#333').pack(pady=20)
        
        # Buttons
        button_frame = tk.Frame(dash_frame, bg='white')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Start Exam Monitoring", 
                 command=self.start_monitoring, bg='#4CAF50', fg='white',
                 font=("Arial", 14), padx=30, pady=10).pack(pady=10)
        
        tk.Button(button_frame, text="AI Interview System", 
                 command=self.create_interview_screen, bg='#673AB7', fg='white',
                 font=("Arial", 14), padx=30, pady=10).pack(pady=10)
        
        tk.Button(button_frame, text="View Previous Sessions", 
                 command=self.view_sessions, bg='#2196F3', fg='white',
                 font=("Arial", 14), padx=30, pady=10).pack(pady=10)
        
        tk.Button(button_frame, text="View Cheating Events", 
                 command=self.view_cheating_events, bg='#FF5722', fg='white',
                 font=("Arial", 14), padx=30, pady=10).pack(pady=10)
        
        tk.Button(button_frame, text="System Settings", 
                 command=self.show_settings, bg='#FF9800', fg='white',
                 font=("Arial", 14), padx=30, pady=10).pack(pady=10)
        
        tk.Button(button_frame, text="Start Web Exam", 
                 command=self.start_web_exam, bg='#9C27B0', fg='white',
                 font=("Arial", 14), padx=30, pady=10).pack(pady=10)
        
        tk.Button(button_frame, text="Logout", 
                 command=self.logout, bg='#F44336', fg='white',
                 font=("Arial", 14), padx=30, pady=10).pack(pady=10)
        
        # Display current user information
        user_info_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=10)
        user_info_frame.pack(fill='x', side='bottom')
        
        # Current date and user info
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_info = f"Logged in as: {self.current_user['email']} | Last login: {current_date}"
        tk.Label(user_info_frame, text=user_info, font=("Arial", 10),bg='#f0f0f0', fg='#666').pack(side='right')
    def create_interview_screen(self):
        """Create interview interface with real-time speech capabilities"""
        self.clear_screen()
        
        # Title
        title_label = tk.Label(self.root, text="AI Interview System",
                               font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=30)
        
        # Interview frame
        interview_frame = tk.Frame(self.root, bg='white', padx=40, pady=30)
        interview_frame.pack(pady=20)
        
        # Upload resume
        tk.Label(interview_frame, text="Upload Resume:", font=("Arial", 12),
                bg='white').pack(anchor='w')
        
        self.resume_path = tk.StringVar()
        resume_frame = tk.Frame(interview_frame, bg='white')
        resume_frame.pack(pady=5, fill='x')
        tk.Entry(resume_frame, textvariable=self.resume_path,
                font=("Arial", 10), width=40).pack(side='left', padx=(0, 5))
        tk.Button(resume_frame, text="Browse",
                 command=self.browse_resume).pack(side='left')
        
        # Upload job description
        tk.Label(interview_frame, text="Upload Job Description:", font=("Arial", 12),
                bg='white').pack(anchor='w', pady=(10, 0))
        
        self.jd_path = tk.StringVar()
        jd_frame = tk.Frame(interview_frame, bg='white')
        jd_frame.pack(pady=5, fill='x')
        tk.Entry(jd_frame, textvariable=self.jd_path,
                font=("Arial", 10), width=40).pack(side='left', padx=(0, 5))
        tk.Button(jd_frame, text="Browse",
                 command=self.browse_jd).pack(side='left')
        
        # Buttons
        button_frame = tk.Frame(interview_frame, bg='white')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Start Real-Time Interview",
                 command=self.start_realtime_interview,
                 bg='#4CAF50', fg='white',
                 font=("Arial", 12), padx=20).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="Back to Dashboard",
                 command=self.create_dashboard,
                 bg='#9E9E9E', fg='white',
                 font=("Arial", 12), padx=20).pack(side='left', padx=5)
    
    def browse_resume(self):
        """Browse for resume file"""
        filename = filedialog.askopenfilename(
            title="Select Resume",
            filetypes=[("All supported", "*.pdf *.docx *.txt"), 
                       ("PDF files", "*.pdf"),
                       ("Word files", "*.docx"),
                       ("Text files", "*.txt")]
        )
        if filename:
            self.resume_path.set(filename)
            
    def browse_jd(self):
        """Browse for job description file"""
        filename = filedialog.askopenfilename(
            title="Select Job Description",
            filetypes=[("All supported", "*.pdf *.docx *.txt"), 
                       ("PDF files", "*.pdf"),
                       ("Word files", "*.docx"),
                       ("Text files", "*.txt")]
        )
        if filename:
            self.jd_path.set(filename)
            
    def start_realtime_interview(self):
        """Start real-time interview with speech interaction"""
        resume_file = self.resume_path.get().strip()
        jd_file = self.jd_path.get().strip()
        
        if not resume_file or not jd_file:
            messagebox.showerror("Error", "Please select both resume and job description files")
            return
            
        try:
            # Read resume and job description
            resume_text = extract_text_from_file(resume_file)
            jd_text = extract_text_from_file(jd_file)
            
            if not resume_text or not jd_text:
                messagebox.showerror("Error", "Could not read the uploaded files")
                return
                
            # Initialize interview system
            interview_system = InterviewSystem(
                AudioInterface(),
                self.current_user,
                gemini_client
            )
            
            # Show interview instructions
            instructions = """
            REAL-TIME INTERVIEW INSTRUCTIONS
            
            1. The system will ask questions verbally
            2. Speak clearly into your microphone to answer
            3. Wait for the system to process each answer
            4. Press 'q' to quit, 'r' to repeat question
            5. The interview will be recorded and analyzed
            
            Ready to start?
            """
            
            result = messagebox.askokcancel("Interview Instructions", instructions)
            if not result:
                return
                
            # Hide main window during interview
            self.root.withdraw()
            
            # Start the interview
            interview_system.start_interview(resume_text, jd_text)
            
            # Show main window again
            self.root.deiconify()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start interview: {str(e)}")
            self.root.deiconify()
            
    def start_monitoring(self):
        """Start exam monitoring session"""
        if self.current_user is None:
            messagebox.showerror("Error", "No user logged in")
            return
        
        # Show exam rules before starting
        rules = """
EXAM RULES AND REGULATIONS:

1. You are allowed maximum 5 tab switches
2. Keep your face visible at all times
3. No multiple persons allowed in the frame
4. No external objects (phones, books) allowed
5. Look at the screen - excessive gaze tracking will be flagged
6. Maintain quiet environment
7. Any violation will be recorded and may result in exam termination

Press 'q' to quit exam
Press 's' to view statistics
Press 't' to test tab switch (for demo)

Click OK to start the exam monitoring.
        """
        
        result = messagebox.askokcancel("Exam Rules", rules)
        if not result:
            return
        
        # Close GUI window and start monitoring
        self.root.withdraw()  # Hide the main window
        
        try:
            self.monitoring_system = ExamMonitoringSystem(self.current_user, self.data_manager)
            self.monitoring_system.start_monitoring()
        except Exception as e:
            logging.error(f"Error starting monitoring: {e}")
            messagebox.showerror("Error", f"Failed to start monitoring: {str(e)}")
        finally:
            self.root.deiconify()  # Show the main window again

    def view_sessions(self):
        """View previous monitoring sessions"""
        sessions = self.data_manager.get_user_sessions(self.current_user['id'])
        
        if not sessions:
            messagebox.showinfo("Info", "No previous sessions found")
            return
        
        # Create sessions window
        sessions_window = tk.Toplevel(self.root)
        sessions_window.title("Previous Sessions")
        sessions_window.geometry("900x600")
        sessions_window.configure(bg='#f0f0f0')
        
        # Title
        tk.Label(sessions_window, text="Previous Exam Sessions", 
                font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#333').pack(pady=20)
        
        # Create treeview for sessions
        frame = tk.Frame(sessions_window, bg='white')
        frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')
        
        # Treeview
        tree = ttk.Treeview(frame, yscrollcommand=scrollbar.set, 
                           columns=('Start', 'End', 'Duration', 'Alerts', 'Tab Switches', 'Terminated'), 
                           show='headings')
        scrollbar.config(command=tree.yview)
        
        # Configure columns
        tree.heading('#1', text='Session Start')
        tree.heading('#2', text='Session End')
        tree.heading('#3', text='Duration')
        tree.heading('#4', text='Total Alerts')
        tree.heading('#5', text='Tab Switches')
        tree.heading('#6', text='Terminated')
        
        tree.column('#1', width=150)
        tree.column('#2', width=150)
        tree.column('#3', width=100)
        tree.column('#4', width=100)
        tree.column('#5', width=100)
        tree.column('#6', width=100)
        
        # Populate treeview
        for session in reversed(sessions):  # Most recent first
            start_time = datetime.fromisoformat(session['session_start'])
            end_time = datetime.fromisoformat(session.get('session_end', session['session_start']))
            duration = end_time - start_time
            
            total_alerts = sum(session['alerts'].values())
            tab_switches = session['alerts'].get('tab_switches', 0)
            terminated = "Yes" if session.get('terminated_by_violations', False) else "No"
            
            tree.insert('', 'end', values=(
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time.strftime("%Y-%m-%d %H:%M:%S"),
                str(duration).split('.')[0],  # Remove microseconds
                total_alerts,
                tab_switches,
                terminated
            ))
        
        tree.pack(fill='both', expand=True)
        
        # Details button
        def show_session_details():
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                session_start = item['values'][0]
                # Find the session with matching start time
                for session in sessions:
                    if datetime.fromisoformat(session['session_start']).strftime("%Y-%m-%d %H:%M:%S") == session_start:
                        self.show_session_details(session)
                        break
        
        tk.Button(sessions_window, text="View Details", command=show_session_details,
                 bg='#2196F3', fg='white', font=("Arial", 12), padx=20).pack(pady=10)
        
        # Close button
        tk.Button(sessions_window, text="Close", command=sessions_window.destroy,
                 bg='#9E9E9E', fg='white', font=("Arial", 12), padx=20).pack(pady=5)

    def view_cheating_events(self):
        """View cheating events for current user"""
        events = self.data_manager.load_cheating_events()
        user_events = [e for e in events if e.get('student_id') == self.current_user['id']]
        
        if not user_events:
            messagebox.showinfo("Info", "No cheating events found")
            return
        
        # Create events window
        events_window = tk.Toplevel(self.root)
        events_window.title("Cheating Events")
        events_window.geometry("700x500")
        events_window.configure(bg='#f0f0f0')
        
        # Title
        tk.Label(events_window, text="Cheating Events Log", 
                font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#333').pack(pady=20)
        
        # Create treeview for events
        frame = tk.Frame(events_window, bg='white')
        frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')
        
        # Treeview
        tree = ttk.Treeview(frame, yscrollcommand=scrollbar.set, 
                           columns=('Event Type', 'Count', 'Flag', 'Last Updated'), 
                           show='headings')
        scrollbar.config(command=tree.yview)
        
        # Configure columns
        tree.heading('#1', text='Event Type')
        tree.heading('#2', text='Count')
        tree.heading('#3', text='Cheating Flag')
        tree.heading('#4', text='Last Updated')
        
        tree.column('#1', width=150)
        tree.column('#2', width=100)
        tree.column('#3', width=100)
        tree.column('#4', width=200)
        
        # Populate treeview
        for event in user_events:
            updated_time = datetime.fromisoformat(event['updated_at'])
            
            tree.insert('', 'end', values=(
                event['event_type'].replace('_', ' ').title(),
                event.get('tab_switch_count', 0),
                "Yes" if event.get('cheating_flag', False) else "No",
                updated_time.strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        tree.pack(fill='both', expand=True)
        
        # Close button
        tk.Button(events_window, text="Close", command=events_window.destroy,
                 bg='#9E9E9E', fg='white', font=("Arial", 12), padx=20).pack(pady=10)

    def show_session_details(self, session):
        """Show detailed session information"""
        details_window = tk.Toplevel(self.root)
        details_window.title("Session Details")
        details_window.geometry("700x500")
        details_window.configure(bg='#f0f0f0')
        
        # Create scrollable text widget
        frame = tk.Frame(details_window, bg='white')
        frame.pack(padx=20, pady=20, fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')
        
        text_widget = tk.Text(frame, yscrollcommand=scrollbar.set, font=("Courier", 10))
        scrollbar.config(command=text_widget.yview)
        
        # Format session details
        details = f"""SESSION DETAILS
{'='*50}

User: {session['user_name']}
Session Start: {datetime.fromisoformat(session['session_start']).strftime("%Y-%m-%d %H:%M:%S")}
Session End: {datetime.fromisoformat(session.get('session_end', session['session_start'])).strftime("%Y-%m-%d %H:%M:%S")}
Terminated by Violations: {'Yes' if session.get('terminated_by_violations', False) else 'No'}

ALERT SUMMARY
{'='*50}
"""
        
        for alert_type, count in session['alerts'].items():
            details += f"{alert_type.replace('_', ' ').title()}: {count}\n"
        
        details += f"\nTOTAL EVENTS: {len(session['events'])}\n\n"
        
        if session['events']:
            details += "EVENT LOG\n"
            details += "="*50 + "\n"
            for event in session['events']:
                timestamp = datetime.fromisoformat(event['timestamp']).strftime("%H:%M:%S")
                details += f"{timestamp} - {event['type'].upper()} ({event['severity']}): {event['description']}\n"
        
        text_widget.insert('1.0', details)
        text_widget.config(state='disabled')  # Make read-only
        text_widget.pack(fill='both', expand=True)
        
        # Close button
        tk.Button(details_window, text="Close", command=details_window.destroy,
                 bg='#9E9E9E', fg='white', font=("Arial", 12), padx=20).pack(pady=10)

    def show_settings(self):
        """Show system settings"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("System Settings")
        settings_window.geometry("500x400")
        settings_window.configure(bg='#f0f0f0')
        
        # Title
        tk.Label(settings_window, text="System Settings", 
                font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#333').pack(pady=20)
        
        # Settings frame
        settings_frame = tk.Frame(settings_window, bg='white', padx=30, pady=20)
        settings_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # System information
        tk.Label(settings_frame, text="System Information", 
                font=("Arial", 14, "bold"), bg='white', fg='#333').pack(pady=(0, 10))
        
        info_text = f"""
User: {self.current_user['name']}
Email: {self.current_user['email']}
Registered: {datetime.fromisoformat(self.current_user['registered_at']).strftime("%Y-%m-%d %H:%M:%S")}

System Status:
- Camera: {'Available' if cv2.VideoCapture(0).isOpened() else 'Not Available'}
- Audio: {'Available' if self._check_audio() else 'Not Available'}
- Face Recognition: Available
- Object Detection: Available

Tab Switch Limit: 5 switches maximum
        """
        
        tk.Label(settings_frame, text=info_text, font=("Arial", 10), 
                bg='white', fg='#666', justify='left').pack(pady=10)
        
        # Action buttons
        button_frame = tk.Frame(settings_frame, bg='white')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Test Camera", command=self.test_camera,
                 bg='#2196F3', fg='white', font=("Arial", 12), padx=20).pack(pady=5)
        
        tk.Button(button_frame, text="Export Data", command=self.export_data,
                 bg='#FF9800', fg='white', font=("Arial", 12), padx=20).pack(pady=5)
        
        tk.Button(button_frame, text="Close", command=settings_window.destroy,
                 bg='#9E9E9E', fg='white', font=("Arial", 12), padx=20).pack(pady=5)

    def _check_audio(self):
        """Check if audio is available"""
        try:
            audio = pyaudio.PyAudio()
            audio.terminate()
            return True
        except:
            return False

    def test_camera(self):
        """Test camera functionality"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return
        
        messagebox.showinfo("Info", "Camera will open for testing. Press ESC to close")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.putText(frame, "Camera Test - Press ESC to close", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def export_data(self):
        """Export user session data"""
        sessions = self.data_manager.get_user_sessions(self.current_user['id'])
        events = self.data_manager.load_cheating_events()
        user_events = [e for e in events if e.get('student_id') == self.current_user['id']]
        
        if not sessions and not user_events:
            messagebox.showinfo("Info", "No data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Session Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                export_data = {
                    'user': {
                        'name': self.current_user['name'],
                        'email': self.current_user['email'],
                        'registered_at': self.current_user['registered_at']
                    },
                    'sessions': sessions,
                    'cheating_events': user_events,
                    'export_date': datetime.now().isoformat()
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def start_web_exam(self):
        """Start web-based exam with tab switch monitoring"""
        # Instead of just showing the "not implemented" message, let's provide a real implementation
        if not self.current_user:
            messagebox.showerror("Error", "You must be logged in to start an exam")
            return
            
        # Show exam rules
        rules_text = """
WEB EXAM RULES:

1. You will be redirected to a web-based exam interface
2. Tab switching is monitored and limited to 5 switches
3. Your exam will be terminated automatically if you exceed the limit
4. The exam interface will open in your default browser
5. You'll need to login with your same credentials 

Would you like to continue?
"""
        proceed = messagebox.askokcancel("Web Exam", rules_text)
        if not proceed:
            return
            
        # Start the Flask server in a separate thread if it's not already running
        try:
            import webbrowser
            import threading
            
            # Store user ID in session for the web interface
            def start_server():
                app.run(debug=False, port=5000)
                
            # Check if server is already running
            server_thread = threading.Thread(target=start_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Wait a moment for server to start
            time.sleep(1)
            
            # Open the exam page in browser
            webbrowser.open('http://localhost:5000/exam_page')
            
            messagebox.showinfo("Web Exam", "Web exam interface opened in your browser.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start web exam: {str(e)}")

    def logout(self):
        """Logout current user"""
        self.current_user = None
        self.monitoring_system = None
        stop_event.clear()
        self.create_login_screen()

    def clear_screen(self):
        """Clear all widgets from the screen"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

# Flask Web Application for Tab Switch API
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global instances
data_manager = DataManager()
tab_tracker = TabSwitchTracker(data_manager)

@app.route('/record_tab_switch', methods=['POST'])
def record_tab_switch_api():
    """API endpoint to record tab switch"""
    try:
        # Get student ID from session or request
        student_id = session.get('user_id') or request.json.get('student_id')
        
        if not student_id:
            return jsonify({"error": "No student ID provided"}), 400
        # Record tab switch
        result = tab_tracker.record_tab_switch(student_id)
        
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Error in tab switch API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login_api():
    """API endpoint for user login"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"success": False, "error": "Missing credentials"}), 400
        
        auth_system = AuthenticationSystem(data_manager)
        user = data_manager.get_user_by_email(email)
        
        if user and user['password'] == auth_system.hash_password(password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            return jsonify({"success": True, "user": user})
        else:
            return jsonify({"success": False, "error": "Invalid credentials"}), 401
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/exam_page')
def exam_page():
    """Render exam page with tab switch monitoring"""
    if 'user_id' not in session:
        return "Please login first", 401
    
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Online Exam - Tab Switch Monitoring</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .alert-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            color: #856404;
        }
        .alert-danger {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .stats {
            background-color: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Online Exam Portal</h1>
        <p>Welcome, {{ session.user_name }}!</p>
        
        <div class="alert alert-warning">
            <strong>Warning:</strong> Tab switching is being monitored. You are allowed maximum 5 tab switches. 
            Exceeding this limit will terminate your exam.
        </div>
        
        <div class="stats">
            <h3>Current Statistics:</h3>
            <p>Tab Switches: <span id="tabCount">0</span> / 5</p>
            <p>Status: <span id="examStatus">Active</span></p>
        </div>
        
        <div id="alerts"></div>
        
        <h2>Exam Content</h2>
        <p>This is your exam content. Please stay on this page during the exam.</p>
        <p>Try switching tabs to test the monitoring system.</p>
        
        <button onclick="simulateTabSwitch()">Simulate Tab Switch (for testing)</button>
        <button onclick="checkStatus()">Check Status</button>
    </div>

    <script>
        let tabSwitchCount = 0;
        let examTerminated = false;

        // Monitor actual tab switches
        document.addEventListener('visibilitychange', function() {
            if (document.hidden && !examTerminated) {
                recordTabSwitch();
            }
        });

        // Monitor window focus/blur
        window.addEventListener('blur', function() {
            if (!examTerminated) {
                recordTabSwitch();
            }
        });

        function recordTabSwitch() {
            fetch('/record_tab_switch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                updateUI(data);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }

        function simulateTabSwitch() {
            if (!examTerminated) {
                recordTabSwitch();
            }
        }

        function updateUI(data) {
            document.getElementById('tabCount').textContent = data.count || 0;
            
            const alertsDiv = document.getElementById('alerts');
            const newAlert = document.createElement('div');
            
            if (data.status === 'terminated') {
                newAlert.className = 'alert alert-danger';
                newAlert.innerHTML = '<strong>EXAM TERMINATED:</strong> ' + data.message;
                document.getElementById('examStatus').textContent = 'TERMINATED';
                examTerminated = true;
                
                // Disable all interactions
                document.body.style.pointerEvents = 'none';
                document.body.style.opacity = '0.5';
            } else {
                newAlert.className = 'alert alert-warning';
                newAlert.innerHTML = data.message;
            }
            
            alertsDiv.appendChild(newAlert);
            
            // Auto-remove alerts after 5 seconds
            setTimeout(() => {
                if (newAlert.parentNode) {
                    newAlert.parentNode.removeChild(newAlert);
                }
            }, 5000);
        }

        function checkStatus() {
            alert('Tab Switches: ' + document.getElementById('tabCount').textContent + '/5\\nStatus: ' + document.getElementById('examStatus').textContent);
        }

        // Prevent right-click and certain key combinations
        document.addEventListener('contextmenu', e => e.preventDefault());
        document.addEventListener('keydown', function(e) {
            // Prevent F12, Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+U, Ctrl+Shift+C
            if (e.keyCode === 123 || // F12
                (e.ctrlKey && e.shiftKey && e.keyCode === 73) || // Ctrl+Shift+I
                (e.ctrlKey && e.shiftKey && e.keyCode === 74) || // Ctrl+Shift+J
                (e.ctrlKey && e.keyCode === 85) || // Ctrl+U
                (e.ctrlKey && e.shiftKey && e.keyCode === 67)) { // Ctrl+Shift+C
                e.preventDefault();
                recordTabSwitch();
                alert('Developer tools are not allowed during the exam!');
                return false;
            }
            
            // Prevent Alt+Tab (though this has limited effectiveness)
            if (e.altKey && e.keyCode === 9) {
                e.preventDefault();
                recordTabSwitch();
                return false;
            }
            
            // Prevent Ctrl+Tab
            if (e.ctrlKey && e.keyCode === 9) {
                e.preventDefault();
                recordTabSwitch();
                return false;
            }
        });

        // Prevent page refresh
        window.addEventListener('beforeunload', function(e) {
            if (!examTerminated) {
                const message = 'Are you sure you want to leave the exam? This will be recorded as a violation.';
                e.returnValue = message;
                recordTabSwitch();
                return message;
            }
        });

        // Monitor for full screen exit
        document.addEventListener('fullscreenchange', function() {
            if (!document.fullscreenElement && !examTerminated) {
                recordTabSwitch();
                alert('Please stay in fullscreen mode during the exam!');
            }
        });

        // Request fullscreen on page load
        window.addEventListener('load', function() {
            requestFullscreen();
        });

        function requestFullscreen() {
            const elem = document.documentElement;
            if (elem.requestFullscreen) {
                elem.requestFullscreen();
            } else if (elem.mozRequestFullScreen) {
                elem.mozRequestFullScreen();
            } else if (elem.webkitRequestFullscreen) {
                elem.webkitRequestFullscreen();
            } else if (elem.msRequestFullscreen) {
                elem.msRequestFullscreen();
            }
        }

        // Disable printing
        window.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.keyCode === 80) { // Ctrl+P
                e.preventDefault();
                alert('Printing is not allowed during the exam!');
                recordTabSwitch();
                return false;
            }
        });

        // Disable text selection
        document.addEventListener('selectstart', function(e) {
            e.preventDefault();
        });

        // Show warning when exam starts
        window.addEventListener('load', function() {
            setTimeout(function() {
                if (!examTerminated) {
                    alert('Exam monitoring is now active. Any suspicious activity will be recorded.');
                }
            }, 1000);
        });

        // Heartbeat to server (optional - to maintain session)
        setInterval(function() {
            if (!examTerminated) {
                fetch('/heartbeat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({timestamp: new Date().toISOString()})
                }).catch(function(error) {
                    console.log('Heartbeat failed:', error);
                });
            }
        }, 30000); // Every 30 seconds

        console.log('Exam monitoring system initialized');
    </script>
</body>
</html>
    """)

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """Heartbeat endpoint to maintain session"""
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    return jsonify({"status": "alive", "timestamp": datetime.now().isoformat()})

@app.route('/exam_terminated')
def exam_terminated():
    """Page shown when exam is terminated"""
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Exam Terminated</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8d7da;
            text-align: center;
        }
        .container {
            max-width: 600px;
            margin: 50px auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border: 2px solid #dc3545;
        }
        .icon {
            font-size: 64px;
            color: #dc3545;
            margin-bottom: 20px;
        }
        h1 {
            color: #dc3545;
            margin-bottom: 20px;
        }
        .reason {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: left;
        }
        button {
            background-color: #dc3545;
            color: white;
            padding: 10px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">⚠️</div>
        <h1>Exam Terminated</h1>
        <p>Your exam has been terminated due to excessive violations.</p>
        
        <div class="reason">
            <h3>Termination Reason:</h3>
            <p>You exceeded the maximum allowed tab switches (5). This is considered a serious violation of exam rules.</p>
        </div>
        
        <p>Your answers have been automatically submitted. Please contact your instructor for further information.</p>
        
        <button onclick="window.close()">Close Window</button>
    </div>
</body>
</html>
    """)

# Main execution
if __name__ == "__main__":
    try:
        # Create and run the GUI application
        app_gui = ExamMonitoringGUI()
        app_gui.run()
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"Application failed to start: {e}")
        print("Please ensure all required libraries are installed:")
        print("pip install opencv-python mediapipe ultralytics face-recognition pyaudio numpy pillow flask sentence-transformers scikit-learn python-docx PyPDF2 google-generativeai python-dotenv speechrecognition torch")
