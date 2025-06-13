from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import os
import json
import base64
import hashlib
from datetime import datetime
import secrets
import logging
from werkzeug.utils import secure_filename

ADMIN_ID = "admin@exam.com"
ADMIN_PASSWORD = "AdminSecure123!"
ADMIN_JPG_PATH = "static/admin.jpg"

try:
    import mediapipe as mp
    from ultralytics import YOLO
except ImportError as e:
    logging.error(f"Required libraries not installed: {e}")
    print("Please install required packages: pip install mediapipe ultralytics opencv-python numpy face-recognition pillow flask flask-socketio")
    exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['CHEATING_IMG_FOLDER'] = 'static/cheating_images'
app.config['CHEATING_AUDIO_FOLDER'] = 'static/cheating_audios'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHEATING_IMG_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHEATING_AUDIO_FOLDER'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("exam_monitoring.log"),
        logging.StreamHandler()
    ]
)

# Data Management (JSON-based)
class DataManager:
    def __init__(self):
        self.users_file = "users_data.json"
        self.monitoring_file = "monitoring_data.json"
        self.cheating_file = "cheating_events.json"
        self.cheating_images_file = "cheating_images.json"
        self.cheating_audios_file = "cheating_audios.json"
        self.ensure_files_exist()

    def ensure_files_exist(self):
        for f, default in [
            (self.users_file, []),
            (self.monitoring_file, []),
            (self.cheating_file, []),
            (self.cheating_images_file, []),
            (self.cheating_audios_file, []),
        ]:
            if not os.path.exists(f):
                with open(f, 'w') as fo:
                    json.dump(default, fo)

    def load_users(self):
        try:
            with open(self.users_file, 'r') as f:  # <-- fixed here
                return json.load(f)
        except:
            return []

    def save_users(self, users):
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)

    def add_user(self, user_data):
        users = self.load_users()
        users.append(user_data)
        self.save_users(users)

    def get_user_by_email(self, email):
        for user in self.load_users():
            if user['email'] == email:
                return user
        return None

    def get_user_by_id(self, uid):
        for user in self.load_users():
            if str(user['id']) == str(uid):
                return user
        return None

    def save_monitoring_session(self, session_data):
        try:
            with open(self.monitoring_file, 'r') as f:
                data = json.load(f)
        except:
            data = []
        data.append(session_data)
        with open(self.monitoring_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_user_sessions(self, user_id):
        try:
            with open(self.monitoring_file, 'r') as f:
                data = json.load(f)
            return [session for session in data if str(session.get('user_id')) == str(user_id)]
        except Exception as e:
            logging.warning(f"Session loading error: {e}")
            return []

    # Cheating Event Management
    def save_cheating_event(self, event):
        try:
            with open(self.cheating_file, 'r') as f:
                data = json.load(f)
        except:
            data = []
        data.append(event)
        with open(self.cheating_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_cheating_events_by_user(self, user_id, all_users=False):
        try:
            with open(self.cheating_file, 'r') as f:
                data = json.load(f)
            if all_users:
                return data
            if user_id is None:
                return []
            return [ev for ev in data if str(ev.get('student_id')) == str(user_id)]
        except:
            return []

    # Cheating Image Management
    def save_cheating_image(self, image_data):
        try:
            with open(self.cheating_images_file, 'r') as f:
                data = json.load(f)
        except:
            data = []
        data.append(image_data)
        with open(self.cheating_images_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_cheating_images_by_event(self, event_id):
        try:
            with open(self.cheating_images_file, 'r') as f:
                data = json.load(f)
            return [img for img in data if img.get('event_id') == event_id]
        except:
            return []

    # Cheating Audio Management
    def save_cheating_audio(self, audio_data):
        try:
            with open(self.cheating_audios_file, 'r') as f:
                data = json.load(f)
        except:
            data = []
        data.append(audio_data)
        with open(self.cheating_audios_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_cheating_audios_by_event(self, event_id):
        try:
            with open(self.cheating_audios_file, 'r') as f:
                data = json.load(f)
            return [aud for aud in data if aud.get('event_id') == event_id]
        except:
            return []

# Authentication
class AuthenticationSystem:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def get_face_encoding(self, image):
        import face_recognition
        try:
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                return None
            return face_recognition.face_encodings(image, face_locations)[0]
        except:
            return None

    def register_user(self, name, email, password, photo_path):
        if self.data_manager.get_user_by_email(email):
            return False, "Email already registered"
        try:
            import cv2
            image = cv2.imread(photo_path)
            if image is None:
                return False, "Could not load photo"
            face_encoding = self.get_face_encoding(image)
            if face_encoding is None:
                return False, "No face detected in photo"
            with open(photo_path, "rb") as img_file:
                photo_base64 = base64.b64encode(img_file.read()).decode()
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

    def login_user(self, email, password, captured_image_data):
        import face_recognition
        user = self.data_manager.get_user_by_email(email)
        if not user:
            return False, "User not found"
        if user['password'] != self.hash_password(password):
            return False, "Invalid password"
        try:
            image_data = base64.b64decode(captured_image_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            captured_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            return False, "Invalid image data"
        captured_encoding = self.get_face_encoding(captured_image)
        if captured_encoding is None:
            return False, "No face detected in captured image"
        stored_encoding = np.array(user['face_encoding'])
        matches = face_recognition.compare_faces([stored_encoding], captured_encoding, tolerance=0.6)
        if not matches[0]:
            return False, "Face does not match registered photo"
        return True, user

# Cheating Event Logic
def save_cheating_image(event_id, img_base64):
    """Save cheating image to disk and reference it in cheating_images.json"""
    img_data = base64.b64decode(img_base64.split(',')[1])
    filename = f"cheating_{event_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
    path = os.path.join(app.config['CHEATING_IMG_FOLDER'], filename)
    with open(path, "wb") as f:
        f.write(img_data)
    # Save to cheating_images.json
    image_record = {
        "event_id": event_id,
        "filename": filename,
        "timestamp": datetime.now().isoformat()
    }
    data_manager.save_cheating_image(image_record)
    return filename

def save_cheating_audio(event_id, audio_file):
    """Save cheating audio file to disk and reference it in cheating_audios.json"""
    filename = f"cheating_{event_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.wav"
    path = os.path.join(app.config['CHEATING_AUDIO_FOLDER'], filename)
    audio_file.save(path)
    audio_record = {
        "event_id": event_id,
        "filename": filename,
        "timestamp": datetime.now().isoformat()
    }
    data_manager.save_cheating_audio(audio_record)
    return filename

def save_cheating_event(student_id, event_type, detected_objects, tab_switch_count, img_base64=None, audio_file=None):
    """Save a cheating event (optionally its image/audio)"""
    event_id = f"{student_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    event = {
        "id": event_id,
        "student_id": student_id,
        "cheating_flag": True,
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "detected_objects": detected_objects or [],
        "tab_switch_count": tab_switch_count or 0,
        "images": [],
        "audios": []
    }
    if img_base64:
        img_filename = save_cheating_image(event_id, img_base64)
        event["images"].append(img_filename)
    if audio_file:
        audio_filename = save_cheating_audio(event_id, audio_file)
        event["audios"].append(audio_filename)
    data_manager.save_cheating_event(event)
    return event

# Monitoring Logic
class ExamMonitoringSystem:
    def __init__(self, user_data, data_manager):
        self.user_data = user_data
        self.data_manager = data_manager
        self.init_computer_vision()
        self.init_object_detection()
        self.monitoring = False
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
                'audio_detected': 0
            },
            'events': []
        }

    def init_computer_vision(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    def init_object_detection(self):
        try:
            self.yolo_model = YOLO("yolo11s.pt")
        except Exception as e:
            self.yolo_model = None

    def log_event(self, event_type, description, severity="medium", detected_objects=None, img_base64=None, audio_file=None):
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'description': description,
            'severity': severity
        }
        self.session_data['events'].append(event)
        socketio.emit('alert', {
            'type': event_type,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        # Save as cheating event if relevant
        if event_type in ['multiple_faces', 'multiple_persons', 'cell_phone', 'book', 'gaze_away', 'audio_detected']:
            save_cheating_event(
                student_id=self.user_data['id'],
                event_type=event_type,
                detected_objects=detected_objects,
                tab_switch_count=0,
                img_base64=img_base64,
                audio_file=audio_file
            )

    def process_frame(self, frame_data):
        try:
            image_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            face_count = self.detect_faces(frame, frame_data)
            gaze_direction = self.track_gaze(frame)
            person_count, suspicious_objects = self.detect_objects(frame, frame_data)
            return {
                'face_count': face_count,
                'gaze_direction': gaze_direction,
                'person_count': person_count,
                'suspicious_objects': suspicious_objects,
                'alerts': self.session_data['alerts'].copy()
            }
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None

    def detect_faces(self, frame, frame_b64):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = self.face_detection.process(rgb_frame)
        face_count = 0
        if detection_results.detections:
            face_count = len(detection_results.detections)
        if face_count > 1:
            self.session_data['alerts']['multiple_faces'] += 1
            self.log_event('multiple_faces', f'{face_count} faces detected', 'high', detected_objects=[], img_base64=frame_b64)
        return face_count

    def track_gaze(self, frame):
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

    def detect_objects(self, frame, frame_b64):
        if self.yolo_model is None:
            return 0, []
        suspicious_objects = []
        person_count = 0
        height, width = frame.shape[:2]
        if width > 640:
            aspect_ratio = height / width
            frame = cv2.resize(frame, (640, int(640 * aspect_ratio)))
        results = self.yolo_model(frame)
        for result in results:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = box
                if score > 0.5:
                    label = self.yolo_model.names[int(class_id)]
                    if label.lower() == "person":
                        person_count += 1
                        if person_count > 1:
                            suspicious_objects.append("multiple_persons")
                            self.session_data['alerts']['multiple_persons'] += 1
                            self.log_event('multiple_persons', f'{person_count} persons detected', 'high', detected_objects=['person'], img_base64=frame_b64)
                    elif label.lower() == "cell phone":
                        suspicious_objects.append("cell_phone")
                        self.session_data['alerts']['cell_phone'] += 1
                        self.log_event('cell_phone', 'Cell phone detected', 'high', detected_objects=['cell phone'], img_base64=frame_b64)
                    elif label.lower() == "book":
                        suspicious_objects.append("book")
                        self.session_data['alerts']['book'] += 1
                        self.log_event('book', 'Book detected', 'medium', detected_objects=['book'], img_base64=frame_b64)
        return person_count, suspicious_objects

    def start_session(self):
        self.monitoring = True
        self.session_data['session_start'] = datetime.now().isoformat()

    def stop_session(self):
        self.monitoring = False
        self.session_data['session_end'] = datetime.now().isoformat()
        self.data_manager.save_monitoring_session(self.session_data)
        return self.session_data

# App globals
data_manager = DataManager()
auth_system = AuthenticationSystem(data_manager)
active_sessions = {}

# --- Flask Routes ---

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        if 'photo' not in request.files or request.files['photo'].filename == '':
            flash('No photo uploaded', 'error')
            return render_template('register.html')
        photo = request.files['photo']
        filename = secure_filename(photo.filename)
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(photo_path)
        success, message = auth_system.register_user(name, email, password, photo_path)
        if os.path.exists(photo_path):
            os.remove(photo_path)
        if success:
            flash(message, 'success')
            return redirect(url_for('index'))
        else:
            flash(message, 'error')
    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')
    captured_image = request.json.get('image')
    if not all([email, password, captured_image]):
        return jsonify({'success': False, 'message': 'Missing required fields'})
    success, result = auth_system.login_user(email, password, captured_image)
    if success:
        session['user_id'] = result['id']
        session['user_name'] = result['name']
        session['user_email'] = result['email']
        return jsonify({'success': True, 'message': f'Welcome, {result["name"]}!'})
    else:
        return jsonify({'success': False, 'message': result})

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    user_sessions = data_manager.get_user_sessions(session['user_id'])
    return render_template('dashboard.html', 
                         user_name=session['user_name'],
                         session_count=len(user_sessions))

@app.route('/monitoring')
def monitoring():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('monitoring.html', user_name=session['user_name'])

@app.route('/sessions')
def sessions():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    user_sessions = data_manager.get_user_sessions(session['user_id'])
    processed_sessions = []
    for sess in reversed(user_sessions):
        try:
            start_time = datetime.fromisoformat(sess['session_start'])
            end_time = datetime.fromisoformat(sess.get('session_end', sess['session_start']))
            duration = end_time - start_time
            total_alerts = sum(sess['alerts'].values())
            processed_sessions.append({
                'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
                'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
                'duration': str(duration).split('.')[0],
                'total_alerts': total_alerts,
                'alerts': sess['alerts'],
                'events': sess['events']
            })
        except Exception as e:
            logging.warning(f"Error processing session: {e}")
    return render_template('sessions.html', sessions=processed_sessions)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    user = data_manager.get_user_by_id(session['user_id'])
    if not user:
        flash('User profile not found.', 'error')
        return redirect(url_for('index'))

    # Ensure cheating_events is always a list
    cheating_events = data_manager.get_cheating_events_by_user(session['user_id'])
    if not cheating_events:
        cheating_events = []

    # Gather all images and audio for these events
    cheating_images = []
    cheating_audios = []
    for ev in cheating_events:
        # Images
        imgs = data_manager.get_cheating_images_by_event(ev['id'])
        for img in imgs:
            cheating_images.append({
                'filename': img['filename'],
                'event_type': ev.get('event_type', 'Unknown'),
                'timestamp': img.get('timestamp', 'Unknown')
            })
        # Audios
        auds = data_manager.get_cheating_audios_by_event(ev['id'])
        for aud in auds:
            cheating_audios.append({
                'filename': aud['filename'],
                'event_type': ev.get('event_type', 'Unknown'),
                'timestamp': aud.get('timestamp', 'Unknown')
            })

    return render_template(
        'profile.html',
        user=user,
        cheating_images=cheating_images,
        cheating_audios=cheating_audios,
        events=cheating_events
    )

@app.route('/cheating_images/<filename>')
def cheating_image(filename):
    return send_from_directory(app.config['CHEATING_IMG_FOLDER'], filename)

@app.route('/cheating_audios/<filename>')
def cheating_audio(filename):
    return send_from_directory(app.config['CHEATING_AUDIO_FOLDER'], filename)

@app.route('/logout')
def logout():
    if session.get('user_id') in active_sessions:
        monitoring_system = active_sessions[session['user_id']]
        monitoring_system.stop_session()
        del active_sessions[session['user_id']]
    session.clear()
    return redirect(url_for('index'))

# --- Socket.IO Events ---
@socketio.on('start_monitoring')
def handle_start_monitoring():
    if 'user_id' not in session:
        emit('error', {'message': 'Not authenticated'})
        return
    user_data = {
        'id': session['user_id'],
        'name': session['user_name'],
        'email': session['user_email']
    }
    monitoring_system = ExamMonitoringSystem(user_data, data_manager)
    monitoring_system.start_session()
    active_sessions[session['user_id']] = monitoring_system
    emit('monitoring_started', {'message': 'Monitoring session started'})

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    if session.get('user_id') in active_sessions:
        monitoring_system = active_sessions[session['user_id']]
        session_data = monitoring_system.stop_session()
        del active_sessions[session['user_id']]
        emit('monitoring_stopped', {'message': 'Monitoring session ended', 'session_data': session_data})

@socketio.on('process_frame')
def handle_process_frame(data):
    if session.get('user_id') not in active_sessions:
        emit('error', {'message': 'No active monitoring session'})
        return
    monitoring_system = active_sessions[session['user_id']]
    if not monitoring_system.monitoring:
        emit('error', {'message': 'Monitoring not active'})
        return
    frame_data = data.get('frame')
    if not frame_data:
        emit('error', {'message': 'No frame data'})
        return
    result = monitoring_system.process_frame(frame_data)
    if result:
        emit('frame_processed', result)

@socketio.on('audio_alert')
def handle_audio_alert(data=None):
    if session.get('user_id') in active_sessions:
        monitoring_system = active_sessions[session['user_id']]
        monitoring_system.session_data['alerts']['audio_detected'] += 1
        # If audio file sent, save it
        audio_file = None
        if data and 'audio' in data:
            audio_file = data['audio']  # Should be a werkzeug FileStorage
        monitoring_system.log_event('audio_detected', 'Suspicious audio detected', 'medium', audio_file=audio_file)

# --- ADMIN ROUTES ---

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin_id = request.form.get('admin_id')
        admin_password = request.form.get('admin_password')
        if admin_id == ADMIN_ID and admin_password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials', 'danger')
    return render_template('admin_login.html')

@app.route('/admin-dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    total_candidates = len(data_manager.load_users())
    total_cheaters = len(set(event['student_id'] for event in data_manager.get_cheating_events_by_user(None, all_users=True)))
    return render_template('admin_dashboard.html', total_candidates=total_candidates, total_cheaters=total_cheaters)

@app.route('/admin-logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('Logged out as admin', 'info')
    return redirect(url_for('admin_login'))

@app.route('/admin/candidates')
def admin_candidates():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    users = data_manager.load_users()
    return render_template('admin_candidates.html', users=users)

@app.route('/admin/cheaters')
def admin_cheaters():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    cheater_ids = set()
    for user in data_manager.load_users():
        events = data_manager.get_cheating_events_by_user(user['id'])
        if events:
            cheater_ids.add(str(user['id']))
    cheaters = [u for u in data_manager.load_users() if str(u['id']) in cheater_ids]
    return render_template('admin_cheaters.html', cheaters=cheaters)

@app.route('/admin/profile/<int:user_id>')
def admin_profile(user_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    user = data_manager.get_user_by_id(user_id)
    cheating_events = data_manager.get_cheating_events_by_user(user_id) or []
    cheating_images = []
    cheating_audios = []
    for ev in cheating_events:
        imgs = data_manager.get_cheating_images_by_event(ev['id'])
        for img in imgs:
            cheating_images.append({'filename': img['filename'], 'event_type': ev.get('event_type'), 'timestamp': img.get('timestamp')})
        auds = data_manager.get_cheating_audios_by_event(ev['id'])
        for aud in auds:
            cheating_audios.append({'filename': aud['filename'], 'event_type': ev.get('event_type'), 'timestamp': aud.get('timestamp')})
    return render_template('profile.html', user=user, cheating_images=cheating_images, cheating_audios=cheating_audios, events=cheating_events, is_admin=True)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=4000)
