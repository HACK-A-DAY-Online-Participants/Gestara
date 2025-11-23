"""
Gestara Backend - Fixed with Proper Hand Detection
Based on working isl_detection.py reference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import base64
import copy
import itertools
import string
from difflib import get_close_matches

app = Flask(__name__)
CORS(app)
word_map = {}
class ISLRecognizer:
    def __init__(self, model_path='model.h5'):
        print("="*50)
        print("Loading ISL Model with Proper Hand Detection")
        print("="*50)
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # MediaPipe setup (same as working code)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,  # 0 for faster processing
            max_num_hands=2,     # Support 2 hands like ISL
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Alphabet classes (1-9 + A-Z)
        self.alphabet = ['1','2','3','4','5','6','7','8','9']
        self.alphabet += list(string.ascii_uppercase)
        
        print(f"✓ Recognizes: {len(self.alphabet)} classes (1-9, A-Z)")
        
        # Dictionary for autocomplete
        self.load_dictionary()
        
        print(f"✓ Dictionary loaded: {len(self.dictionary)} words")
        print("="*50)
    word_map = {}

    def load_word_videos():
        """Load word-to-video mapping"""
        global word_map
        
        word_json_path = 'word_videos.json'
        
        if not Path(word_json_path).exists():
            print(f"⚠ {word_json_path} not found. Creating sample mapping...")
            create_sample_word_map()
            return
        
        try:
            with open(word_json_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            
            # Normalize keys
            word_map = {}
            for k, v in raw.items():
                nk = normalize_word(k)
                word_map[nk] = v
            
            print(f"✓ Loaded {len(word_map)} word-to-video mappings")
        except Exception as e:
            print(f"⚠ Error loading word_videos.json: {e}")

    def normalize_word(w):
        """Normalize word: lowercase, strip, remove punctuation"""
        w = w.lower().strip()
        return w.translate(str.maketrans('', '', string.punctuation))

    def create_sample_word_map():
        """Create sample word_videos.json"""
        sample = {
            "hello": ["videos/hello_1.mp4", "videos/hello_2.mp4"],
            "good": ["videos/good_1.mp4"],
            "morning": ["videos/morning_1.mp4"],
            "good morning": ["videos/good_morning_1.mp4"],
            "thank you": ["videos/thank_you_1.mp4"],
            "please": ["videos/please_1.mp4"],
            "sorry": ["videos/sorry_1.mp4"],
            "yes": ["videos/yes_1.mp4"],
            "no": ["videos/no_1.mp4"],
            "help": ["videos/help_1.mp4"],
            "water": ["videos/water_1.mp4"],
            "food": ["videos/food_1.mp4"]
        }
        
        with open('word_videos.json', 'w') as f:
            json.dump(sample, f, indent=2)
        
        print("✓ Created sample word_videos.json")

    def parse_text_to_words(text):
        """
        Parse text into ISL words using greedy longest-match
        Prioritizes phrases over individual words
        """
        phrase = normalize_word(text)
        
        # Sort vocabulary by length (longest first)
        vocab_sorted = sorted(word_map.keys(), key=len, reverse=True)
        
        result = []
        idx = 0
        
        # Greedy matching
        while idx < len(phrase):
            matched = False
            
            for vocab_word in vocab_sorted:
                if phrase[idx:].startswith(vocab_word):
                    result.append(vocab_word)
                    idx += len(vocab_word)
                    matched = True
                    break
            
            if not matched:
                # Skip character (space or unknown)
                idx += 1
        
        return result

    @app.route('/text-to-sign', methods=['POST'])
    def text_to_sign():
        """
        Convert text to ISL sign video URLs
        
        Request JSON:
        {
            "text": "hello good morning"
        }
        
        Response:
        {
            "success": true,
            "text": "hello good morning",
            "words": ["hello", "good morning"],
            "videos": [
                {"word": "hello", "video": "videos/hello_1.mp4"},
                {"word": "good morning", "video": "videos/good_morning_1.mp4"}
            ]
        }
        """
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text:
                return jsonify({
                    'success': False,
                    'error': 'No text provided'
                }), 400
            
            # Parse text into words
            words = parse_text_to_words(text)
            
            if not words:
                return jsonify({
                    'success': False,
                    'error': 'No recognized words in input',
                    'text': text
                })
            
            # Get videos for each word
            videos = []
            for word in words:
                video_paths = word_map.get(word, [])
                
                if video_paths:
                    # Pick random video if multiple available
                    video_path = random.choice(video_paths)
                    videos.append({
                        'word': word,
                        'video': video_path
                    })
                else:
                    # Word not found
                    videos.append({
                        'word': word,
                        'video': None,
                        'error': 'No video available'
                    })
            
            return jsonify({
                'success': True,
                'text': text,
                'words': words,
                'videos': videos,
                'count': len(videos)
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/available-words', methods=['GET'])
    def available_words():
        """Get list of all available ISL words"""
        return jsonify({
            'success': True,
            'words': sorted(word_map.keys()),
            'count': len(word_map)
        })

    def load_dictionary(self):
        """Load word dictionary"""
        self.dictionary = set([
            "hello", "hi", "hey", "good", "morning", "afternoon", "evening", "night",
            "goodbye", "bye", "thanks", "thank", "please", "sorry", "yes", "no",
            "okay", "water", "food", "help", "home", "work", "school", "friend",
            "family", "mother", "father", "brother", "sister", "child", "baby",
            "happy", "sad", "love", "like", "want", "need", "know", "think",
            "come", "go", "eat", "drink", "sleep", "play", "study", "read",
            "today", "tomorrow", "yesterday", "time", "day", "week", "month",
            "what", "when", "where", "who", "how", "why", "which"
        ])
    
    def calc_landmark_list(self, image, landmarks):
        """
        Calculate landmark list from MediaPipe results
        (Same as working reference code)
        """
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        
        return landmark_point
    
    def pre_process_landmark(self, landmark_list):
        """
        Preprocess landmarks: relative coordinates + normalization
        (Same as working reference code)
        """
        temp_landmark_list = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        
        # Convert to one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        
        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))
        
        def normalize_(n):
            return n / max_value
        
        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        
        return temp_landmark_list
    
    def predict(self, frame):
        """Predict letter from frame"""
        # Convert BGR to RGB (MediaPipe needs RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image)
        
        if not results.multi_hand_landmarks:
            return None, 0
        
        # Get first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Calculate landmarks
        landmark_list = self.calc_landmark_list(frame, hand_landmarks)
        
        # Preprocess (same as training)
        pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
        
        # Convert to DataFrame (model expects this format)
        df = pd.DataFrame(pre_processed_landmark_list).transpose()
        
        # Predict
        predictions = self.model.predict(df, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class])
        
        # Get letter
        letter = self.alphabet[predicted_class]
        
        return letter, confidence
    
    def autocomplete(self, partial, max_results=10):
        """Get word suggestions"""
        if not partial or len(partial) < 2:
            return []
        
        partial_lower = partial.lower()
        matches = [w for w in self.dictionary if w.startswith(partial_lower)]
        
        return sorted(matches, key=len)[:max_results]
    
    def autocorrect(self, word, max_results=5):
        """Auto-correct typos"""
        if not word:
            return []
        
        word_lower = word.lower()
        
        if word_lower in self.dictionary:
            return [word_lower]
        
        matches = get_close_matches(word_lower, self.dictionary, 
                                    n=max_results, cutoff=0.6)
        return matches

# Global recognizer
recognizer = None

@app.route('/')
def index():
    return jsonify({
        'status': 'online',
        'name': 'Gestara ISL API (Fixed)',
        'model': 'Working hand detection',
        'version': '2.0',
        'classes': '1-9, A-Z (35 total)'
    })

@app.route('/health')
def health():
    if recognizer is None:
        return jsonify({'status': 'error'}), 503
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'classes': len(recognizer.alphabet)
    })

@app.route('/predict-letter', methods=['POST'])
def predict_letter():
    """Predict letter from image"""
    if recognizer is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image'}), 400
        
        # Decode image
        img_str = data['image']
        if ',' in img_str:
            img_str = img_str.split(',')[1]
        
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image'}), 400
        
        # Flip image horizontally (mirror mode like webcam)
        frame = cv2.flip(frame, 1)
        
        # Predict
        letter, confidence = recognizer.predict(frame)
        
        if letter is None:
            return jsonify({
                'success': False,
                'error': 'No hand detected'
            })
        
        return jsonify({
            'success': True,
            'letter': letter,
            'confidence': confidence
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    """Autocomplete suggestions"""
    try:
        data = request.get_json()
        partial = data.get('partial', '').strip()
        max_suggestions = data.get('max_suggestions', 10)
        
        if not partial:
            return jsonify({'success': False, 'error': 'No input'}), 400
        
        suggestions = recognizer.autocomplete(partial, max_suggestions)
        
        return jsonify({
            'success': True,
            'partial': partial,
            'suggestions': suggestions,
            'count': len(suggestions)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/autocorrect', methods=['POST'])
def autocorrect():
    """Auto-correct spelling"""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        max_suggestions = data.get('max_suggestions', 5)
        
        if not word:
            return jsonify({'success': False, 'error': 'No word'}), 400
        
        corrections = recognizer.autocorrect(word, max_suggestions)
        
        return jsonify({
            'success': True,
            'word': word,
            'corrections': corrections,
            'count': len(corrections),
            'is_correct': word.lower() in recognizer.dictionary
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/suggest-words', methods=['POST'])
def suggest_words():
    """Smart suggestions"""
    try:
        data = request.get_json()
        current_input = data.get('current_input', '').strip()
        max_suggestions = data.get('max_suggestions', 10)
        
        if not current_input:
            return jsonify({'success': False, 'error': 'No input'}), 400
        
        # Try autocomplete first
        suggestions = recognizer.autocomplete(current_input, max_suggestions)
        
        if not suggestions:
            # Try autocorrect
            suggestions = recognizer.autocorrect(current_input, max_suggestions)
            suggestion_type = 'autocorrect'
        else:
            suggestion_type = 'autocomplete'
        
        return jsonify({
            'success': True,
            'input': current_input,
            'suggestions': suggestions,
            'type': suggestion_type,
            'count': len(suggestions)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("GESTARA ISL BACKEND (FIXED)")
    print("="*50)
    
    try:
        recognizer = ISLRecognizer('model.h5')
        print("\n✓ Backend ready!")
        print("\nStarting server on http://localhost:5000")
        print("="*50 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    
    except FileNotFoundError:
        print("\n❌ Error: model.h5 not found!")
        print("Download from: https://github.com/MaitreeVaria/Indian-Sign-Language-Detection")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
