#!/usr/bin/env python3
"""
ARC AGI Captcha System
A web-based captcha using Abstract Reasoning Corpus visual puzzles
"""

import json
import random
import os
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from datetime import datetime, timedelta
import uuid

app = Flask(__name__)
app.secret_key = 'arc_captcha_secret_key_change_in_production'
CORS(app)

# Configuration
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')

class ARCCaptcha:
    def __init__(self):
        self.challenges = {}
        self.challenge_files = []
        self.load_arc_data()
        
    def load_arc_data(self):
        """Load ARC challenges directly from dataset files"""
        self.challenge_files = []
        
        # Load from ARC-1 and ARC-2 training and evaluation sets
        arc_paths = [
            os.path.join(DATASET_PATH, 'ARC-1', 'grouped-tasks', 'training'),
            os.path.join(DATASET_PATH, 'ARC-1', 'grouped-tasks', 'evaluation'),
            os.path.join(DATASET_PATH, 'ARC-2', 'grouped-tasks', 'training'),
            os.path.join(DATASET_PATH, 'ARC-2', 'grouped-tasks', 'evaluation'),
        ]
        
        for arc_path in arc_paths:
            if os.path.exists(arc_path):
                for filename in os.listdir(arc_path):
                    if filename.endswith('.json'):
                        file_path = os.path.join(arc_path, filename)
                        self.challenge_files.append(file_path)
        
        print(f"Found {len(self.challenge_files)} ARC challenge files")
        
        # Pre-load a sample for validation
        if self.challenge_files:
            sample_file = random.choice(self.challenge_files)
            try:
                with open(sample_file, 'r') as f:
                    sample_data = json.load(f)
                    print(f"Sample challenge loaded successfully from {os.path.basename(sample_file)}")
            except Exception as e:
                print(f"Error loading sample challenge: {e}")
    
    def get_random_challenge(self, difficulty='medium'):
        """Get a random ARC challenge based on difficulty"""
        if not self.challenge_files:
            return None
        
        # Try multiple files to find one that matches difficulty
        attempts = 0
        max_attempts = min(50, len(self.challenge_files))
        
        while attempts < max_attempts:
            # Pick a random challenge file
            challenge_file = random.choice(self.challenge_files)
            
            try:
                with open(challenge_file, 'r') as f:
                    task_data = json.load(f)
                
                # Validate the challenge has training data
                if 'train' not in task_data or len(task_data['train']) == 0:
                    attempts += 1
                    continue
                
                # Calculate difficulty based on grid size and pattern complexity
                first_example = task_data['train'][0]
                input_grid = first_example['input']
                output_grid = first_example['output']
                
                grid_size = len(input_grid) * len(input_grid[0])
                
                # Count unique colors (simpler = fewer colors)
                input_colors = set()
                output_colors = set()
                for row in input_grid:
                    input_colors.update(row)
                for row in output_grid:
                    output_colors.update(row)
                
                color_complexity = len(input_colors) + len(output_colors)
                
                # Enhanced difficulty classification
                total_complexity = grid_size + (color_complexity * 10)
                
                # Check if this challenge matches the requested difficulty
                difficulty_match = False
                if difficulty == 'easy' and total_complexity <= 120:
                    difficulty_match = True
                elif difficulty == 'medium' and 80 <= total_complexity <= 250:
                    difficulty_match = True
                elif difficulty == 'hard' and total_complexity > 200:
                    difficulty_match = True
                
                if difficulty_match or attempts > max_attempts // 2:
                    # Use this challenge
                    task_id = os.path.splitext(os.path.basename(challenge_file))[0]
                    
                    return {
                        'id': task_id,
                        'data': task_data,
                        'file': challenge_file,
                        'difficulty_score': total_complexity
                    }
                
            except Exception as e:
                print(f"Error loading challenge from {challenge_file}: {e}")
            
            attempts += 1
        
        # Fallback: just return any valid challenge
        for challenge_file in random.sample(self.challenge_files, min(10, len(self.challenge_files))):
            try:
                with open(challenge_file, 'r') as f:
                    task_data = json.load(f)
                
                if 'train' in task_data and len(task_data['train']) > 0:
                    task_id = os.path.splitext(os.path.basename(challenge_file))[0]
                    return {
                        'id': task_id,
                        'data': task_data,
                        'file': challenge_file,
                        'difficulty_score': 100
                    }
            except Exception as e:
                continue
        
        return None
    
    def validate_solution(self, challenge_file, user_solution):
        """Validate user's solution against the correct answer"""
        if not challenge_file:
            return False
        
        try:
            with open(challenge_file, 'r') as f:
                task_data = json.load(f)
            
            # Get the correct solution from the test cases
            if 'test' not in task_data or len(task_data['test']) == 0:
                return False
            
            # Check against all test case outputs
            for test_case in task_data['test']:
                if 'output' in test_case:
                    if self.grids_match(user_solution, test_case['output']):
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error validating solution: {e}")
            return False
    
    def grids_match(self, grid1, grid2):
        """Check if two grids are identical"""
        if len(grid1) != len(grid2):
            return False
        
        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        
        return True

# Initialize captcha system
captcha_system = ARCCaptcha()

@app.route('/')
def index():
    """Main captcha page"""
    return render_template('index.html')

@app.route('/api/challenge')
def get_challenge():
    """Get a new ARC challenge for the user"""
    difficulty = request.args.get('difficulty', 'medium')
    
    challenge = captcha_system.get_random_challenge(difficulty)
    if not challenge:
        return jsonify({'error': 'No challenges available'}), 500
    
    # Store challenge info in session for validation
    session['current_challenge'] = challenge['id']
    session['current_challenge_file'] = challenge['file']
    session['challenge_start_time'] = datetime.now().isoformat()
    
    # Return challenge data (training examples + test input only, no solution)
    challenge_data = {
        'id': challenge['id'],
        'train': challenge['data']['train'],
        'test': [{
            'input': test_case['input'] 
        } for test_case in challenge['data']['test']]
    }
    
    return jsonify(challenge_data)

@app.route('/api/verify', methods=['POST'])
def verify_solution():
    """Verify user's solution to the captcha"""
    data = request.get_json()
    
    if not data or 'solution' not in data:
        return jsonify({'success': False, 'error': 'No solution provided'}), 400
    
    current_challenge = session.get('current_challenge')
    current_challenge_file = session.get('current_challenge_file')
    if not current_challenge or not current_challenge_file:
        return jsonify({'success': False, 'error': 'No active challenge'}), 400
    
    user_solution = data['solution']
    
    # Validate the solution
    is_correct = captcha_system.validate_solution(current_challenge_file, user_solution)
    
    if is_correct:
        # Generate verification token
        verification_token = str(uuid.uuid4())
        session['verification_token'] = verification_token
        session['verification_time'] = datetime.now().isoformat()
        
        # Clean up challenge data
        session.pop('current_challenge', None)
        session.pop('current_challenge_file', None)
        session.pop('challenge_start_time', None)
        
        return jsonify({
            'success': True,
            'message': 'Captcha solved successfully!',
            'token': verification_token
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Solution incorrect. Please try again.'
        })

@app.route('/api/validate_token', methods=['POST'])
def validate_token():
    """Validate a verification token (for integration with other systems)"""
    data = request.get_json()
    
    if not data or 'token' not in data:
        return jsonify({'valid': False, 'error': 'No token provided'}), 400
    
    token = data['token']
    session_token = session.get('verification_token')
    verification_time = session.get('verification_time')
    
    if not session_token or not verification_time:
        return jsonify({'valid': False, 'error': 'No verification found'})
    
    # Check if token matches and is not expired (valid for 1 hour)
    if token == session_token:
        verify_time = datetime.fromisoformat(verification_time)
        if datetime.now() - verify_time < timedelta(hours=1):
            return jsonify({'valid': True})
        else:
            return jsonify({'valid': False, 'error': 'Token expired'})
    
    return jsonify({'valid': False, 'error': 'Invalid token'})

@app.route('/demo')
def demo():
    """Demo page showing how the captcha works"""
    return render_template('demo.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 