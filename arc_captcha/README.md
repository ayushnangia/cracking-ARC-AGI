# 🧠 ARC AGI Captcha

A next-generation human verification system using visual pattern recognition challenges from the Abstract Reasoning Corpus (ARC). Instead of traditional text-based CAPTCHAs, users solve intuitive visual puzzles that test core cognitive abilities.

## Features

- **🧩 Cognitive Challenge**: Tests genuine human reasoning abilities difficult for AI to replicate
- **🌍 Language Independent**: Visual puzzles work across all languages and cultures
- **🎨 Engaging Experience**: Users enjoy solving puzzles rather than struggling with distorted text
- **🔒 Robust Security**: Harder for bots to solve using current AI pattern recognition
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **🔧 Easy Integration**: RESTful API for integration with any technology stack

## Quick Start

### 1. Installation

```bash
# Clone the repository (if not already done)
git clone <repository_url>
cd cracking-ARC-AGI

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# From the project root directory
python run_captcha.py
```

The server will start at `http://localhost:5001`

### 3. Try the Demo

- **Main Captcha**: [http://localhost:5001](http://localhost:5001)
- **Demo Page**: [http://localhost:5001/demo](http://localhost:5001/demo)

## How It Works

### User Experience

1. **Study Examples**: Users analyze training examples showing input-output transformations
2. **Identify Pattern**: Users recognize the visual transformation rule
3. **Apply Pattern**: Users complete a test case by applying the discovered pattern
4. **Verification**: System validates the solution and provides a verification token

### Technical Flow

1. Frontend requests a challenge from `/api/challenge`
2. User interacts with the visual grid interface
3. Solution is submitted to `/api/verify`
4. Server validates against correct answer
5. Verification token is generated for successful solutions

## API Reference

### Get Challenge

```http
GET /api/challenge?difficulty=medium
```

**Parameters:**
- `difficulty` (optional): `easy`, `medium`, `hard` (default: `medium`)

**Response:**
```json
{
  "id": "challenge_id",
  "train": [
    {
      "input": [[0,1,0], [1,0,1], [0,1,0]],
      "output": [[2,1,2], [1,3,1], [2,1,2]]
    }
  ],
  "test": [
    {
      "input": [[0,1,0], [1,0,1], [0,1,0]]
    }
  ]
}
```

### Verify Solution

```http
POST /api/verify
```

**Request Body:**
```json
{
  "solution": [[2,1,2], [1,3,1], [2,1,2]]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Captcha solved successfully!",
  "token": "uuid-verification-token"
}
```

### Validate Token

```http
POST /api/validate_token
```

**Request Body:**
```json
{
  "token": "uuid-verification-token"
}
```

**Response:**
```json
{
  "valid": true
}
```

## Integration Examples

### HTML/JavaScript

```html
<!DOCTYPE html>
<html>
<head>
    <title>My Form with ARC Captcha</title>
</head>
<body>
    <form id="myForm">
        <!-- Your form fields -->
        <input type="text" name="username" required>
        
                 <!-- ARC Captcha -->
         <iframe src="http://localhost:5001" 
                 width="100%" height="600" 
                 frameborder="0" id="captcha-frame"></iframe>
        
        <button type="submit">Submit</button>
    </form>

    <script>
        // Listen for captcha completion
        window.addEventListener('message', function(event) {
            if (event.data.type === 'captcha-success') {
                // Token received, enable form submission
                document.getElementById('myForm').dataset.token = event.data.token;
            }
        });
    </script>
</body>
</html>
```

### Python Backend Validation

```python
import requests

def verify_captcha_token(token):
    """Verify a captcha token server-side"""
    try:
                 response = requests.post('http://localhost:5001/api/validate_token', 
                                json={'token': token})
        result = response.json()
        return result.get('valid', False)
    except:
        return False

# In your form handler
@app.route('/submit_form', methods=['POST'])
def submit_form():
    token = request.form.get('captcha_token')
    
    if not verify_captcha_token(token):
        return "Invalid captcha", 400
    
    # Process form submission
    return "Form submitted successfully"
```

### Node.js/Express Integration

```javascript
const axios = require('axios');

async function verifyCaptcha(token) {
    try {
                 const response = await axios.post('http://localhost:5001/api/validate_token', {
             token: token
         });
        return response.data.valid;
    } catch (error) {
        return false;
    }
}

app.post('/submit', async (req, res) => {
    const { captcha_token } = req.body;
    
    if (!await verifyCaptcha(captcha_token)) {
        return res.status(400).json({ error: 'Invalid captcha' });
    }
    
    // Process form submission
    res.json({ success: true });
});
```

## Customization

### Difficulty Levels

- **Easy**: Smaller grids (≤50 cells), simpler patterns
- **Medium**: Medium grids (50-150 cells), moderate complexity
- **Hard**: Larger grids (>150 cells), complex patterns

### Styling

The interface can be customized using CSS. Key classes:

- `.container`: Main container
- `.grid`: Grid display area
- `.grid-cell`: Individual grid cells
- `.color-0` to `.color-9`: ARC color classes
- `.btn-primary`, `.btn-secondary`: Button styles

### Color Scheme

ARC uses a 10-color palette (0-9):
- 0: Black (`#000000`)
- 1: Blue (`#0074D9`)
- 2: Red (`#FF4136`)
- 3: Green (`#2ECC40`)
- 4: Yellow (`#FFDC00`)
- 5: Gray (`#AAAAAA`)
- 6: Magenta (`#F012BE`)
- 7: Orange (`#FF851B`)
- 8: Light Blue (`#7FDBFF`)
- 9: Brown (`#870C25`)

## Security Considerations

### Token Validation

- Tokens expire after 1 hour
- Each token can only be used once
- Tokens are tied to user sessions
- Server-side validation is required

### Rate Limiting

Consider implementing rate limiting to prevent abuse:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/challenge')
@limiter.limit("10 per minute")
def get_challenge():
    # ... existing code
```

## Development

### Project Structure

```
arc_captcha/
├── app.py              # Main Flask application
├── static/
│   ├── css/
│   │   └── style.css   # Styling
│   └── js/
│       └── captcha.js  # Frontend logic
├── templates/
│   ├── index.html      # Main captcha interface
│   └── demo.html       # Demo/documentation page
└── README.md           # This file
```

### Local Development

```bash
# Run in development mode
cd arc_captcha
python app.py

# Or use Flask development server
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

### Testing

Test the API endpoints:

```bash
# Get a challenge
curl "http://localhost:5001/api/challenge?difficulty=easy"

# Submit a solution (replace with actual solution)
curl -X POST "http://localhost:5001/api/verify" \
     -H "Content-Type: application/json" \
     -d '{"solution": [[0,1,0], [1,0,1], [0,1,0]]}'

# Validate a token (replace with actual token)
curl -X POST "http://localhost:5001/api/validate_token" \
     -H "Content-Type: application/json" \
     -d '{"token": "your-token-here"}'
```

## Deployment

### Production Setup

1. **Use a production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5001 app:app
   ```

2. **Set production configuration**:
   ```python
   # In app.py
   app.secret_key = os.environ.get('SECRET_KEY', 'your-secure-secret-key')
   ```

3. **Use environment variables**:
   ```bash
   export SECRET_KEY="your-secure-secret-key"
   export FLASK_ENV="production"
   ```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
WORKDIR /app/arc_captcha

EXPOSE 5001
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "app:app"]
```

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure ARC dataset files exist in the correct location
2. **Import errors**: Install requirements with `pip install -r requirements.txt`
3. **Port conflicts**: Change the port in `app.py` if 5001 is in use
4. **CORS issues**: Ensure `flask-cors` is installed and configured

### Debug Mode

Run with debug enabled:

```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## License

This project uses the ARC dataset, which is available under the Apache License 2.0. Please refer to the original ARC repository for dataset licensing terms.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Test with the demo page first
- Ensure all dependencies are installed 