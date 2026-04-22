from flask import Flask, request, render_template, jsonify
import os
import sys
import numpy as np
from tensorflow.keras import backend as K
import traceback
from tensorflow.keras.preprocessing import image

def preprocess_local_image(img_path):
    # 1. Load image in grayscale ('grayscale=True' or 'color_mode="grayscale"')
    # 2. Resize to 256x256 to match the model's expected dimensions
    img = image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')
    
    # 3. Convert to array
    img_array = image.img_to_array(img)
    
    # 4. Normalize (the model was trained on 1/255.0 scaling)
    img_array = img_array / 255.0
    
    # 5. Add batch dimension: (1, 256, 256, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Example usage for your local script:
# prepared_image = preprocess_local_image('path_to_patient_xray.jpg')
# prediction = model.predict(prepared_image)
# print("Pneumonia Probability: ", prediction[0][1]) # Assuming [Normal, Pneumonia]

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_pneumonia_cnn.keras')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("=" * 70)
print("🫁 PNEUMONIA DETECTION SYSTEM - DETAILED DIAGNOSTICS")
print("=" * 70)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check if model file exists
print(f"\n1️⃣ Checking model file...")
print(f"   Looking for: {MODEL_PATH}")
print(f"   In directory: {os.getcwd()}")

if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"   ✅ Model file found!")
    print(f"   📦 File size: {file_size:.2f} MB")
else:
    print(f"   ❌ Model file NOT found!")
    print(f"\n   Files in current directory:")
    for f in os.listdir('.'):
        print(f"      - {f}")
    print(f"\n   ⚠️  Please move best_pneumonia_cnn.keras to: {os.getcwd()}")

# Check TensorFlow
print(f"\n2️⃣ Checking TensorFlow...")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"   ❌ TensorFlow not installed: {e}")
    print(f"   📝 Install with: pip install tensorflow")
    sys.exit(1)

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

# Load model
print(f"\n4️⃣ Loading model...")
model = None
model_error = None

if os.path.exists(MODEL_PATH):
    try:
        from tensorflow import keras
        print(f"   Loading from: {MODEL_PATH}")
        model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
        )
        print(f"   ✅ Model loaded successfully!")
        print(f"   📊 Model type: {type(model).__name__}")

        
    except Exception as e:
        print(f"   ❌ Error loading model:")
        traceback.print_exc()
        print(f"      {type(e).__name__}: {e}")
        
        model_error = str(e)
        model = None
else:
    model_error = "Model file not found"
    print(f"   ❌ Cannot load - file doesn't exist")

print("\n" + "=" * 70)

CLASS_LABELS = ['NORMAL', 'PNEUMONIA']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    import cv2
    import numpy as np
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def predict_pneumonia(image_path):
    if model is None:
        raise ValueError(f"Model not loaded. Error: {model_error}")
    
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image, verbose=0)

    pneumonia_prob = float(predictions[0][1]) * 100
    normal_prob = float(predictions[0][0]) * 100

    # 🎯 Threshold logic
    if pneumonia_prob >= 85:
        result = "PNEUMONIA"
        confidence_level = "HIGH"
        message = "High likelihood of pneumonia detected."
    
    elif pneumonia_prob <= 40:
        result = "NORMAL"
        confidence_level = "HIGH" if normal_prob >= 85 else "LOW"
        message = "No strong indicators of pneumonia detected." if confidence_level == "HIGH" else "Low confidence normal prediction."
    
    else:
        result = "INCONCLUSIVE"
        confidence_level = "MEDIUM"
        message = "The model is uncertain about this prediction. Further medical evaluation is recommended."

    return {
        'result': result,
        'confidence': round(max(pneumonia_prob, normal_prob), 2),
        'confidence_level': confidence_level,
        'message': message,
        'probabilities': {
            'NORMAL': round(normal_prob, 2),
            'PNEUMONIA': round(pneumonia_prob, 2)
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': f'Model not loaded. {model_error}',
            'help': 'Please check that best_pneumonia_cnn.keras is in the project folder'
        }), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    try:
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_pneumonia(filepath)
        result['image_path'] = filepath
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'type': type(e).__name__
        }), 500

@app.route('/status')
def status():
    """Check system status"""
    return jsonify({
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'model_error': model_error,
        'current_directory': os.getcwd(),
        'tensorflow_version': tf.__version__
    })

if __name__ == '__main__':
    if model is None:
        print("\n" + "⚠️  " * 20)
        print("WARNING: Model not loaded!")
        print("The server will start but predictions will fail.")
        print("Please fix the model loading issue above.")
        print("⚠️  " * 20 + "\n")
    
    print("\n🚀 Starting Flask server...")
    print(f"   Main app: http://localhost:5000")
    print(f"   Status:   http://localhost:5000/status")
    print("\n" + "=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)