from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
from PIL import Image

app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Model
DIR = os.path.dirname(os.path.abspath(__file__))
PROTOTXT = os.path.join(DIR, "colorize.prototext")
POINTS = os.path.join(DIR, "pts_in_hull.npy")
MODEL = os.path.join(DIR, "release.caffemodel")

# Check if model files exist
if not os.path.exists(MODEL):
    raise FileNotFoundError(f"Model file not found: {MODEL}")
if not os.path.exists(PROTOTXT):
    raise FileNotFoundError(f"Prototxt file not found: {PROTOTXT}")
if not os.path.exists(POINTS):
    raise FileNotFoundError(f"Points file not found: {POINTS}")

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
pts = pts.transpose().reshape(2, 313, 1, 1)

for layer in net.getLayerNames():
    if "class8_ab" in layer:
        class8 = net.getLayerId(layer)
    if "conv8_313_rh" in layer:
        conv8 = net.getLayerId(layer)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    result_path = os.path.join(RESULT_FOLDER, f"colorized_{file.filename}")
    file.save(filepath)
    
    # Read original image
    image = cv2.imread(filepath)
    original_size = (image.shape[1], image.shape[0])  # Save original dimensions
    
    # Convert image to LAB color space
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # Colorization process
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, original_size)  # Resize colorized result to original size
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (np.clip(colorized, 0, 1) * 255).astype("uint8")
    
    cv2.imwrite(result_path, colorized)
    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    import webbrowser
    from threading import Timer
    
    def open_browser():
        webbrowser.open('http://127.0.0.1:5000')
    
    Timer(1, open_browser).start()
    app.run(debug=True, host='0.0.0.0', port=5000)
