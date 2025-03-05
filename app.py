from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
import requests
from PIL import Image

app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_FOLDER = "/tmp/uploads"
RESULT_FOLDER = "/tmp/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

BASE_DIR = os.getcwd()
PROTOTXT = os.path.join(BASE_DIR, "models", "colorize.prototext")
MODEL = os.path.join(BASE_DIR, "models", "release.caffemodel")
POINTS = os.path.join(BASE_DIR, "models", "pts_in_hull.npy")

MODEL_URL = "https://drive.google.com/uc?export=download&id=1WgRO50tPMeWAvb8pOhyyrA0siZcMAnpQ"
os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL):
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully.")

if not os.path.exists(PROTOTXT) or not os.path.exists(MODEL) or not os.path.exists(POINTS):
    raise FileNotFoundError("Model files are missing. Upload them to the 'models/' directory.")

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
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
    
    image = cv2.imread(filepath)
    original_size = (image.shape[1], image.shape[0])
    
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, original_size)
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (np.clip(colorized, 0, 1) * 255).astype("uint8")
    
    cv2.imwrite(result_path, colorized)
    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
