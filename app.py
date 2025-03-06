from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    if response.status_code == 200:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        raise Exception(f"Failed to download file with ID: {file_id}")

app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

PROTOTXT_PATH = os.path.join(MODEL_FOLDER, "colorize.prototext")
MODEL_PATH = os.path.join(MODEL_FOLDER, "release.caffemodel")
POINTS_PATH = os.path.join(MODEL_FOLDER, "pts_in_hull.npy")

FILE_IDS = {
    "prototxt": "1TTQ3NHtozTcPHn5igdVHEdOngWAJeoLR",
    "model": "1WgRO50tPMeWAvb8pOhyyrA0siZcMAnpQ",
    "points": "13gO3-9krpU4Tht4qcyda1Yq_XhSnli6G"
}

for key, file_id in FILE_IDS.items():
    destination = globals()[f"{key.upper()}_PATH"]
    if not os.path.exists(destination):
        print(f"Downloading {key} file...")
        download_file_from_google_drive(file_id, destination)
        print(f"{key} file downloaded successfully.")

net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
pts = np.load(POINTS_PATH)
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
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    result_path = os.path.join(RESULT_FOLDER, f"colorized_{file.filename}")
    file.save(filepath)
    
    image = cv2.imread(filepath)
    if image is None:
        return "Invalid image format", 400
    
    original_size = (image.shape[1], image.shape[0])
    lab = cv2.cvtColor(image.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
    L = cv2.split(cv2.resize(lab, (224, 224)))[0] - 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = cv2.resize(net.forward()[0].transpose((1, 2, 0)), original_size)
    colorized = cv2.cvtColor(np.concatenate((cv2.split(lab)[0][:, :, np.newaxis], ab), axis=2), cv2.COLOR_LAB2BGR)
    colorized = (np.clip(colorized, 0, 1) * 255).astype("uint8")
    
    cv2.imwrite(result_path, colorized)
    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
