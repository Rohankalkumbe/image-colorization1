from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
import gdown

app = Flask(__name__, template_folder="templates", static_folder="static")
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
MODEL_FOLDER = "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Google Drive Model Links
MODEL_FILES = {
    "release.caffemodel": "1TTQ3NHtozTcPHn5igdVHEdOngWAJeoLR",
    "pts_in_hull.npy": "13gO3-9krpU4Tht4qcyda1Yq_XhSnli6G",
    "colorize.prototext": "1WgRO50tPMeWAvb8pOhyyrA0siZcMAnpQ",
}

# Function to download missing model files
def download_models():
    for filename, file_id in MODEL_FILES.items():
        file_path = os.path.join(MODEL_FOLDER, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)

# Ensure models are available
download_models()

# Load Model
PROTOTXT = os.path.join(MODEL_FOLDER, "colorize.prototext")
POINTS = os.path.join(MODEL_FOLDER, "pts_in_hull.npy")
MODEL = os.path.join(MODEL_FOLDER, "release.caffemodel")

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
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    result_path = os.path.join(RESULT_FOLDER, f"colorized_{file.filename}")
    file.save(filepath)

    # Read and validate the image
    image = cv2.imread(filepath)
    if image is None:
        return "Invalid image file", 400

    original_size = (image.shape[1], image.shape[0])

    # Convert image to LAB color spac
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorization process
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
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open('http://127.0.0.1:5000')

    Timer(1, open_browser).start()
    app.run(debug=True, host='0.0.0.0', port=5000)
