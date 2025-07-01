import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

# ✅ Set up Flask app
app = Flask(__name__)

# ✅ Set up upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load model and prepare
device = torch.device("cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 classes: update if needed
model.load_state_dict(torch.load('resnet18_fast/model.pt', map_location=device))
model.to(device)
model.eval()

# ✅ Label index
class_labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# ✅ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'pc_image' not in request.files:
        return "No file uploaded"

    file = request.files['pc_image']
    if file.filename == '':
        return "No selected file"

    # Save the uploaded image
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        prediction = class_labels[predicted.item()]

    # Render output
    return render_template('contact.html', predict=prediction)

# ✅ Main
if __name__ == '__main__':
    app.run(debug=True)
