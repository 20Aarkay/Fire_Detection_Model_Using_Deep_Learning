# 🔥 Fire Detection Model Using Deep Learning

A deep learning-based web application that detects **fire**, **smoke**, or **safe conditions** from both uploaded images and live webcam footage. Built using **TensorFlow/Keras**, **Flask**, and **OpenCV**, with a responsive UI powered by **Bootstrap**.

---

## 🚀 Features

- 🔍 Upload image for fire/smoke detection
- 🎥 Real-time webcam fire detection
- 🔔 Plays alarm when fire is detected with high confidence
- 💡 Intuitive web interface
- 📱 Android and desktop browser compatibility

---

## 📁 Project Structure

```
Fire_Detection_Model_Using_Deep_Learning/
├── app/
│   ├── app.py                  # Flask web server
│   ├── realtime.py             # Real-time webcam logic
│   ├── templates/
│   │   └── index.html          # UI layout
│   └── static/
│       ├── uploads/            # Uploaded image storage
│       └── alarm.mp3           # Alarm sound on fire detection
│
├── fire detection model/
│   ├── fire_detection_model.keras   # Trained DL model
│   └── class_indices.json           # Class name to index mapping
│
├── Dataset/                   # Training/test images
│   ├── train/
│   │   ├── fire/
│   │   ├── smoke/
│   │   └── none fire/
│   └── test/
│       ├── fire/
│       ├── smoke/
│       └── none fire/
│
├── train.py                  # Training script
└── README.md                 # Project description
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Fire_Detection_Model_Using_Deep_Learning.git
cd Fire_Detection_Model_Using_Deep_Learning/app
```

### 2. Install Dependencies

```bash
pip install -r ../requirements.txt
```

Or manually install:

```bash
pip install flask tensorflow opencv-python numpy
```

### 3. Run the App

```bash
python app.py
```

Then visit:  
🌐 `http://127.0.0.1:5000/` in your browser.

---

## 🧪 How It Works

1. **Upload Image**: App uses a CNN model to classify input into `fire`, `smoke`, or `none fire`.
2. **Real-Time Detection**: Accesses your webcam feed and runs continuous predictions.
3. **Alarm Trigger**: If `fire` is detected with >70% confidence, an alarm sound is played.

---

## 🔍 Model Info

- Model type: Convolutional Neural Network (CNN)
- Input size: 224x224 RGB images
- Output classes:
  - `fire`
  - `smoke`
  - `none fire`

---

## 🧠 Training (Optional)

To train or retrain the model:

```bash
python train.py
```

Ensure your `Dataset/` folder is properly structured with `train/` and `test/` subfolders.

---

## 📱 Mobile Compatibility

- Mobile browsers supported (e.g., Chrome on Android)
- Webcam permissions required for real-time view

---

## 👥 Authors

- **Ram Krishna Singh**
