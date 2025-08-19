# 🖐 HandTalk

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.7.0.68-blue)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/mediapipe-0.10.21-orange)](https://mediapipe.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-green)](https://scikit-learn.org/)

A Python project for *real-time hand gesture recognition* using *OpenCV, **MediaPipe, and **scikit-learn*.  
This project allows you to collect hand gesture images, create a dataset, train a classifier, and perform live gesture recognition.

---

## 🚀 Features

- Capture hand gesture images using a webcam.  
- Extract hand landmarks and create a normalized dataset.  
- Train a *Random Forest classifier* on the collected data.  
- Real-time gesture recognition from webcam input.  

---

## 📂 Project Structure



HandTalk/
│
├── collect\_imgs.py           # Capture hand gesture images
├── create\_dataset.py         # Extract hand landmarks and save dataset
├── train\_classifier.py       # Train Random Forest classifier
├── inference\_classifier.py   # Real-time gesture recognition
├── data/                     # Folder to store captured images
├── data.pickle               # Pickled dataset of hand landmarks
├── model.p                   # Trained Random Forest model
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

`

---

## ⚡ Requirements

- Python 3.10+  
- Install dependencies via:

bash
pip install -r requirements.txt
`

*requirements.txt*

text
opencv-python==4.7.0.68
mediapipe==0.10.21
scikit-learn==1.2.0
numpy>=1.24


---

## 🖐 Usage Guide

### 1️⃣ Collect Hand Gesture Images

bash
python collect_imgs.py


* Images are saved in ./data/<class_number>/.
* Press *Q* to start capturing after positioning your hand.
* Each class collects *100 images* by default.
* Modify number_of_classes and dataset_size in the script if needed.

---

### 2️⃣ Create Dataset (Hand Landmarks)

bash
python create_dataset.py


* Extracts hand landmarks using MediaPipe.
* Normalizes coordinates and saves to data.pickle.

---

### 3️⃣ Train Classifier

bash
python train_classifier.py


* Trains a Random Forest classifier on the dataset.
* Prints test set accuracy.
* Saves the model to model.p for later inference.

---

### 4️⃣ Real-Time Gesture Recognition

bash
python inference_classifier.py


* Uses your webcam to detect hand gestures.
* Predicts gestures using the trained model.

---

## 🔧 Customization

* *Number of Classes*: number_of_classes in collect_imgs.py
* *Images per Class*: dataset_size in collect_imgs.py
* *Model Parameters*: Adjust RandomForestClassifier settings in train_classifier.py
* *Camera Index*: If webcam index 0 does not work, try 1 or 2 in cv2.VideoCapture(0)

---

## 📌 Tips for Best Results

* Ensure *good lighting* and a *plain background*.
* Use *one hand per image* for better detection.
* Keep hand *fully visible* in the camera frame.

---

## 📝 License

This project is *open source* – feel free to use, modify, and share.

---

## 👩‍💻 Author

*Dhanya-prabhu*
