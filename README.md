# Facial Emotion Recognition using CNN & Ensemble Learning

## 📖 Table of Contents
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Model Architecture](#model-architecture)  
4. [Training Process](#training-process)  
5. [Installation & Usage](#installation--usage)  
6. [Results](#results)  
7. [Future Improvements](#future-improvements)  
8. [Contributors](#contributors)  

---

## 📝 Introduction  
This project implements a **Facial Emotion Recognition System** using **Deep Learning (CNNs)** trained on the **FER2013 dataset**. The model can classify faces into **7 emotions**:  
😠 Angry | 🤢 Disgust | 😨 Fear | 😀 Happy | 😢 Sad | 😲 Surprise | 😐 Neutral  

We use an **ensemble learning approach** where multiple CNN models are trained, and their predictions are combined to improve accuracy.

---

## 📊 Dataset  
- The model is trained on the **FER2013 dataset** 📂 (`fer2013.csv`).  
- It contains **35,887 grayscale images** (48x48 pixels) categorized into **7 emotions**.  
- Source: [FER2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  

---

## 🛠 Model Architecture  
Our model consists of **4 convolutional modules** with **residual connections**, followed by:  
✔ **Batch Normalization** for stable training  
✔ **Separable Convolutions** for efficiency  
✔ **Global Average Pooling** to reduce parameters  
✔ **Softmax Output Layer** for classification  

Additionally, an **ensemble model** is trained on CNN predictions using a simple **feedforward neural network**.

---

## 📌 Training Process  
1. **Data Preprocessing:** Normalization, augmentation (rotation, zoom, flip).  
2. **Training Two CNN Models:**  
   - One on the **original images**.  
   - One on **horizontally flipped images**.  
3. **Ensemble Learning:** A final model is trained on the predictions of both CNNs.  
4. **Evaluation Metrics:** Accuracy & Loss curves are plotted.  

---

## 🚀 Installation & Usage  

### 🔹 1. Clone the Repository  
```bash
git clone https://github.com/prasad999999/student-classroom-attention.git
cd student-classroom-attention
```



### 🔹 2. Create a Conda Environment and Install Dependencies 
```bash
conda create --name attention_detection python=3.11 -y
conda activate attention_detection
pip install Flask==2.3.2 numpy==1.26.0 opencv-python==4.8.1.78 keras==2.14.0 \
            face-recognition==1.3.0 tensorflow==2.14.0 matplotlib==3.8.0 pandas==2.1.1 \
            six==1.16.0 pyserial==3.5 flask_socketio==5.5.1 scipy argparse sklearn joblib

```

### 🔹 3. Train the Model  
```bash
python train.py
```
- The trained models will be saved in the `saved_model/` directory.
- Check if the saved_model Directory Contains the Following Files:
- cnn0.h5 // Trained on straight images of humans
- cnn1.h5 // Trained on tilted images of humans
- ensemble.h5 // Combined cnn0.h5 and cnn1.h5
- opencv_face_detector_uint8.pb // Downloaded from GitHub
- opencv_face_detector.pbtxt // Downloaded from GitHub

---

## 📈 Results  
| Model | Test Accuracy |  
|--------|--------------|  
| CNN (Original Images) | 72.5% |  
| CNN (Flipped Images) | 73.1% |  
| **Ensemble Model** | **75.8%** |  

✔ **Ensembling improves accuracy by ~3%** compared to a single model.  

---

### 🔹 4. Install ThinkGear Extension for MindLink EEG Sensor Setup
- Download the ThinkGear extension from the link below: Brainwave Visualizer Download
- https://brainwave-visualizer.software.informer.com/download

### 🔹 5. Connect the MindLink Sensor via Bluetooth
- Select the port number where it is executed in the ThinkGear extension.
- headset = mindwave.Headset('enter port number here')

---

### 🔹 Run simple_custom.py Flask App  
```bash
python simple_custom.py
```
- You should see the following output:
- Running on http://127.0.0.1:5000

### 🔹 Navigate to the Live Monitoring Page
- Enter the current classroom number (as of now, it's static, so you can enter anything).

### 🔹 Navigate to the EEG Monitoring Page
- Enter the student's name to monitor attention through the EEG headset.

### 🔹 View Live Analytics
- The analytics section provides real-time attention analysis of students.

### 🔹 Download the Report
- Switch to the InsightHub page and click on the Download Report button to generate and save the report.


---

## 👥 Contributors  
- **[Prasad Ghadge]** – Developer  
- **[Atharva Atterkar]** – Developer 

---

### 📢 If you like this project, give it a ⭐ on GitHub!  
> **Let's connect on [LinkedIn](https://www.linkedin.com/in/prasad-ghadge-499564260/) 🚀**  

