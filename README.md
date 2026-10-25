# Heart Sound Classifier: Normal vs. Abnormal Screening
This project screens heart audio for abnormalities using Random Forest and MFCC features. Trained on mixed data, it scores 74% accuracy. However, evaluation exposes the "Stethoscope-to-Phone Gap": a model trained strictly on clinical stethoscopes drops to 37% accuracy and 0% recall on mobile recordings, proving domain adaptation is vital.

A machine learning pipeline to classify audio heart recordings as **normal** or **abnormal** (e.g., murmurs, extra heart sounds, and extrasystoles). 

This project establishes a machine learning baseline for heart sound screening and quantifies the **Stethoscope-to-Phone Gap**—the drop in performance when models trained on clinical equipment are deployed on consumer mobile devices.

---

## 📌 Project Overview
* **Task**: Binary audio classification of short heart sounds.
* **Objective**: Screen for abnormal cardiac activity to flag cases that require clinical attention.
* **Core Challenge**: Bridging the acoustic differences between clinical digital stethoscopes and consumer smartphone microphones.

---

## 📊 Dataset
This project uses the **Heartbeat Sounds** dataset from Kaggle, containing:
1. **Set A (Phone app)**: Recordings captured by the general public using the *iStethoscope Pro* iPhone app.
2. **Set B (Clinical)**: Recordings captured in hospital settings using a *DigiScope* digital stethoscope.

### Data Labeling
* **Normal (0)**: Audio with verified normal cardiac rhythms.
* **Abnormal (1)**: Audio containing murmurs, extra heart sounds (`extrahls`), or extrasystole (`extrastole`).
* **Artifacts & Unlabeled**: Filtered out during the preprocessing pipeline to ensure high label quality.

---

## 🛠️ Feature Engineering & Preprocessing
Since raw audio cannot be fed directly into a classifier, we extract **Mel-Frequency Cepstral Coefficients (MFCCs)** to represent the shape of the sound frequency:
1. **Resampling**: Resampled to 4,000 Hz (sufficient for low-frequency heart sounds).
2. **Padding**: Cut or padded to exactly 10 seconds to ensure consistent feature dimensions.
3. **Extraction**: Compute 20 MFCCs over time, summarizing each with its **mean** and **standard deviation** to create **40 features** per recording.

---

## 🧠 Model Pipeline
* **Classifier**: Random Forest Classifier (`n_estimators=300`).
* **Class Balancing**: Adjusted for class imbalance using `class_weight="balanced"` to prioritize predicting the rarer abnormal class.
* **Evaluation Split**: Stratified 80/20 train/test split.

---

## 📈 Experimental Results

### 1. General Performance (Mixed Train/Test)
* **Overall Accuracy**: **74%**
* **Abnormal Precision**: **79%** (When the model flags an abnormality, it is correct 79% of the time)
* **Abnormal Recall (Sensitivity)**: **38%** (The model currently misses 62% of actual abnormalities)

### 2. The Stethoscope-to-Phone Gap (Robustness Test)
To simulate a real-world deployment where a model trained on clinical data is used via a phone app, we trained the classifier **exclusively on Set B** (clinical stethoscope) and tested it **exclusively on Set A** (iPhone app):
* **Result**: Accuracy dropped to **37%** with an abnormal recall of **0%**.
* **Key Finding**: Standard models trained on high-quality clinical stethoscopes do not generalize to noisy smartphone recordings, highlighting the need for noise augmentation and domain adaptation.

---

## 🚀 Next Steps
1. **Data Augmentation**: Inject ambient noise and smartphone-style mic distortions into clinical recordings during training to close the domain gap.
2. **Advanced Architectures**: Convert the audio into spectrograms and train a 2D Convolutional Neural Network (CNN).
3. **Web Deployment**: Wrap the `predict_file` function into a web interface (such as Gradio or Streamlit) where users can upload or record audio.

---

## ⚠️ Medical Disclaimer
*This project is an experimental screening proof-of-concept and is **not** a diagnostic tool. Any results are preliminary and should not be used as a substitute for professional clinical evaluation.*
"""

with open("README.md", "w") as f:
    f.write(readme_content)

print("README.md successfully created! You can now download it from the left sidebar.")
