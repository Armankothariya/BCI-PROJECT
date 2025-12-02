# 📘 Emotion-Aware Brain–Computer Interface (BCI) Using EEG Signals**

## 🎧 **Emotion-Aware Brain–Computer Interface for Real-Time Music Control**

**Authors:** Kothariya Mohamad Arman. 
**Dataset:** Prof. Jordan J. Bird – *EEG Brainwave Dataset: Feeling Emotions*

---

# 🧠 **1. Project Overview**

This project develops a **real-time, reproducible, and interpretable EEG-based BCI** that can detect emotional state using consumer-grade EEG signals.
The system controls **music playback** based on predicted emotion.

### **Key Achievements**

* **99.06% accuracy** using *Random Forest*
* **Processing latency <10 ms** (true real-time capability)
* **Reproducible ML pipeline** via modular code + config file
* **Statistically validated results (CV, bootstrap, permutation test)**
* **Complete feature-level interpretability (spectral features)**

---

# 🧩 **2. Emotion Classes**

We classify EEG signals into **three emotional states**:

* **Positive (P)**
* **Neutral (N)**
* **Negative (N)**

This matches the structure used in Prof. Bird’s dataset.

---

# 📂 **3. Repository Structure**

```
BCI-Emotion-Recognition/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model_trainer.py
│   ├── validator.py
│   ├── real_time_sim.py
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_features.ipynb
│   ├── 03_training.ipynb
│   └── 04_validation.ipynb
│
├── results/
│   ├── confusion_RF.png
│   ├── confusion_XGB.png
│   ├── feature_importance.png
│   ├── latency_breakdown.png
│   └── accuracy_curve.png
│
├── docs/
│   ├── System_Architecture.png
│   ├── Pipeline.png
│   └── Mini_Paper.pdf
│
├── config.yaml
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# 🧪 **4. Dataset Information**

**Dataset Used:** Prof. Jordan J. Bird – *EEG Brainwave Dataset: Feeling Emotions*

* **2131 samples** (4-channel consumer-grade EEG)
* Recorded during emotional stimuli
* Pre-labeled into **Positive, Neutral, Negative**
* Frequency-rich signals ideal for spectral analysis

---

# 🔧 **5. Methodology Pipeline**

### **1. Data Loading**

* Load EEG CSV files
* Merge channels, timestamps, labels
* Clean missing values

### **2. Preprocessing**

* Band-pass filter **1–40 Hz**
* Notch filter **50/60 Hz**
* Standardization

### **3. Feature Extraction (Your implementation)**

You selected **Bandpower features**, extracted across canonical EEG bands:

| Band  | Frequency | Emotional Relevance  |
| ----- | --------- | -------------------- |
| Delta | <4 Hz     | Deep cognitive state |
| Theta | 4–8 Hz    | Emotional engagement |
| Alpha | 8–12 Hz   | Relaxation, calmness |
| Beta  | 12–30 Hz  | Arousal, stress      |
| Gamma | >30 Hz    | Higher cognition     |

Calculated for all channels → Feature vector.

### **4. Model Training**

Models tested:

* Random Forest (Best)
* XGBoost
* Logistic Regression (baseline)

---

# 🏆 **6. Results**

### **Best Model:** **Random Forest**

* **Accuracy:** **99.06%**
* **Latency:** `<10 ms` per inference
* **Balanced performance across all classes**
* Fast & interpretable → ideal for BCI

#### **Confusion Matrix (RF)**

*(Add as image in results folder)*

#### **Feature Importance**

Alpha & Beta bandpower were most influential → matches neuroscience literature.

---

# 🔬 **7. Validation (Scientific Rigor)**

To confirm results aren’t random or overfitted:

### ✔ **5-Fold Cross Validation**

`Mean = 98.30% ± 1.01%`

### ✔ **Bootstrap (1000×)**

`95% CI = [98.21%, 99.91%]`

### ✔ **Permutation Test (1000×)**

`p < 0.001`
The model learns meaningful patterns, not noise.

### ✔ **Sanity Check**

Random labels → ~33% accuracy
(Chance level for 3-class problem)

---

# ⚡ **8. Real-Time Mode**

I implemented **real-time simulation**:

* Live feature stream
* Instant classification (ms range)
* Music control logic based on emotion:

  * Positive → Energetic track
  * Neutral → Balanced track
  * Negative → Calming track

This demonstrates **true interactive BCI capability**.

---

# ⚠️ **9. Limitations**

* Only **4 EEG channels** (limited spatial resolution)
* Neutral vs Negative is still challenging
* Dataset is controlled (not noisy real-world EEG)
* No cross-dataset generalization yet

---

# 🚀 **10. Future Work**

* Integrate **OpenBCI/Emotiv** for live streaming
* Add **CSP, entropy, and wavelet features**
* Subject-independent models (transfer learning)
* Validate on larger datasets (DEAP, SEED)
* Expand to IoT controls (lights, appliances)
* Assistive devices (wheelchairs, prosthetics)

---

# 📦 **11. Installation**

```
git clone https://github.com/<your-username>/BCI-Emotion-Recognition
cd BCI-Emotion-Recognition
pip install -r requirements.txt
```

Run full pipeline:

```
python run_pipeline.py --config config.yaml
```

---

# 📝 **12. Citation**

Bird, J.J. et al., “EEG Brainwave Dataset: Feeling Emotions,” **Open Source**, 2020.

---

# 🙌 **13. Acknowledgments**

* Prof. Jordan Bird (Dataset)
* DEPSTAR IT Department
* Open-source community
