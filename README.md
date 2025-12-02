# Brain-Computer Interface (BCI) for Emotion Detection

## 🧠 Project Overview
A real-time emotion detection system using EEG signals, featuring machine learning models and an interactive Streamlit interface. The system classifies emotions into three categories: Negative, Neutral, and Positive.

## 🚀 Features
- Real-time EEG signal processing
- Multiple feature extraction methods (Bandpower, Wavelet, Entropy)
- Machine Learning models (Random Forest, XGBoost, EEGNet)
- Interactive Streamlit dashboard
- Cross-dataset validation support
- Model performance visualization

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bci-emotion-detection.git
   cd bci-emotion-detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Quick Start

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Train a new model**
   ```bash
   python run_pipeline.py
   ```

## 📁 Project Structure
```
.
├── app.py                  # Main Streamlit application
├── app_enhanced.py         # Enhanced UI version
├── run_pipeline.py         # Training pipeline
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore             
└── modules/
    ├── data_loader.py     # Data loading and preprocessing
    ├── feature_extractor.py # Feature extraction utilities
    └── models/            # Model architectures
```

## 📊 Results
- Model Accuracy: 99.06%
- Inference Speed: ~5ms per prediction
- Supported Emotions: Negative, Neutral, Positive

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact
For questions or feedback, please open an issue or contact arman.kothariya786@gmail.com 

