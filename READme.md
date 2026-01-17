# American Sign Language Detection System

Real-time ASL alphabet detection using deep learning and computer vision.

## 📋 Project Overview

This system detects American Sign Language (ASL) hand signs in real-time through a webcam and classifies them into 29 classes (A-Z, SPACE, DELETE, NOTHING).

**Key Features:**
- Real-time hand detection and classification
- Transfer learning with MobileNetV2
- Auto hand region detection
- Live prediction with confidence scores
- Prediction history tracking

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the ASL dataset and extract it. Your folder structure should be:
```
asl-detection/
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       ├── C/
│       └── ... (all 29 classes)
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

### 3. Train the Model

```bash
python train_model.py
```

**Training Details:**
- Uses MobileNetV2 with transfer learning
- Data augmentation for better generalization
- Early stopping to prevent overfitting
- Takes ~15-30 minutes on a decent GPU (2-4 hours on CPU)

**Output Files:**
- `best_asl_model.h5` - Best model (use this)
- `asl_model_final.h5` - Final epoch model
- `class_indices.json` - Class mappings
- `training_history.png` - Training plots

### 4. Run Real-time Detection

```bash
python app.py
```

**Controls:**
- `q` - Quit
- `r` - Toggle hand detection (ROI)
- `c` - Clear prediction history
- `s` - Save screenshot

---

## 📊 How It Works

### Training Pipeline

1. **Data Preparation**
   - Images resized to 224x224
   - 80% training, 20% validation split
   - Data augmentation: rotation, shift, zoom, flip

2. **Model Architecture**
   - Base: MobileNetV2 (pre-trained on ImageNet)
   - Custom layers: Global pooling → Dense(256) → Output(29)
   - Dropout for regularization

3. **Training Strategy**
   - Optimizer: Adam (lr=0.001)
   - Loss: Categorical crossentropy
   - Early stopping on validation loss
   - Learning rate reduction on plateau

### Detection Pipeline

1. **Hand Detection** (Optional)
   - Skin color detection in HSV space
   - Morphological operations for noise removal
   - Bounding box around largest contour

2. **Prediction**
   - Preprocess frame (resize, normalize)
   - Model inference
   - Confidence thresholding (70%)

3. **Display**
   - Live prediction overlay
   - FPS counter
   - Prediction history

---

## 📈 Expected Performance

**Training Results (typical):**
- Training Accuracy: ~98-99%
- Validation Accuracy: ~95-97%
- Training Time: 15-30 mins (GPU)

**Real-time Performance:**
- FPS: 15-30 (depending on hardware)
- Latency: ~30-60ms per prediction

---

## 🎯 Tips for Best Results

### During Training:
- Ensure all 29 class folders exist
- Check for class imbalance
- Monitor validation loss for overfitting
- Use GPU if available (`tensorflow-gpu`)

### During Detection:
- Good lighting conditions
- Plain background preferred
- Keep hand centered in ROI
- Clear hand gestures
- ~1-2 feet from camera

---

## 🛠️ Troubleshooting

### Model Training Issues

**"No module named tensorflow"**
```bash
pip install tensorflow==2.15.0
```

**"Out of memory" error**
- Reduce BATCH_SIZE in train_model.py (try 16 or 8)
- Close other applications

**Low accuracy**
- Train for more epochs (increase EPOCHS)
- Check data quality
- Ensure proper class distribution

### Detection Issues

**"Cannot access camera"**
- Check camera permissions
- Try different camera index: `cv2.VideoCapture(1)`

**Low FPS**
- Use smaller model or reduce image size
- Close background applications
- Consider using GPU

**Poor predictions**
- Ensure good lighting
- Use plain background
- Toggle ROI detection (press 'r')
- Adjust hand distance from camera

---

## 📁 Project Structure

```
asl-detection/
├── train_model.py          # Model training script
├── app.py                  # Real-time detection app
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── best_asl_model.h5      # Trained model (generated)
├── class_indices.json     # Class mappings (generated)
└── training_history.png   # Training plots (generated)
```

---

## 🎓 For Recruiters / Interviewers

### Technical Skills Demonstrated

**Machine Learning:**
- Transfer learning implementation
- Data augmentation techniques
- Model evaluation and validation
- Hyperparameter tuning

**Computer Vision:**
- Image preprocessing
- Real-time video processing
- Hand detection using color segmentation
- ROI extraction

**Software Engineering:**
- Clean, modular code
- Error handling
- User-friendly interface
- Documentation

### Potential Interview Questions & Answers

**Q: Why MobileNetV2?**
- Lightweight (good for real-time)
- Pre-trained on ImageNet (transfer learning)
- Good accuracy-speed tradeoff
- ~3.5M parameters vs ResNet50's 25M

**Q: How did you handle overfitting?**
- Data augmentation (rotation, shift, zoom)
- Dropout layers (30%)
- Early stopping
- Validation split monitoring

**Q: What challenges did you face?**
- Class imbalance in dataset
- Hand detection in varying lighting
- Real-time performance optimization
- False positives with similar signs

**Q: How would you improve this?**
- Add more robust hand detection (MediaPipe/YOLO)
- Implement sequence detection for words
- Deploy as web app (Flask/FastAPI)
- Add more data for rare classes
- Try different architectures (EfficientNet)

---

## 🔮 Future Improvements

- [ ] MediaPipe integration for better hand detection
- [ ] Word/phrase detection (sequence of letters)
- [ ] Web deployment (Flask/Streamlit)
- [ ] Mobile app version
- [ ] Support for dynamic signs (not just static)
- [ ] Multi-hand detection
- [ ] Custom dataset expansion

---

## 📝 Notes

- **Dataset**: Make sure to update `TRAIN_PATH` in train_model.py
- **GPU**: TensorFlow will auto-detect and use GPU if available
- **Camera**: Uses default camera (index 0), change if needed
- **Python Version**: Tested on Python 3.8-3.10

---

## 📧 Contact

Feel free to reach out for questions or improvements!

**Project by:** [Your Name]  
**Email:** [Your Email]  
**LinkedIn:** [Your LinkedIn]  
**GitHub:** [Your GitHub]

---

## 📄 License

This project is for educational purposes. Dataset credits to the original creators.# Real-Time-American-Sign-Language-detection
