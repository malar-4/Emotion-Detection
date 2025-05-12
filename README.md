# Emotion-Detection
Emotion-detection
This project demonstrates emotion recognition using both audio and facial image inputs. It includes two main components:

Audio Emotion Recognition using MFCC + Pitch features and an SVM classifier.
Facial Emotion Recognition using a pre-trained deep learning model (Mini-XCEPTION).
⚙️ Installation
This project is designed for Google Colab, but can also run locally with proper setup.

Install dependencies:
pip install librosa soundfile scikit-learn openai-whisper streamlit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install moviepy transformers keras opencv-python
Download Pretrained Models:
wget https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5 -O emotion_model.h5
🧪 How to Use
1. 🎵 Audio Emotion Recognition
Upload emotion-labeled audio files (e.g., happy.wav, sad.wav).

Extract MFCC + Pitch features using librosa.

Train a linear SVM model.

Use OpenAI's Whisper model to transcribe speech.

Predict emotion from a new test file (e.g., test.wav).

Example Workflow:
features = extract_features("happy.wav")
model = train_model(X, y)
emotion = model.predict(test_features)
2. 😊 Facial Emotion Recognition
Upload a face image.

Detect the face using OpenCV.

Resize and normalize the region of interest.

Predict emotion using the pre-trained Mini-XCEPTION model.

Example Output:
Detected Emotion: Happy
🔍 Dependencies
librosa

soundfile

scikit-learn

openai-whisper

opencv-python

keras

torch, torchaudio

moviepy

transformers

💡 Future Improvements
Expand dataset with more emotion categories.

Real-time webcam detection for facial emotions.

Merge both models for multi-modal emotion analysis.
