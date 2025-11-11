# DeepShield-AI-Powered-Deepfake-Image-Detection-using-ResNet50
DeepShield demonstrates the use of deep transfer learning for detecting deepfake images with high accuracy. By leveraging ResNet50’s residual learning capabilities, the project efficiently distinguishes between real and manipulated faces. Its intuitive Streamlit interface and scalable design make it ideal for research, cybersecurity, and forensic applications in the fight against misinformation and AI-generated fake media.
Dataset Preparation:
Images labeled as REAL and FAKE.
Preprocessed using ImageDataGenerator (rescaling, augmentation).

Model Building
Base model: ResNet50 (pre-trained on ImageNet).
Top layers customized for binary classification.

Training:
Fine-tuned on deepfake dataset.

Loss: binary_crossentropy, Optimizer: adam.

Prediction:
Uploaded image → Preprocessed → Model inference → Output label.
