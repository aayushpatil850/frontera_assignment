# frontera_assignment# Audio Classification: Crying, Screaming, and Normal Sounds

This project involves training a deep learning model to classify audio recordings into three categories: "crying", "screaming", and "normal". The dataset consists of labeled audio files, which are preprocessed and used for training a model using the `wav2vec2` architecture. This project leverages the Hugging Face library and transformers for model fine-tuning and evaluation.

## Project Overview

The objective of this project is to classify three types of sounds: crying, screaming, and normal sounds, from audio files. The workflow includes:

1. **Data Preprocessing**: Load, resample, and save audio files.
2. **Data Splitting**: Split the data into training, validation, and test sets.
3. **Model Training**: Fine-tune the `wav2vec2` model for audio classification.
4. **Evaluation**: Evaluate the trained model's performance using metrics such as accuracy, ROC AUC score, and confusion matrix.
5. **Inference**: Provide a function to predict labels for new audio samples.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- [PyTorch](https://pytorch.org/get-started/locally)
- [Transformers](https://huggingface.co/docs/transformers/)
- [Librosa](https://librosa.org/)
- [Soundfile](https://pypi.org/project/SoundFile/)
- [Dataset for crying](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus&ved=2ahUKEwiuk_2Nj9uLAxWCSWwGHXV0KMgQFnoECBMQAQ&usg=AOvVaw0HuDpjnjvDIP1E87xgUmNz)
- [Dataset for Screaming and normal](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset&ved=2ahUKEwjo_Zq2j9uLAxWmSGwGHZmrOaAQFnoECBoQAQ&usg=AOvVaw1hSis4VFi9Rtnkm8xdJEfZ)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Evaluate](https://huggingface.co/docs/evaluate/)
- for testing any single audio file please use wav2vec2.py and for weights use link https://drive.google.com/drive/folders/1PhXknXUffHA70yFVGYIPU72XowtkiKEd?usp=drive_link

You can install the necessary dependencies by running:

```bash
pip install torch transformers librosa soundfile datasets scikit-learn evaluate
