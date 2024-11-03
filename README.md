# Voice Recognition Using Deep Learning

## Project Overview
This project demonstrates a basic voice recognition system utilizing deep learning techniques. Audio data is preprocessed into spectrograms and MFCCs (Mel-frequency cepstral coefficients) to serve as features for a classification model, which is trained to differentiate between specific voices.

The project leverages libraries like `librosa` for audio processing, `TensorFlow` for model training, and `plotly` and `matplotlib` for data visualization.

## Features
- Audio file loading and processing: converts audio clips into spectrograms and MFCC features.
- Deep learning model training using TensorFlow.
- Visualization of audio features (waveforms, spectrograms, and MFCCs).
- Model evaluation with metrics like accuracy, confusion matrix, and classification report.

## Requirements
- Python 3.x
- Required libraries:
    - `librosa`
    - `tensorflow`
    - `scikit-learn`
    - `matplotlib`
    - `plotly`
    - `seaborn`

## Setup Instructions
1. **Prepare Audio Data**:
    - Place audio files of the specific voice in the `my_voice/` folder (labeled `1`).
    - Place other audio files in the `other_voice/` folder (labeled `0`).
    - Audio files will be processed, padded, or truncated to a consistent length for training.

2. **Preprocess Data**:
    - Audio is processed using `librosa` to create spectrograms with consistent dimensions.
    - Spectrograms and MFCCs are used as input features for the model.

3. **Train the Model**:
    - Run the provided script to train the deep learning model on the processed dataset.
    - Model accuracy and loss are visualized after training.

4. **Evaluate the Model**:
    - The script outputs a classification report and confusion matrix to evaluate model performance on the test set.

## Usage
- To train and evaluate the model, run the main script after preparing the dataset.
- Visualize audio features by using the `plot_audio_features()` function to display waveforms, spectrograms, and MFCCs.

## Visualizations
This project includes functions to plot:
- **Waveform**: Shows the audio signal in the time domain.
- **Spectrogram**: Displays the frequency spectrum of the audio.
- **MFCC**: Plots the Mel-frequency cepstral coefficients.

## Results
The model achieves a classification accuracy of 100%. The final evaluation includes a confusion matrix and a detailed classification report.
