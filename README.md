# Real-Time Sign Language Prediction

## Overview
This project aims to predict American Sign Language (ASL) gestures for alphabets A-Z in real-time using a webcam. It leverages a pre-trained VGG16 model for image classification and integrates with a user-friendly interface built with Streamlit.

## Features
- **Real-time prediction of ASL gestures (A-Z)**
- **Webcam integration for capturing gestures**
- **User-friendly interface built with Streamlit**
- **High accuracy of 0.99**

## Interface
### Real-Time-Prediction
![Alphabet F](https://github.com/GOURIKP/Real-Time-Sign-Language-Prediction/blob/main/Related_images/F.png)

![Alphabet Y](https://github.com/GOURIKP/Real-Time-Sign-Language-Prediction/blob/main/Related_images/Y.png)

## Dataset
The project uses the ASL Alphabet Dataset from Kaggle:
[ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Tools and Technologies
- **Python**: Programming language used for the project.
- **TensorFlow and Keras**: For building and training the VGG16 model.
- **OpenCV**: For webcam integration and image processing.
- **Streamlit**: For creating the web interface.
- **Numpy**: For numerical operations.
- **Pandas**: For data handling and preprocessing.
- **Matplotlib**: For visualizing performance metrics.

## Model
The project utilizes the VGG16 model, a pre-trained convolutional neural network, to classify ASL gestures. The model has been fine-tuned on the ASL Alphabet Dataset to achieve a high accuracy of 0.99.

## Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/real-time-sign-language-prediction.git
   cd real-time-sign-language-prediction
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit interface:**
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Open the Streamlit app**: After running the above command, a local URL will be provided.
2. **Allow webcam access**: The app will request permission to access the webcam.
3. **Show ASL gestures**: Show any ASL alphabet gesture (A-Z) in front of the webcam.
4. **Real-time prediction**: The predicted alphabet will be displayed in real-time.

## Performance
The model achieves an accuracy of 0.99 on the test set. Below are some performance metrics and visualizations:

### Accuracy
![Accuracy](https://github.com/GOURIKP/Real-Time-Sign-Language-Prediction/blob/main/Related_images/training%20accuracy.png)

### Loss
![Loss](https://github.com/GOURIKP/Real-Time-Sign-Language-Prediction/blob/main/Related_images/training%20loss.png)

### Confusion Matrix
![Confusion Matrix](https://github.com/GOURIKP/Real-Time-Sign-Language-Prediction/blob/main/Related_images/Confusion%20matrix-sign.png)


## Contributing
Contributions are welcome! Please fork this repository and submit pull requests for any features, improvements, or bug fixes.


## Acknowledgements
- The [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) provided by GrassKnoted on Kaggle.
- The developers and maintainers of TensorFlow, Keras, OpenCV, and Streamlit for their invaluable tools and libraries.
