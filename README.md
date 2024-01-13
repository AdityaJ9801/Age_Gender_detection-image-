# Age and Gender Prediction Model with GUI

This project consists of two main components: 
1. A machine learning model trained on the Kaggle UTKFace dataset.
2. A graphical user interface (GUI) application that uses the trained model for age and gender prediction.

## Usage![Screenshot 2024-01-14 012623](https://github.com/AdityaJ9801/Age_Gender_detection-image-/assets/124603391/7cad4e40-a13f-4601-aabb-fe74c8a15602)

## 1. Age and Gender Prediction Model

### Dataset
The model was trained on the UTKFace dataset, which can be found [here](https://www.kaggle.com/datasets/jangedoo/utkface-new).

### Training Notebook
The training process is documented in the Jupyter Notebook file:
- [age-gender-prediction-model.ipynb](age-gender-prediction-model.ipynb)

#### Dependencies
- TensorFlow
- OpenCV
- Matplotlib
- PIL (Pillow)
- Scikit-learn

#### Instructions
1. Download the UTKFace dataset from the provided link.
2. Execute the code in the notebook to train the age and gender prediction model.
3. Adjust hyperparameters or model architecture if needed.

## 2. GUI Application

### GUI Script
The GUI application is implemented in Python using Tkinter for the user interface and PIL for image processing.

#### File
- [gui.py](gui.py)

#### Dependencies
- Tkinter
- PIL (Pillow)
- TensorFlow
- Numpy

#### Instructions
1. Run the GUI application script using the command `python gui.py`.
2. Upload an image using the "Upload Image" button.
3. Click on the "Detect Image" button to predict the age and gender using the trained model.



- Ensure that the necessary dependencies are installed.
- For the GUI application, make sure the pre-trained model file (`Age_gender_detector_v4_5.h5`) is present in the same directory as `gui.py`.



