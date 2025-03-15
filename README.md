Here's a **README.md** file for your **Face Analysis Algorithm** repository:  

```md
# Face Analysis Algorithm  

This repository contains an advanced **Face Analysis Algorithm** that detects and analyzes facial features using deep learning models. The project leverages state-of-the-art convolutional neural networks (CNNs) for facial classification and recognition tasks.  


## Installation  

1. **Clone the Repository**  
   ```bash
   git clone --branch new-main https://github.com/vanshika1510/Face-Analysis-Algorithm.git
   cd Face-Analysis-Algorithm
   ```

2. **Create a Virtual Environment (Optional but Recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## Dataset  
The model is trained on a dataset containing labeled facial images for 
Acne
Blackheads
Dark Spots
Dry Skin
Eye bags
Normal Skin
Oily Skin
Pores
Skin Redness
Wrinkles. 
## Model Training  

To train the model, run:  
```bash
```
model is trained and compared accuracy using different models
EfficientNetB0
EfficientNetV2B0
MobileNetV2
MobileNetV3Small
ResNet50
VGG16
Roboflow (included separately with highest accuracy)

## Performance Visualization  

The repository includes scripts to visualize model accuracy and loss.  
- **Accuracy Pie Chart & Table**:  
  ```bash
  cd Models
  python compare.py
  ```

## Results  

The trained models achieve **high accuracy on face classification tasks**. Model comparisons are saved in `Json Files/` and visualized in `plots/`.  


