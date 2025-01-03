# 🐱🐶 Cats vs Dogs Image Classification

A deep learning project that classifies images of cats and dogs using a Convolutional Neural Network (CNN). This project demonstrates how to preprocess image datasets, build and train CNN models, and evaluate their performance.

---

## 🚀 Project Overview
In this project, we utilize TensorFlow and Keras to build a CNN-based image classifier to distinguish between cats and dogs. The dataset contains labeled images of cats and dogs, and the goal is to create a model that can accurately classify unseen images into the respective categories.

---

## 📂 Dataset
- **Source**: [Kaggle - Dogs vs Cats Dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data)

---

## 🛠️ Features
- **Data Preprocessing**:  
- Resizing and normalizing images.  
- Splitting the dataset into training, validation, and testing sets.  

- **Model Architecture**:  
- **Convolutional Layers**: Extract spatial features.  
- **MaxPooling Layers**: Downsample feature maps.  
- **Fully Connected Layers**: Perform classification.  

- **Evaluation Metrics**:  
- Accuracy  
- Loss  

---

## 🔧 Setup and Installation

### **Prerequisites**
1. Python (>= 3.8)  
2. TensorFlow/Keras  
3. NumPy, Matplotlib, OpenCV  

### **Steps**
1. Clone the repository:  
 ```bash
 git clone https://github.com/yourusername/cats-vs-dogs-classification.git
 cd cats-vs-dogs-classification
```

2. Install dependencies:
```bash
Copy code
pip install -r requirements.txt
```
3. Download the dataset:
Place the dataset in the kaggle/datasets/dogs_vs_cats/ directory.

## ⚙️ How It Works

1. **Data Preprocessing**:  
   Images are resized to `150x150` pixels, normalized, and augmented for training.

2. **Model Training**:  
   - A CNN model is trained using the preprocessed data.  
   - The model learns features like edges, textures, and shapes to differentiate between cats and dogs.

3. **Evaluation**:  
   - The model is evaluated on a test set to determine its accuracy and robustness.

---

## 📊 Results

| Metric                | Value  |
|-----------------------|--------|
| **Training Accuracy** | 80%    |
| **Validation Accuracy** | 83%  |
| **Test Accuracy**     | 86%    |

---

## 💻 Usage

Run the following command to classify images:

```bash
python classify.py --image_path path_to_image.jpg
```
## 🧠 Model Architecture

### Conv2D Layers
- Filter sizes: `(3x3)`  
- Activation: `ReLU`  
- Pooling: `MaxPooling2D`  

### Fully Connected Layers
- Dense layers: `[512, 1]`  
- Activation: `Sigmoid` (for binary classification)  

---

## 🌟 Project Highlights

- Utilizes **Transfer Learning** for better accuracy.  
- Demonstrates best practices for data augmentation and model evaluation.  
- Built with modular and reusable code.  

---

## 🤝 Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

---


