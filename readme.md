# Neural Network Implementation

This repository contains a neural network implementation for various machine learning tasks.

## Contents

- `neural-network.ipynb`: Jupyter notebook containing the neural network implementation and experiments.
- `train.csv`: Training dataset used for training the neural network.
- `readme.md`: This README file.

## Requirements

To run the code in this repository, you need to have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

You can install the required packages using the following command:

```sh
pip install -r requirements.txt
```

## Usage

Clone the repository:

```sh
git clone https://github.com/yourusername/neural-network.git
cd neural-network/nn
```

Open the Jupyter notebook:

```sh
jupyter notebook neural-network.ipynb
```
This code implements a simple **neural network** with **one hidden layer** using **NumPy**. The network is designed for **classification tasks**, and it updates its parameters using **gradient descent**. Let's break down each function:

---


Run the cells in the notebook to train and evaluate the neural network.

## Dataset

The training dataset `train.csv` is used to train the neural network. Make sure the dataset is in the same directory as the Jupyter notebook.


### **1. `initialize_parameters()`**
- Initializes the **weights** (`W1`, `W2`) and **biases** (`B1`, `B2`) randomly.
- `W1` (10, 784): First layer weights (assuming input size is **784**, like MNIST images).
- `B1` (10, 1): Biases for the first layer.
- `W2` (10, 10): Second layer weights.
- `B2` (10, 1): Biases for the second layer.
- The values are initialized between **-0.5 and 0.5** for randomness.

---

### **2. `ReLU(X)`**
- Implements the **ReLU (Rectified Linear Unit)** activation function.
- If `X > 0`, return `X`, else return `0`.
- Used in the **hidden layer** to introduce non-linearity.

---

### **3. `softmax_calculator(Z)`**
- Implements the **Softmax function**, which converts logits into **probabilities**.
- Formula:  
  \[
  A2 = \frac{e^Z}{\sum e^Z}
  \]
- Used in the **output layer** to classify multiple categories.

---

### **4. `forward_propagation(W1, B1, W2, B2, X)`**
- Computes **Z1** (weighted sum for the first layer).
- Applies **ReLU** activation → **A1**.
- Computes **Z2** (weighted sum for the second layer).
- Applies **Softmax** activation → **A2**.
- **Returns**: `Z1, A1, Z2, A2`.

---

### **5. `one_hot_converter(Y)`**
- Converts **labels (Y) into one-hot encoding**.
- Example: If `Y = [1, 2, 0]`, the one-hot representation is:
  ```
  [[0, 1, 0],
   [0, 0, 1],
   [1, 0, 0]]
  ```
- This is used for proper **backpropagation calculations**.

---

### **6. `backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y)`**
- Computes gradients (**partial derivatives**) of the loss function w.r.t. **weights and biases**.
- **Steps:**
  1. Compute the error in the **output layer** (`dZ2 = A2 - one_hot_Y`).
  2. Compute gradients for `W2` and `B2`:
     \[
     dW2 = \frac{1}{m} dZ2 \cdot A1^T
     \]
     \[
     dB2 = \frac{1}{m} \sum dZ2
     \]
  3. Compute error for **hidden layer** (`dZ1`).
  4. Compute gradients for `W1` and `B1`.
- **Returns**: `dW1, dB1, dW2, dB2`.

---

### **7. `update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate)`**
- Updates the weights and biases using **gradient descent**:
  \[
  W = W - \alpha dW
  \]
  \[
  B = B - \alpha dB
  \]
- **Returns**: Updated `W1, B1, W2, B2`.

---

### **8. `get_predictions(A2)`**
- Returns the index of the highest probability for each sample.
- **Example:**
  ```
  A2 = [[0.1, 0.8, 0.05],
        [0.6, 0.1, 0.8],
        [0.3, 0.1, 0.15]]
  ```
  Output:
  ```
  [1, 0, 1]  (because 0.8, 0.6, and 0.8 are the highest)
  ```
---

### **9. `get_accuracy(predictions, Y)`**
- Compares **predictions** with actual labels **Y** and calculates accuracy.
- Formula:
  \[
  \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}}
  \]

---

### **10. `gradient_descent(X, Y, alpha, iterations)`**
- **Main training loop**:
  1. Initializes **parameters**.
  2. Runs **forward propagation**.
  3. Computes **gradients** via **backpropagation**.
  4. Updates parameters using **gradient descent**.
  5. Every **20 iterations**, it prints the accuracy.
- **Returns**: Trained `W1, B1, W2, B2`.

---

### **Overall Summary**
This code implements a **simple 2-layer neural network** for classification tasks:
1. **Input Layer:** 784 neurons (if working with images like MNIST).
2. **Hidden Layer:** 10 neurons with **ReLU activation**.
3. **Output Layer:** 10 neurons with **Softmax activation**.
4. **Training** uses **Gradient Descent** and **Backpropagation**.


## License

This project is licensed under the MIT License. See the LICENSE file for details.