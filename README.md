# Chess Board Detection and Best Move Prediction using Machine Learning

## Overview

This project implements a simpler machine learning-based system for identifying chess pieces from an image of a chessboard, generating the corresponding **FEN (Forsyth-Edwards Notation)**, and determining the best move using a chess engine. The primary focus lies in recognizing individual pieces on the board, constructing a board array for the Python Chess library, and leveraging it to compute the best move.

## Features

- **Chess Piece Detection**: Uses a trained machine learning model to identify individual pieces (both white and black) on the board.
- **FEN Generation**: Converts the identified board state into a valid FEN string.
- **Best Move Prediction**: Integrates with a chess engine to compute the best move based on the generated FEN.
- **Web Application**: A user-friendly web interface where users can upload chessboard images, specify whose turn it is (white or black), and receive the best move.
- **Customizable Dataset**: Train the model on other chessboard formats using the provided code and dataset generation tools.

---

## Project Structure

### 1. Folder: **Dataset and Machine Learning**

This folder contains all code and resources related to creating the dataset, training the model, and making predictions:

- **Dataset Creation**: Scripts for generating datasets and CSV headers from chessboard images.
- **Model Training**: Code for training a machine learning model using the provided dataset.
- **Board Prediction**: Scripts to predict the board state using the trained model.
- **Best Move Prediction**: Logic to determine the best move using the predicted board state and a chess engine.

#### Dataset Details

- **Source**: The dataset was created using screenshots of chess board taken from chess.com.
- **Statistics**:
  - 120 board images were used for training.
  - Extracted **3,439 white piece images** and **3,437 black piece images**.
- **Customization**: You can expand the dataset by adding more images of chess pieces and retrain the model using the provided scripts.

### 2. Folder: **Trained Model and Web Application**

This folder contains the web application built using Flask and Python:

- **Web Interface**: Upload an image of the chessboard, specify whose turn it is (white or black), and get the best move.
- **Chess Engine Integration**: Requires the user to specify the path to the chess engine executable.
- **Requirements**: Includes a `requirements.txt` file to simplify dependency installation.

---

## Installation and Setup

### Prerequisites

1. Python 3.8+
2. NumPy
3. Pandas
4. Flask
5. OpenCV
6. Scikit-learn
7. Pillow
8. A compatible chess engine (e.g., [Stockfish](https://stockfishchess.org/))

### Steps to Run the Web App

1. Clone the repository:

   ```bash
   git clone https://github.com/LovejeetM/Chess_ML_Algorithm.git
   cd Chess_ML_Algorithm
   ```

2. Navigate to the `Trained Model Webapp` folder:

   ```bash
   cd "Trained Model Webapp"
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up the chess engine path in the web application:

   - Edit the `ml.py` file (or the appropriate script) to include the path to your chess engine executable.

1) Run the Flask application:

   ```bash
   python app.py
   ```

2) Open the web app in your browser at `http://127.0.0.1:5000`.

Additionally, you can use virtual environment if it does not run on your system and for testing.

---

## Usage

### Web Application

1. **Upload Image**: Upload a chessboard image (e.g., a screenshot from chess.com).
2. **Specify Turn**: Indicate whether it is White's or Black's turn.
3. **Generate Best Move**: The app processes the image, generates the FEN, and displays the best move.

### Training a Custom Model

1. Add new chessboard images to the dataset.
2. Use the scripts in the `Dataset and Machine Learning` folder to generate piece images and prepare a CSV file.
3. Retrain the model using the provided training scripts.
4. Update the model file in the web app folder to use the new model.

---

## Accuracy

- **Current Model Accuracy**: Achieves 100% accuracy on the provided dataset of chess board images.
- **Custom Datasets**: Accuracy may vary based on the quality and diversity of the training dataset.
- Light Weight Model: This codebase uses simple method for classification of pieces board.

---

## Future Work

- **Support for Other Chessboard Styles**: Expand the dataset to include different board formats.
- **Real-Time Detection**: Implement live video feed detection for real-time chess analysis.
- **Enhanced Web Features**: Add more functionality to the web app, such as game history tracking and multi-move prediction.

---

## License

This project is licensed under a custom license. Usage of this code is restricted to non-commercial purposes only. Modifications are allowed, but commercial redistribution is prohibited without permission.

---

## Feel free to Connect

LinkedIn: https\://www\.linkedin.com/in/lovejeet-singh-matharu-975679213/ 

Feel free to ask any questions too.
