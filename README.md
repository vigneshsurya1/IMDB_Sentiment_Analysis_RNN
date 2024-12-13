# IMDB Sentiment Analysis using RNN

## Project Overview
This project focuses on performing **sentiment analysis** on the IMDB reviews dataset. The goal is to classify movie reviews as either **positive** or **negative** using a **Recurrent Neural Network (RNN)**. The project demonstrates the power of deep learning in natural language processing (NLP) tasks.

---

## Dataset
- **Name**: IMDB Reviews Dataset
- **Description**: A collection of 50,000 movie reviews labeled as positive or negative.
- **Usage**: The dataset is split into training and testing sets for model evaluation.

---

## Techniques
- **Text Preprocessing**: Tokenization, padding, and truncation of sequences.
- **Model Architecture**: Simple RNN implemented using TensorFlow and Keras.

---

## Libraries Used
- **TensorFlow**: For building and training the RNN model.
- **Keras**: High-level API for defining the model.
- **Pandas**: For dataset manipulation.
- **scikit-learn**: For metrics like accuracy.
- **NumPy**: For numerical computations.

---

## Results
- **Accuracy Score**: 0.8565
- The model effectively predicts the sentiment of movie reviews with a high degree of accuracy.

---

## How to Run
1. Clone the repository or download the notebook file.
2. Install the required libraries:
   ```bash
   pip install tensorflow pandas scikit-learn numpy
   ```
3. Run the notebook to preprocess the data, train the model, and evaluate its performance.

---

## Future Improvements
- Experiment with more advanced architectures like LSTMs or GRUs.
- Use pre-trained word embeddings such as GloVe or Word2Vec.
- Hyperparameter tuning to improve accuracy.

---

## Acknowledgments
- **Dataset**: IMDB dataset provided by Stanford AI Lab.
- **Libraries**: TensorFlow and Keras for deep learning implementation.

