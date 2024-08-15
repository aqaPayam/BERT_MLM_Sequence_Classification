
# Masked Language Modeling and Sequence Classification with BERT

This repository contains the implementation of Masked Language Modeling (MLM) and Sequence Classification tasks using a smaller version of the BERT model in PyTorch. The project is part of HW3 for the Deep Learning course instructed by Dr. Soleymani.

## Project Overview

This project leverages a smaller, customized version of BERT to accomplish two primary objectives:
1. **Masked Language Modeling (MLM)**: Predicting masked tokens in a sequence, which helps the model understand context and language structure.
2. **Sequence Classification**: Classifying movie reviews from the Rotten Tomatoes dataset as positive or negative.

The combination of these tasks demonstrates BERT's versatility in handling both generative and discriminative tasks.

### Key Features:

- **Custom BERT Model**: A scaled-down version of BERT that allows efficient training on smaller datasets and resources.
- **Dual Objectives**: The notebook implements both MLM for unsupervised pre-training and sequence classification for a downstream task.
- **Efficient Training**: The project is designed to be computationally feasible, making it suitable for educational purposes and experimentation on smaller datasets.

## Dataset

The dataset used is the **Rotten Tomatoes movie review dataset**, available from the HuggingFace Datasets library. This dataset includes movie reviews labeled as positive or negative, which are used for the sequence classification task.

### Dataset Details:

- **Train Set**: 8,530 reviews
- **Validation Set**: 1,066 reviews
- **Test Set**: 1,066 reviews

The dataset is loaded directly in the notebook using:
```python
from datasets import load_dataset
dataset = load_dataset('rotten_tomatoes')
```

## Model Architecture

The model is based on a smaller variant of BERT (Bidirectional Encoder Representations from Transformers) with the following key components:

- **Embedding Layer**: Converts input tokens into dense vectors.
- **Transformer Encoder Layers**: Multiple layers of self-attention and feedforward networks.
- **MLM Head**: Predicts masked tokens during pre-training.
- **Classification Head**: Maps the final hidden states to output classes (positive/negative) for sequence classification.

### Key Components:

1. **Self-Attention Mechanism**: Allows the model to focus on different parts of the input sequence, capturing long-range dependencies.
2. **Feedforward Neural Networks**: Processes the attended information and transforms it into higher-level features.
3. **Masked Language Modeling**: A form of unsupervised learning where some tokens in the input are masked and predicted by the model.
4. **Sequence Classification**: The model is fine-tuned to classify entire sequences (e.g., determining if a review is positive or negative).

## Installation and Setup

To get started with this project:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AqaPayam/BERT_MLM_Sequence_Classification.git
    ```

2. **Install Dependencies**:
    Install Python and the following libraries:
    - PyTorch
    - Transformers (HuggingFace)
    - Datasets (HuggingFace)
    - Numpy
    - Pandas
    - Matplotlib

    Install the necessary dependencies:
    ```bash
    pip install torch transformers datasets numpy pandas matplotlib
    ```

3. **Run the Jupyter Notebook**:
    Open the notebook and execute the cells in order:
    ```bash
    jupyter notebook BERT_MLM_Sequence_Classification.ipynb
    ```

## Running the Model

### 1. Masked Language Modeling (MLM)

MLM is a self-supervised pre-training task where certain tokens in a sentence are masked and the model is trained to predict them. This helps the model learn contextual word representations.

### 2. Sequence Classification

After pre-training, the model is fine-tuned on the sequence classification task where it predicts whether a given movie review is positive or negative.

- **Input**: Textual data (movie reviews).
- **Training Loop**: The notebook includes a training loop that updates model weights based on cross-entropy loss.
- **Evaluation**: The model's performance is evaluated on the test set using accuracy and other relevant metrics.

## Example Usage

The trained BERT model can be used for various NLP tasks beyond this notebook, such as:

- **Sentiment Analysis**: Classifying texts based on sentiment.
- **Language Understanding**: Leveraging the model's understanding of language to perform other NLP tasks like question answering, named entity recognition, etc.

### Sample Code for Inference:
```python
text = "The movie was fantastic and thrilling!"
input_ids = tokenizer.encode(text, return_tensors='pt')
outputs = model(input_ids)
predictions = torch.argmax(outputs.logits, dim=-1)
print("Prediction:", "Positive" if predictions == 1 else "Negative")
```

## Customization

The notebook can be adapted for other datasets and tasks:
- **Change Dataset**: Replace the Rotten Tomatoes dataset with any other text dataset available on HuggingFace or locally.
- **Model Variants**: Use other pre-trained models like DistilBERT, RoBERTa, etc., available through the HuggingFace library.
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and other training parameters to optimize performance.

## Visualization

The notebook includes visualizations to monitor the training process:
- **Loss Curves**: Plotting training loss over epochs to observe convergence.
- **Accuracy Curves**: Visualizing model accuracy during training and validation.

## Conclusion

This project demonstrates the dual capabilities of BERT in both generative (MLM) and discriminative (sequence classification) tasks. It serves as a practical example of how BERT can be fine-tuned for specific NLP tasks, making it a versatile tool in the field of natural language processing.

## Acknowledgments

This project is part of a deep learning course by Dr. Soleymani.
