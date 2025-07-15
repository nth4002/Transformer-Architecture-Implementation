# Transformer from Scratch in PyTorch

This repository contains a Jupyter Notebook that implements the Transformer architecture from the ground up using PyTorch. It is designed as a learning tool to understand the inner workings of the model introduced in the seminal paper, ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

The entire implementation, from data loading and preprocessing to the final model training and inference, is contained within a single, well-commented notebook.

## Model Architecture

The implementation follows the original Transformer architecture.

![Transformer Architecture](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)
> *The Transformer Model Architecture (from Vaswani et al., 2017)*

## Key Features

-   **From-Scratch Implementation:** Every core component is built from basic PyTorch modules to expose the underlying logic.
-   **Detailed Comments:** The code is commented to explain each part of the architecture, including tensor shapes and the purpose of each operation.
-   **Core Components Covered:**
    -   Positional Encoding
    -   Multi-Head Self-Attention
    -   Masking (Padding and Look-ahead)
    -   Encoder and Decoder Layers
    -   Final Linear and Softmax Layers
-   **End-to-End Example:** The notebook covers data loading (Multi30k dataset), vocabulary creation, model training, and inference for a German-to-English translation task.

## Getting Started

Follow these steps to run the notebook on your own machine.

### 1. Prerequisites

-   Python 3.11.13
-   Jupyter Notebook or JupyterLab

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    You can create a `requirements.txt` file with the content below and run `pip install -r requirements.txt`.
    
    **requirements.txt:**
    ```
    altair==5.5.0
    GPUtil==1.4.0
    torch==2.1.0
    torchtext==0.16.0
    torchdata==0.7.0
    numpy==1.26.4
    spacy==3.7.2
    portalocker==3.2.0
    ```
    
    Then install using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download SpaCy language models:**
    These are needed for tokenization.
    ```bash
    python -m spacy download de_core_news_sm
    python -m spacy download en_core_web_sm
    ```

### 3. Running the Notebook

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
    or
    ```bash
    jupyter lab
    ```

2.  Open the main `.ipynb` notebook file and run the cells sequentially. The notebook will automatically download the Multi30k dataset on the first run.

## Notebook Structure

The notebook is organized into the following sections:

1.  **Setup & Imports:** Importing necessary libraries and setting up the environment.
2.  **Data Loading and Preprocessing:** Defining tokenizers, building vocabularies, and creating data iterators for the Multi30k dataset.
3.  **Model Components:** Building each part of the Transformer step-by-step (Multi-Head Attention, Positional Encoding, etc.).
4.  **Assembling the Transformer:** Combining the components into the final Encoder-Decoder model.
5.  **Training:** Defining the training loop, loss function, and optimizer to train the model.
6.  **Inference:** A function to translate a new German sentence into English using the trained model.

## Acknowledgments

-   This implementation is heavily inspired by the original paper: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). [Attention is all you need](https://arxiv.org/abs/1706.03762).
-   Guidance was also taken from tutorials like Harvard NLP's ["The Annotated Transformer"](http://nlp.seas.harvard.edu/2018/04/03/attention.html).

