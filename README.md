# AI Text Summarization App

## Overview
This project focuses on building and evaluating text summarizations models using the CNN/Daily News dataset from Kaggle. The dataset contains around 287,000 news articles in the training samples, along with summarizations written by editors for the models to train from.


## Project Structure
```plaintext
AITextSummarizer/
├── data/                # Dataset downloaded from Kaggle (ONLY generated after download dataset from src/download_data.py)
├── Notebooks/           # Project Notebooks
├── saved_models/        # Saved models after training (ONLY generated after training the models)
├── src/                 # Python scripts
├── .gitignore
├── requirements.txt
└── README.md
```


## Setup Instructions

**1. Clone the Repository**
```
git clone https://github.com/BikinGhimire/AITextSummarizer.git
cd AITextSummarizer
```

**2. Create and Activate Virtual Environment**
```
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```
pip install -r requirements.txt
```

**4. Kaggle API Setup**
- Download your Kaggle API token (kaggle.json) from your Kaggle account settings.
- Place the kaggle.json file in the project root directory.

**5. Download Dataset**
```
python src/download_data.py
```


## Training the models
Simply run the Notebooks/Main.ipynb file for all three models to train.


## Running the app
Enter the project directory and run the streamlit app to load the app and start using text summarization.
```
cd AITextSummarizer
streamlit run app.py
```

## License
This project is unlicensed.

---
© 2024 Bikin Ghimire. All rights reserved.
