# Academic Machine Learning Pipeline: Bug Severity Classification

This project presents an academic, end-to-end Machine Learning pipeline for classifying bug severity. It is structured entirely in a Jupyter Notebook to allow for easy grading, visualization, and step-by-step evaluation.

## 📁 Project Structure
- `main_ml_pipeline.ipynb`: The core academic notebook containing all data preprocessing, EDA, feature engineering, and model training.
- `data/`: Contains the `bug_dataset_50k.csv` dataset.
- `models/`: Destination folder where trained models can be saved (optional).
- `archive/`: Previous experimental and SaaS application files.

## 🚀 How to Run

### 1. Install Requirements
Open your terminal or command prompt in this directory and install the dependencies:
```bash
pip install -r requirements.txt
```

*(Note: The `requirements.txt` includes standard data science packages like `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `jupyter`)*

### 2. Launch Jupyter Notebook
Start the Jupyter Notebook server:
```bash
jupyter notebook
```

### 3. Execute the Pipeline
- Open `main_ml_pipeline.ipynb` in your browser.
- Run the cells sequentially (`Shift + Enter` or `Run All`).
- The notebook will load the data from the `data/` folder, generate visualizations, and train the Naive Bayes and Random Forest models, outputting their respective confusion matrices at the end.
