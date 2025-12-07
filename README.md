# Flight Delay Prediction Project

## Setup

1. Install dependencies:
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly joblib shap matplotlib
```

## Execution Order

### 1. Create Sampled Dataset
```bash
python create_sampled_dataset.py
```
Creates `data/sampled_flights_2018_2022.csv` from raw yearly CSV files.

### 2. Generate Train/Test Datasets
```bash
python generate_datasets.py
```
Generates balanced training set and real-distribution test set.

### 3. Exploratory Data Analysis
Open and run `eda.ipynb` in Jupyter Notebook. This can be skipped if the intention is to observe the resfrom the streamlit app, which is cleaner and nicer.

### 4. Train Model
```bash
python train_model.py
```
Trains XGBoost model and saves to `models/` directory.

### 5. Run Streamlit App
```bash
streamlit run app.py
```
Opens the interactive dashboard in your browser.

