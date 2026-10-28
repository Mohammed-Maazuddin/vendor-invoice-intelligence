# 📦 Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Anomaly Flagging

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning intelligence platform designed for supply chain and procurement teams. It automates freight cost forecasting and flags suspicious or abnormal vendor invoices for manual approval, mitigating financial leakage and streamlining accounts payable workflows.

---

## 🌟 Key Features

### 1. 🚚 Freight Cost Prediction (Regression)
- Accurately forecasts estimated freight / shipping costs based on invoice dollar value and shipment quantities.
- Evaluated across multiple regression algorithms (**Linear Regression**, **Decision Trees**, **Random Forest Regressor**), selecting the model with the lowest Mean Absolute Error (MAE).
- Built-in inference safeguards against negative predictions.

### 2. 🚨 Invoice Risk & Manual Approval Flagging (Classification)
- Evaluates incoming purchase orders and invoices to determine if manual audit/approval is required.
- Detects discrepancies between invoice amounts and item totals, as well as irregular receiving delays.
- Powered by a tuned **Random Forest Classifier** with feature scaling for robust classification.

### 3. 🖥️ Interactive Streamlit Web Application
- **Executive UI**: Clean, intuitive interface allowing analysts to switch between Freight Prediction and Invoice Risk Assessment.
- **Instant Decisioning**: Real-time evaluation of risk badges (✅ Safe for Auto-Approval vs 🚨 Requires Manual Approval).
- **Interactive Visual Feedback**: Metric indicators and warning flags when anomalies occur.

---

## 📁 Repository Structure

```text
├── app.py                             # Streamlit web portal
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Excluded temporary, cache, and large DB files
├── .gitattributes                     # Line endings and binary file declarations
├── LICENSE                            # MIT License
├── README.md                          # Project documentation
│
├── data/                              # Database directory (inventory.db excluded from git)
│   └── .gitkeep
│
├── models/                            # Trained inference models & scalers
│   ├── predict_freight_model.pkl      # Best regression model for freight cost
│   ├── predict_flag_invoice.pkl       # Random Forest classifier for invoice flagging
│   └── scaler.pkl                     # Standard scaler for input feature normalization
│
├── inference/                         # Production inference modules
│   ├── predict_freight.py             # Inference pipeline for freight cost prediction
│   └── predict_invoice_flag.py        # Inference pipeline for invoice risk detection
│
├── freight_cost_prediction/           # Training pipeline for freight regression
│   ├── data_preprocessing.py          # Data ingestion and train/test split
│   ├── model_evaluation.py            # Multi-model benchmarking (LR, DT, RF)
│   └── train.py                       # Retraining workflow and model serialization
│
├── invoice_flagging/                  # Training pipeline for invoice classification
│   ├── data_preprocessing.py          # SQL feature extraction and risk labeling
│   ├── model_evaluation.py            # Hyperparameter tuning and metric evaluation
│   └── train.py                       # Classifier training script
│
├── Notebooks/                         # Exploratory Data Analysis & experiments
│   ├── Predicting Freight Cost.ipynb
│   └── Invoice Flagging.ipynb
│
└── gpt_test/                          # Model experimentation & cross-validation notebooks
    ├── compare_freight_models.ipynb
    ├── cross_validate_freight.ipynb
    └── evaluate_freight.ipynb
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/Mohammed-Maazuddin/vendor-invoice-intelligence.git
cd vendor-invoice-intelligence
```

### 3. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application
The repository comes pre-packaged with trained models under `models/`, allowing you to run the web application immediately:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Model Training & Pipeline Details

### Freight Cost Prediction
1. **Data Extraction**: Pulls historical vendor shipments and invoice dollars from SQLite inventory store.
2. **Benchmarking**: Tests Linear Regression, Decision Tree, and Random Forest models using 80/20 train-test splits.
3. **Selection**: Automatically serializes the model with the minimal Mean Absolute Error (MAE) to `models/predict_freight_model.pkl`.
4. **Execution**:
   ```bash
   python freight_cost_prediction/train.py
   ```

### Invoice Risk Flagging
1. **Feature Engineering**: Calculates variance between billed dollars and received items, plus average receiving delay in days.
2. **Preprocessing**: Normalizes features using `StandardScaler`.
3. **Training & Tuning**: Trains a `RandomForestClassifier` with GridSearchCV parameter tuning.
4. **Execution**:
   ```bash
   python invoice_flagging/train.py
   ```

---

## 🛠️ Tech Stack

- **Application & UI**: [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [Joblib](https://joblib.readthedocs.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SQLite3](https://docs.python.org/3/library/sqlite3.html)
- **Prototyping**: Jupyter Notebooks

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mohammed Maazuddin**  
- GitHub: [@Mohammed-Maazuddin](https://github.com/Mohammed-Maazuddin)

