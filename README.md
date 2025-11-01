# 🏠 House Price Prediction System

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pipeline](https://img.shields.io/badge/Pipeline-ZenML-green.svg)](https://zenml.io/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue.svg)](https://mlflow.org/)

An end-to-end machine learning pipeline for predicting house prices using the Ames Housing dataset. Built with ZenML for pipeline orchestration and MLflow for experiment tracking.

## 🎯 Project Overview

This project implements a complete ML pipeline that:
- Ingests and processes the Ames Housing dataset
- Performs comprehensive feature engineering and data cleaning
- Trains multiple regression models
- Tracks experiments with MLflow
- Provides deployment capabilities

## 🔧 Tech Stack

- **Pipeline Orchestration**: ZenML
- **Experiment Tracking**: MLflow 
- **ML Framework**: Scikit-learn
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Configuration**: YAML

## 📊 Key Features

- **Modular Design**: Each ML step is isolated and reusable
- **Strategy Pattern**: Flexible model evaluation and building
- **Data Quality**: Comprehensive outlier detection and missing value handling
- **Feature Engineering**: Advanced feature transformations
- **Experiment Tracking**: Complete model versioning and metrics tracking

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Saksham932007/prices-predictor-system.git
cd prices-predictor-system

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# Train the model
python run_pipeline.py

# Make predictions
python sample_predict.py
```

## 📁 Project Structure

```
prices-predictor-system/
├── analysis/                 # Exploratory Data Analysis
│   ├── EDA.ipynb            # Jupyter notebook with comprehensive EDA
│   └── analyze_src/         # Analysis modules
├── src/                     # Core ML modules
│   ├── ingest_data.py       # Data ingestion
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluator.py
├── steps/                   # ZenML pipeline steps
├── pipelines/               # Pipeline definitions
├── config.yaml             # Configuration file
└── requirements.txt        # Dependencies
```

## 🔬 Model Performance

Current model achieves:
- **Algorithm**: Linear Regression with preprocessing pipeline
- **Features**: 79 engineered features from original dataset
- **Preprocessing**: Missing value imputation, one-hot encoding, outlier detection

## 🎨 Key Design Patterns

- **Strategy Pattern**: Flexible model evaluation strategies
- **Factory Pattern**: Dynamic model instantiation
- **Template Pattern**: Consistent pipeline steps

## 📈 MLflow Integration

Access experiment tracking:
```bash
mlflow ui
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Ames Housing Dataset
- ZenML Community
- MLflow Project