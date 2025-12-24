# NYC Taxi Deep Learning Pipeline

End-to-end deep learning pipeline for analyzing NYC Yellow Taxi trip data, featuring comprehensive data exploration and neural network regression models.

## 📊 Project Overview

This project analyzes 2M+ NYC Yellow Taxi trip records from 2020, implementing a complete machine learning workflow from exploratory data analysis to building production-ready regression models for trip duration prediction.

## 🚀 Quick Start

**Recommended:** Open in [Google Colab](https://colab.research.google.com/) - all dependencies pre-installed!

**Local Setup:** Requires Python 3.10

## 🚀 Pipeline Components

### Phase 1: Exploratory Data Analysis & Data Preprocessing
**Notebook:** `01_exploratory_analysis.ipynb`

- Processed 2M+ NYC taxi trip records (January-May 2020)
- Comprehensive data cleaning: missing values, outliers, type conversions
- Feature engineering and correlation analysis
- TensorFlow Data Validation (TFDV) for automated data quality checks
- COVID-19 impact analysis on taxi industry patterns (March 2020)
- Statistical analysis and visualization

**Key Insights:**
- Identified key features affecting trip duration
- Discovered anomalies in data during pandemic onset
- Created cleaned dataset ready for modeling

### Phase 2: Neural Network Regression Models
**Notebook:** `02_regression_models.ipynb`

- Integrated NYC weather data for enhanced predictions
- Built and compared three architectures:
  - **Multi-Layer Perceptron (MLP)**
  - **Linear Regression** (Keras Sequential, no hidden layers)
  - **Deep Neural Network** (DNN with 2+ hidden layers)
- Optimizer comparison: SGD, Adam, RMSProp
- Loss functions: MSE and MAE
- 80/20 train-validation split with 100 epochs
- TensorBoard visualization for training monitoring

**Model Performance:**
- Compared multiple optimizers and learning rates
- Selected best model based on validation loss
- Achieved production-ready prediction accuracy

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras** - Deep learning framework
- **Pandas & NumPy** - Data manipulation
- **TensorFlow Data Validation** - Automated data quality
- **Apache Beam** - Data processing
- **Matplotlib & Seaborn** - Visualization
- **TensorBoard** - Training monitoring

## 📈 Key Results

- ✅ Processed and cleaned 2M+ taxi trip records
- ✅ Automated data validation pipeline
- ✅ Built multiple neural network architectures
- ✅ Compared performance across optimizers
- ✅ Integrated external weather data sources
- ✅ Production-ready regression models

## 💻 Installation & Usage
```bash
# Clone repository
git clone https://github.com/thepalakshah/nyc-taxi-deep-learning.git
cd nyc-taxi-deep-learning

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook
# 1. Open 01_exploratory_analysis.ipynb
# 2. Open 02_regression_models.ipynb
```

## 📁 Repository Structure
```
├── 01_exploratory_analysis.ipynb     # Data exploration & preprocessing
├── 02_regression_models.ipynb        # Neural network regression
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Exclude large files
└── README.md                         # Documentation
```

## 📊 Dataset

**Source:** [NYC TLC Trip Record Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Time Period:** January - May 2020
- **Records:** 2M+ taxi trips
- **Weather Data:** Integrated from Meteostat

*Note: Large datasets not included due to size. Download from source.*

## 🎯 Use Cases

- Trip duration prediction for ride-sharing services
- Demand forecasting for transportation planning
- Real-time prediction systems
- Data pipeline template for time-series tabular data

## 👤 Author

**Palak Shah** - Data Engineer & ML Practitioner

🔗 [LinkedIn](https://linkedin.com/in/thepalakshah) | 🌐 [Portfolio](https://palakshahportfolio.netlify.app/) | 💻 [GitHub](https://github.com/thepalakshah)

# Compatible with Python 3.10
tensorflow>=2.15.0,<2.17.0
pandas>=1.5.0,<2.3.0
numpy>=1.23.0,<1.27.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
fastparquet>=2023.0.0
pyarrow>=14.0.0

# Note: TFDV installation is complex, run separately:
# pip install tensorflow-data-validation

## 💻 Installation & Usage

### Option 1: Google Colab (Recommended)

1. Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thepalakshah/nyc-taxi-deep-learning/blob/main/01_exploratory_analysis.ipynb)
2. All dependencies are pre-installed
3. Free GPU access included
4. Just run the cells!

### Option 2: Local Installation

**Requirements:**
- Python 3.10 (NOT 3.11 or 3.13 - TensorFlow compatibility)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

**Note:** TensorFlow and PyTorch require Python 3.10. If you have Python 3.11+, use Google Colab instead.

