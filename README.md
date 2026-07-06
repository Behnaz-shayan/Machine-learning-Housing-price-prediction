# Housing Price Prediction using Machine Learning

A comprehensive machine learning analysis predicting house prices in King County, Washington. This project demonstrates the complete ML workflow including exploratory data analysis, feature engineering, model training, hyperparameter tuning, and comparative evaluation of four different algorithms (Decision Tree, Random Forest, Linear Regression, and SVM).

## 📊 Project Overview

**Goal:** Build predictive models to estimate house prices based on property features using multiple ML algorithms and identify the optimal approach.

**Dataset:** King County House Sales (21,613 houses, 21 features)  
**Time Period:** May 2014 - May 2015  
**Location:** King County, Washington (including Seattle)  
**Data Completeness:** 100% (no missing values)

## 🎯 Final Results - Model Comparison

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Random Forest (Tuned)** | **0.6925** | **$215,599** | **$127,275** |
| Linear Regression | 0.6522 | $229,309 | $143,680 |
| Decision Tree (Tuned) | 0.6498 | $230,103 | $139,440 |
| SVM (Tuned) | 0.5998 | $245,963 | $139,726 |

**🏆 Best Model:** Random Forest with optimized hyperparameters
- **Hyperparameters:** n_estimators=200, max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=2
- **Performance:** Explains 69.25% of price variance with average prediction error of $127,275
- **Improvement:** 4.3% better than Decision Tree (0.6925 vs 0.6498)

## 📈 Top Predictive Features (Random Forest)

1. **Grade (36.2%)** - Overall quality and construction standard ⭐ Strongest predictor
2. **Square Footage Living (29.0%)** - Living area size ⭐ Second strongest
3. **Year Built (11.1%)** - Age of the property
4. **Square Footage Lot (6.4%)** - Land size
5. **Square Footage Above Ground (3.9%)** - Above-ground area

**Key Insight:** Top 2 features account for 65% of model's decision-making power

## 📁 Project Structure

```
housing-price-prediction-ML/
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
├── Housing_Price_Prediction_Detailed_Report.docx      # Comprehensive technical report (0.79 MB)
├── data/
│   ├── housing_data.csv                              # Dataset (21,613 records)
│   └── DATA_SOURCE.md                                # Data source & feature documentation
└── notebooks/
    └── Housing_Price_Analysis.ipynb                  # Complete interactive analysis notebook
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd housing-price-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Analysis

#### Option A: Google Colab (Quickest Start) ☁️

1. **Open notebook directly:**
   - Go to [Google Colab](https://colab.research.google.com)
   - Click "File" → "Open notebook" → "GitHub"
   - Paste this repo URL (after pushing to GitHub)
   - Or download `Housing_Price_Analysis.ipynb` and upload it to Colab

2. **Upload the data file:**
   - Run the notebook cells
   - When prompted, upload `housing_data.csv` directly to Colab
   - Or uncomment the file upload cell and use it
   ```python
   from google.colab import files
   files.upload()  # Select housing_data.csv
   ```

3. **Run the notebook:**
   - Execute all cells (Shift+Enter or "Run all")
   - The notebook auto-detects Colab and uses the local CSV path
   - Done! ✨

#### Option B: Local Jupyter

1. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   Navigate to `notebooks/Housing_Price_Analysis.ipynb` and run all cells

**Note:** The notebook automatically detects the environment (Colab vs Local) and handles file paths accordingly!

## 📓 Notebook Structure

The analysis is organized in 4 main steps:

### Step 1: Exploratory Data Analysis (EDA)
Comprehensive data exploration with 8 visualizations and statistical tests:
- Dataset overview and basic statistics (21,613 samples, 21 features)
- Missing values analysis (100% complete dataset - no missing values!)
- Descriptive statistics with normality tests (Shapiro-Wilk)
- Target variable (price) distribution analysis
- Feature distributions for all key variables
- Correlation analysis with Pearson correlation significance tests
- Outlier detection using IQR method (documented but retained)
- Data quality assessment

### Step 2: Data Preprocessing & Feature Selection
- Dropped non-predictive features: id, date, zipcode, lat, long, sqft_living15, sqft_lot15
- Train-test split: 80-20 (17,290 training, 4,323 testing samples)
- Final dataset: 13 features
- Data normalization for Linear Regression & SVM using StandardScaler

### Step 3: Model Development & Hyperparameter Tuning

**Decision Tree Regressor:**
- Baseline: R² = 0.4598
- Tuned: R² = 0.6498 (+18.9% improvement)
- Best params: max_depth=8, min_samples_leaf=5
- Cross-validation: 5-fold KFold

**Random Forest Regressor:**
- Baseline: R² = 0.6849
- Tuned: R² = 0.6925 (+1.1% improvement)
- Best params: n_estimators=200, max_depth=None, max_features='sqrt'
- Cross-validation: 5-fold KFold

**Linear Regression:**
- R² = 0.6522
- RMSE = $229,309
- Baseline statistical method for comparison

**Support Vector Machine (SVM):**
- R² = 0.5998
- Best params: C=100, epsilon=0.1, kernel='rbf'
- Kernel-based method for non-linear patterns

### Step 4: Results & Comparative Analysis
- Diagnostic plots for all models (residuals, distributions, actual vs predicted)
- Feature importance ranking comparison
- Model performance metrics (RMSE, MAE, R²)
- Statistical validation with cross-validation scores

## 🔍 Key Findings

1. **Ensemble methods significantly outperform single models**
   - Random Forest (R² = 0.6925) beats Decision Tree by 4.3%
   - Ensemble averaging reduces overfitting and improves robustness

2. **Grade & Living Area dominate price predictions**
   - Grade (36.2%) and Sqft Living (29.0%) account for 65% of importance
   - Quality and size are the primary value drivers in real estate

3. **Model achieves useful prediction accuracy**
   - R² = 0.6925 explains 69.25% of price variance
   - MAE = $127,275 on median price of $450,000 = ~28% error band
   - Suitable for initial valuations and market screening

4. **Non-linear relationships are crucial**
   - Tree-based models outperform linear regression by 4%
   - Market exhibits threshold effects and feature interactions

5. **Data quality is excellent**
   - 100% completeness (no missing values)
   - 21,613 samples provide robust training data
   - Outliers documented but retained for integrity

6. **Statistical validation confirms generalization**
   - 5-fold cross-validation shows consistent performance
   - Hyperparameter tuning improves Decision Tree significantly (+18.9%)
   - Random Forest improvement modest (+1.1%), indicating good baseline


## 🛠 Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning models and evaluation
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **SciPy** - Statistical tests
- **Jupyter** - Interactive notebook environment

## 📝 Notes

- The analysis includes data quality checks and outlier documentation (not removed)
- Cross-validation is used throughout to ensure model generalization
- Diagnostic plots help assess model assumptions and prediction quality
- Feature importance rankings provide interpretability

## 🎓 Educational Use

This project demonstrates:
- Complete ML workflow 
- Importance of EDA before modeling
- Hyperparameter tuning best practices
- Model comparison and selection criteria
- Diagnostic visualization for model assessment
- Statistical testing in ML contexts


## 📚 References

- **Dataset Source:** [Kaggle King County House Sales](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- **Scikit-learn Documentation:** [Decision Trees](https://scikit-learn.org/stable/modules/tree.html) & [Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- **Data Source Details:** See [data/DATA_SOURCE.md](data/DATA_SOURCE.md)

---

**Last Updated:** 2026  
**Author:** Behnaz Korzebor
