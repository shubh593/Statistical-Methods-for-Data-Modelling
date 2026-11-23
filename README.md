# Statistical Methods for Data Modelling (SMDM)

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Package Manager](https://img.shields.io/badge/Package_Manager-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![SciPy](https://img.shields.io/badge/SciPy-Statistics-blue.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Econometrics-darkblue.svg)](https://www.statsmodels.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

A rigorous, end-to-end laboratory of statistical computing, econometric diagnostics, parametric & non-parametric hypothesis testing, regression analysis, and classical time-series modeling implemented in Python.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Architecture & Modules](#-project-architecture--modules)
  - [1. Exploratory Data Analysis & Descriptive Statistics](#1-exploratory-data-analysis--descriptive-statistics)
  - [2. Sampling & Central Limit Theorem (CLT)](#2-sampling--central-limit-theorem-clt)
  - [3. Inferential Statistics & Hypothesis Testing (t-Tests)](#3-inferential-statistics--hypothesis-testing-t-tests)
  - [4. Categorical Data Analysis (Chi-Square Test of Independence)](#4-categorical-data-analysis-chi-square-test-of-independence)
  - [5. Bivariate Association & Correlation Analysis](#5-bivariate-association--correlation-analysis)
  - [6. Univariate Linear Regression](#6-univariate-linear-regression)
  - [7. Multivariate Linear Regression & Generalization](#7-multivariate-linear-regression--generalization)
  - [8. Econometric Regression Diagnostics & Gauss-Markov Validation](#8-econometric-regression-diagnostics--gauss-markov-validation)
  - [9. Time-Series Analysis & Classical Seasonal Decomposition](#9-time-series-analysis--classical-seasonal-decomposition)
  - [10. Classical Time-Series Forecasting & Model Comparison](#10-classical-time-series-forecasting--model-comparison)
- [Summary Matrix of Projects](#-summary-matrix-of-projects)
- [Repository Structure](#-repository-structure)
- [Environment Setup & Installation](#-environment-setup--installation)
- [How to Run](#-how-to-run)
- [Statistical Formulations Reference](#-statistical-formulations-reference)
- [Tech Stack](#-tech-stack)
- [Author & License](#-author--license)

---

## 🔬 Overview

This repository demonstrates practical statistical methodologies for data modeling and inference. Each Jupyter Notebook is an independent case study covering core statistical theory, experimental formulation, code implementation, model validation, and diagnostic visualization.

The workflow spans the full statistical modeling lifecycle:
1. **Descriptive & Exploratory Analysis**: Summarizing distributions, identifying skewness, detecting dispersion and outliers.
2. **Sampling Theory & Probability**: Empirical verification of the Central Limit Theorem and Standard Error convergence.
3. **Hypothesis Testing**: Formulating null/alternative hypotheses, conducting parametric ($t$-tests) and non-parametric ($\chi^2$) significance testing.
4. **Relational Analysis**: Distinguishing between linear (Pearson) and monotonic (Spearman) dependencies.
5. **Supervised Regression Modeling**: Fitting univariate and multivariate Ordinary Least Squares (OLS) models.
6. **Model Diagnostics**: Auditing regression assumptions (linearity, homoscedasticity, error normality, absence of multicollinearity).
7. **Temporal Dynamics**: Additive time-series decomposition and comparative multi-horizon forecasting (Moving Average vs. Simple Exponential Smoothing vs. Holt-Winters).

---

## 🗂️ Project Architecture & Modules

### 1. Exploratory Data Analysis & Descriptive Statistics
- **Notebook**: [`EDA_Descriptive Statistics.ipynb`](EDA_Descriptive%20Statistics.ipynb)
- **Dataset**: Fisher's Iris Dataset (`sklearn.datasets.load_iris`)
- **Key Techniques**:
  - Measures of Central Tendency: Sample Mean ($\bar{x}$), Median, and Mode.
  - Measures of Dispersion: Range, Sample Variance ($s^2$), Standard Deviation ($s$), Interquartile Range (IQR).
  - Distribution Geometry: Fisher-Pearson coefficient of skewness ($g_1$) to quantify asymmetry.
  - Missing value audits and tabular summary metrics (`df.describe()`, `df.info()`).
- **Visualizations**: Kernel Density Estimation (KDE) Histogram overlay, Box Plot for quantile boundaries and outlier detection.

---

### 2. Sampling & Central Limit Theorem (CLT)
- **Notebook**: [`Sampling_&_Central_Limit_Theorem.ipynb`](Sampling_&_Central_Limit_Theorem.ipynb)
- **Population**: Asymmetric, heavily right-skewed Exponential distribution ($\beta = 2.0$, $N = 10,000$).
- **Key Techniques**:
  - Repeated random sampling simulation ($k = 1,000$ iterations with sample size $n = 30$).
  - Law of Large Numbers (LLN) verification: convergence of sample means toward population mean $\mu$.
  - Empirical Standard Error vs. Theoretical Standard Error:
    $$SE = \frac{\sigma}{\sqrt{n}}$$
  - Demonstration that the distribution of sample averages approximates a Gaussian normal curve $\mathcal{N}(\mu, \sigma^2/n)$ despite non-normal underlying population data.
- **Visualizations**: Dual-panel histogram contrasting the raw exponential distribution against the bell-shaped sampling distribution of the mean.

---

### 3. Inferential Statistics & Hypothesis Testing (t-Tests)
- **Notebook**: [`Hypothesis_Testing _using _T-Test.ipynb`](Hypothesis_Testing%20_using%20_T-Test.ipynb)
- **Dataset**: UCI Wine Recognition Dataset (`sklearn.datasets.load_wine`)
- **Key Techniques**:
  - **One-Sample $t$-Test**:
    - Null Hypothesis ($H_0$): Mean alcohol content $\mu = 13.0\%$.
    - Alternative Hypothesis ($H_1$): Mean alcohol content $\mu \neq 13.0\%$.
    - Two-tailed evaluation at significance threshold $\alpha = 0.05$.
  - **Two-Sample Independent Welch's $t$-Test**:
    - Null Hypothesis ($H_0$): Mean alcohol content of Cultivar 0 equals Cultivar 1 ($\mu_0 = \mu_1$).
    - Applied Welch-Satterthwaite approximation (`equal_var=False`) to account for heteroscedasticity across groups.
  - Automated hypothesis decision criteria based on $p$-value comparison against $\alpha$.

---

### 4. Categorical Data Analysis (Chi-Square Test of Independence)
- **Notebook**: [`Chi-Square_Test.ipynb`](Chi-Square_Test.ipynb)
- **Dataset**: Fisher's Iris Dataset (`sklearn.datasets.load_iris`)
- **Key Techniques**:
  - Feature Discretization: Binning continuous `sepal length (cm)` into tertile quantiles (`pd.qcut` into 'Small', 'Medium', 'Large').
  - Contingency Table Construction: Cross-tabulating discretized size categories against botanical species classes (`Setosa`, `Versicolor`, `Virginica`).
  - Pearson's $\chi^2$ Test of Independence:
    $$\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
  - Computation of degrees of freedom $(r - 1)(c - 1)$, expected frequencies matrix, and hypothesis evaluation ($p < 0.05 \implies$ rejection of independence).

---

### 5. Bivariate Association & Correlation Analysis
- **Notebook**: [`Correlation_Analysis.ipynb`](Correlation_Analysis.ipynb)
- **Dataset**: Diabetes Progression Dataset (`sklearn.datasets.load_diabetes`)
- **Key Techniques**:
  - **Pearson Correlation ($r$)**: Parametric measurement of linear co-variation between physiological variables (`bmi`, `bp`, `s5`, `target`).
  - **Spearman Rank Correlation ($\rho$)**: Non-parametric measurement of monotonic association based on ranked values, resistant to bivariate outliers.
  - Significance testing: Extracting two-tailed $p$-values for both linear and rank coefficients.
- **Visualizations**: Annotated symmetric Pearson correlation heatmap with diverging `coolwarm` palette bounded between $[-1.0, 1.0]$.

---

### 6. Univariate Linear Regression
- **Notebook**: [`Simple_Linear_Regression.ipynb`](Simple_Linear_Regression.ipynb)
- **Dataset**: Diabetes Progression Dataset (`sklearn.datasets.load_diabetes`)
- **Key Techniques**:
  - Single predictor modeling: Regressing quantitative disease progression ($y$) on Body Mass Index ($X = \text{BMI}$).
  - Train/Test validation split (80/20 train-test ratio, `random_state=42`).
  - Parameter estimation: Ordinary Least Squares derivation of intercept ($\beta_0$) and slope coefficient ($\beta_1$).
  - Goodness-of-Fit evaluation: Coefficient of Determination ($R^2$) computed independently on training and out-of-sample test splits.
- **Visualizations**: 2D scatter plot of observed test points overlaid with the fitted OLS linear regression line.

---

### 7. Multivariate Linear Regression & Generalization
- **Notebook**: [`Multiple_Linear_Regression.ipynb`](Multiple_Linear_Regression.ipynb)
- **Dataset**: Diabetes Progression Dataset (10 baseline physiological predictors: `age`, `sex`, `bmi`, `bp`, `s1` - `s6`)
- **Key Techniques**:
  - Multivariate Ordinary Least Squares formulation:
    $$y = \beta_0 + \sum_{j=1}^{p} \beta_j X_j + \varepsilon$$
  - Multi-feature coefficient vector extraction and sign/magnitude interpretation.
  - Comprehensive model evaluation metrics:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Unadjusted $R^2$
    - **Adjusted $R^2$**: Penalizing model complexity against feature proliferation:
      $$\text{Adj } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1}$$

---

### 8. Econometric Regression Diagnostics & Gauss-Markov Validation
- **Notebook**: [`Regression_Diagnostics.ipynb`](Regression_Diagnostics.ipynb)
- **Framework**: `statsmodels.api.OLS`
- **Key Diagnostic Checks**:
  1. **Linearity & Homoscedasticity**: Residuals vs. Fitted Values plot against zero-baseline; auditing for non-linear residual curvature and heteroscedastic fan shapes.
  2. **Normality of Residuals**: Quantile-Quantile (Q-Q) plot against standard normal quantiles to evaluate heavy tails or skewness in error terms.
  3. **Multicollinearity Screening**: Variance Inflation Factor (VIF) quantification across all predictor variables:
     $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
     Detecting severe collinearity among interrelated serum markers (`s1` through `s6`).
  4. Full econometric summary report: Omnibus tests, Durbin-Watson autocorrelation statistic, Jarque-Bera normality tests, and condition numbers.

---

### 9. Time-Series Analysis & Classical Seasonal Decomposition
- **Notebook**: [`Time-Series_Analysis.ipynb`](Time-Series_Analysis.ipynb)
- **Dataset**: Mauna Loa Atmospheric $CO_2$ Concentration (`statsmodels.datasets.co2`)
- **Key Techniques**:
  - Temporal indexing and frequency regularization: Resampling to Monthly Start (`MS`) frequency.
  - Imputation of missing historical observations using backward fill (`bfill`).
  - Additive Seasonal Decomposition with period $m = 12$:
    $$Y_t = T_t + S_t + R_t$$
  - Disentangling macro-level deterministic trend, recurrent 12-month cyclical variation, and stochastic irregular residuals.
- **Visualizations**: Time-series trajectory line charts, 4-panel decomposition visualizer (Observed, Trend, Seasonal, Residual).

---

### 10. Classical Time-Series Forecasting & Model Comparison
- **Notebook**: [`Basic_Time-Series_Forecasting.ipynb`](Basic_Time-Series_Forecasting.ipynb)
- **Dataset**: Monthly Atmospheric $CO_2$ Time-Series
- **Key Techniques**:
  - Chronological out-of-sample partitioning: Reserving the terminal 12 months as hold-out evaluation period.
  - Models Implemented & Benchmarked:
    1. **Rolling Moving Average (MA)**: Trailing 12-month moving window baseline.
    2. **Simple Exponential Smoothing (SES)**: Level-only smoothing with estimated parameter $\alpha$.
    3. **Holt-Winters Exponential Smoothing**: Triple exponential smoothing with additive trend ($\beta$) and additive seasonality ($\gamma$, period = 12).
  - Out-of-Sample Performance Comparison: Evaluation using Root Mean Squared Error (RMSE):
    $$\text{RMSE} = \sqrt{\frac{1}{h}\sum_{t=1}^{h}(y_{T+t} - \hat{y}_{T+t})^2}$$
- **Visualizations**: Multi-model forecast trajectory chart comparing model projections directly against actual hold-out data.

---

## 📊 Summary Matrix of Projects

| # | Notebook | Domain / Topic | Dataset | Key Statistical Technique | Primary Metrics / Criteria | Primary Visualization |
|---|---|---|---|---|---|---|
| **01** | [`EDA_Descriptive Statistics`](EDA_Descriptive%20Statistics.ipynb) | Exploratory Data Analysis | Iris Dataset | Parametric/non-parametric summaries, skewness | Mean, Median, Mode, Variance, Skew | KDE Histogram, Box Plot |
| **02** | [`Sampling_&_Central_Limit_Theorem`](Sampling_&_Central_Limit_Theorem.ipynb) | Probability & Sampling | Synthetic Exponential ($\beta=2$) | Monte Carlo sampling ($k=1000, n=30$) | Population vs. Sample Mean, $SE = \sigma/\sqrt{n}$ | Dual Population vs. CLT Plot |
| **03** | [`Hypothesis_Testing _using _T-Test`](Hypothesis_Testing%20_using%20_T-Test.ipynb) | Parametric Inference | Wine Recognition | 1-Sample & Welch's 2-Sample $t$-test | $t$-statistic, $p$-value ($\alpha=0.05$) | Terminal Decision Summary |
| **04** | [`Chi-Square_Test`](Chi-Square_Test.ipynb) | Non-Parametric Inference | Iris Dataset | Contingency Table & $\chi^2$ Independence | $\chi^2$ statistic, degrees of freedom, $p$-value | Contingency Matrix |
| **05** | [`Correlation_Analysis`](Correlation_Analysis.ipynb) | Relational Statistics | Diabetes Dataset | Pearson $r$ & Spearman Rank $\rho$ | Correlation coefficients, $p$-values | Diverging Heatmap |
| **06** | [`Simple_Linear_Regression`](Simple_Linear_Regression.ipynb) | Regression Analysis | Diabetes Dataset (BMI) | Univariate OLS Fitting | $\beta_0, \beta_1, R^2_{\text{train}}, R^2_{\text{test}}$ | Fitted Regression Line |
| **07** | [`Multiple_Linear_Regression`](Multiple_Linear_Regression.ipynb) | Multivariate Modeling | Diabetes Dataset (10 features) | Multivariate OLS Fitting | MAE, MSE, RMSE, $R^2$, Adjusted $R^2$ | Coefficient Profiles |
| **08** | [`Regression_Diagnostics`](Regression_Diagnostics.ipynb) | Econometric Validation | Diabetes Dataset | Gauss-Markov & OLS Assumption Audits | VIF, Omnibus, Condition Number | Residuals vs Fitted, Q-Q Plot |
| **09** | [`Time-Series_Analysis`](Time-Series_Analysis.ipynb) | Temporal Modeling | Mauna Loa $CO_2$ | Additive Seasonal Decomposition ($m=12$) | Resampled Trend & Seasonality components | 4-Panel Decomposition Plot |
| **10** | [`Basic_Time-Series_Forecasting`](Basic_Time-Series_Forecasting.ipynb) | Predictive Forecasting | Mauna Loa $CO_2$ | Moving Average, SES, Holt-Winters | Out-of-sample RMSE benchmark | Multi-Forecast vs. Actuals |

---

## 🗂️ Repository Structure

```text
Statistical-Methods-for-Data-Modelling/
├── Basic_Time-Series_Forecasting.ipynb     # Out-of-sample forecasting benchmarks (MA, SES, Holt-Winters)
├── Chi-Square_Test.ipynb                   # Non-parametric contingency analysis of independence
├── Correlation_Analysis.ipynb              # Pearson and Spearman rank correlation with heatmaps
├── EDA_Descriptive Statistics.ipynb        # Exploratory analysis, central tendencies, and distributions
├── Hypothesis_Testing _using _T-Test.ipynb # One-sample and independent Welch's two-sample t-tests
├── Multiple_Linear_Regression.ipynb        # Multivariate OLS regression with MAE/MSE/RMSE/Adj R²
├── Regression_Diagnostics.ipynb            # OLS residual diagnostics, Q-Q plots, and VIF multicollinearity
├── Sampling_&_Central_Limit_Theorem.ipynb  # Monte Carlo verification of the Central Limit Theorem
├── Simple_Linear_Regression.ipynb          # Univariate regression modeling and train/test evaluation
├── Time-Series_Analysis.ipynb              # Additive time-series decomposition (trend, seasonal, noise)
├── src/
│   └── statistical_methods_for_data_modelling/
│       └── __init__.py                     # Package initialization module
├── LICENSE                                 # Apache License 2.0
├── pyproject.toml                          # Project configuration & build specification
├── .gitignore                              # Git exclusion patterns
└── README.md                               # Project documentation & reference guide
```

---

## ⚙️ Environment Setup & Installation

### Prerequisites
- Python `>= 3.10` (Python `3.13` recommended)
- `git`
- `uv` (recommended) or `pip`

### 1. Clone the Repository
```bash
git clone https://github.com/sshinde5930/Statistical-Methods-for-Data-Modelling.git
cd Statistical-Methods-for-Data-Modelling
```

### 2. Environment Creation & Dependency Installation

#### Using `uv` (Fastest):
```bash
# Create and activate virtual environment
uv venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install required scientific packages
uv pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn jupyter ipykernel
```

#### Using standard `pip`:
```bash
# Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn jupyter ipykernel
```

---

## 🚀 How to Run

Launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter lab
# or
jupyter notebook
```
Navigate to any notebook in the root directory and execute cells sequentially (`Shift + Enter`).

---

## 🧮 Statistical Formulations Reference

### Central Limit Theorem
For a random sample $X_1, X_2, \dots, X_n$ from any population with mean $\mu$ and finite variance $\sigma^2$:
$$\bar{X}_n \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty$$

### Welch's Two-Sample t-Statistic
$$\begin{aligned}
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
\end{aligned}$$
with degrees of freedom calculated via the Welch–Satterthwaite equation.

### Pearson vs. Spearman Correlation
- **Pearson $r$**:
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
- **Spearman $\rho$**: Pearson correlation applied directly to ranked variables:
  $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

### Adjusted $R^2$
$$\text{Adj } R^2 = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - k - 1}\right]$$
where $n$ is sample size and $k$ is the number of explanatory predictors.

### Variance Inflation Factor (VIF)
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
where $R_j^2$ is the coefficient of determination from regressing predictor $X_j$ on all remaining explanatory variables. $\text{VIF} > 5 \text{ to } 10$ indicates severe multicollinearity.

### Holt-Winters Additive Method
- **Level Update**:
  $$\ell_t = \alpha(y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})$$
- **Trend Update**:
  $$b_t = \beta(\ell_t - \ell_{t-1}) + (1 - \beta)b_{t-1}$$
- **Seasonal Update**:
  $$s_t = \gamma(y_t - \ell_{t-1} - b_{t-1}) + (1 - \gamma)s_{t-m}$$
- **Forecast**:
  $$\hat{y}_{t+h|t} = \ell_t + h b_t + s_{t+h-m(k+1)}$$

---

## 🛠️ Tech Stack

- **Language**: Python 3.13+
- **Numerical & Data Manipulation**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Scientific Computing & Inference**: [SciPy (scipy.stats)](https://scipy.org/)
- **Statistical & Econometric Modeling**: [Statsmodels](https://www.statsmodels.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Interactive Development**: [Jupyter](https://jupyter.org/)
- **Packaging & Tooling**: [uv](https://github.com/astral-sh/uv)

---

## 👤 Author & License

- **Author**: Shubham Shinde
- **Email**: [sshinde5930@gmail.com](mailto:sshinde5930@gmail.com)
- **License**: Released under the [Apache License 2.0](LICENSE).
