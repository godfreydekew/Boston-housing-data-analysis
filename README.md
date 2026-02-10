
![](/images/boston.png)
# Comprehensive Analysis of Boston Housing Data in R: A Step-by-Step Overview

This notebook performs a complete predictive modeling analysis of Boston housing data, progressing from exploratory data analysis through multiple regression and classification techniques to predict house values.

---

## Part 1: Data Exploration and Preparation

### 1.1 Initial Data Loading and Inspection
- **Loaded housing dataset** with 506 observations and 10 variables
- **Set factor variables** (ptRatioBin, NoxGroups) for proper categorical handling
- **Examined structure** using `dim()`, `str()`, `head()`, `tail()`, and `summary()`

### 1.2 Data Quality Assessment
- **Missing values check**: Verified no missing data across all columns
- **Duplicate detection**: Confirmed no duplicate rows in the dataset
- **Target variable distribution**: Analyzed HouseValue distribution
  - Found right-skewed distribution (skewness calculated)
  - Median house value: ~22.5k
  - Range: 5k to 50k

### 1.3 Outlier Detection
- **Boxplots** created for all numerical variables
- **Z-score method** applied (threshold > 3)
- **Key findings**:
  - Crime rate (crim): Extremely right-skewed with numerous outliers
  - Residential zoning (zn): Many outliers present
  - Age: Centered around older units
  - Rooms (rm): Relatively normal distribution

### 1.4 Correlation Analysis
- **Correlation matrix** computed for all numeric variables
- **Visualization** using ggcorrplot
- **Key relationships identified**:
  - **Strong positive**: rm (rooms) with HouseValue (r=0.7)
  - **Moderate positive**: zn (zoning) with HouseValue (r=0.36)
  - **Strong negative**: dis (distance to employment) with HouseValue (r=-0.49), indus (industrial) with HouseValue (r=-0.48)
- **Multicollinearity concerns**:
  - age-dis: r=-0.75 (old homes near employment)
  - indus-age: r=0.64 (industrial areas have old homes)
  - rad-indus: r=0.60 (highways near industries)

### 1.5 Relationship Visualization
- **Scatterplot matrix** for first 5 variables
- **Individual plots** with regression lines for key predictors:
  - Rooms vs HouseValue
  - Crime vs HouseValue
  - Distance to employment vs HouseValue
  - Industrial area vs HouseValue
  - Zoning vs HouseValue
  - Highway accessibility vs HouseValue

### 1.6 Data Splitting
- **80/20 train-test split** using `createDataPartition()`
- **Set seed (123)** for reproducibility
- **Saved datasets** as train_housing.csv and test_housing.csv

---

## Part 2: Linear Regression Analysis

### 2.1 Model Assumptions Testing

#### Test 1: Normality of Residuals
- **Shapiro-Wilk test** performed on full model residuals
- **Q-Q plot** created for visual assessment
- **Result**: Residuals NOT normally distributed (p < 0.05)

#### Test 2: Homoscedasticity (Constant Variance)
- **Residuals vs Fitted plot** created
- **Non-Constant Variance Score Test** (ncvTest) conducted
- **Result**: Non-constant variance detected

#### Test 3: Independence of Errors
- **Durbin-Watson test** performed
- **Interpretation**: DW statistic ≈ 2 indicates no autocorrelation
- **Result**: Residuals are independent

#### Test 4: Multicollinearity
- **VIF (Variance Inflation Factor)** calculated for all predictors
- **Interpretation thresholds**:
  - VIF 1-5: Acceptable
  - VIF 5-10: Concerning
  - VIF > 10: Remove variable
- **Key findings**: Some variables showed moderate VIF values

### 2.2 Model Selection Using BIC

#### Full Model
- **Formula**: HouseValue ~ crim + zn + indus + rm + age + dis + rad + NoxGroups + ptRatioBin
- All 9 predictors included

#### Stepwise Regression
- **Used BIC criterion** (k = log(n)) for variable selection
- **Step 1**: Removed NoxGroups
- **Step 2**: Removed rad
- **Final model**: 7 predictors retained

#### Model Comparison
- **BIC values** calculated for all intermediate models
- **Bayes Factor analysis** performed:
  - Compared Full vs Step1 vs Final models
  - Final model significantly preferred over full model

### 2.3 Final Linear Model Results
- **Predictors retained**: crim + zn + indus + rm + age + dis + ptRatioBin
- **R²**: 0.61 (explains 61% of variance)
- **F-statistic**: F(7, 399) = 91.71, p < 0.001
- **All predictors significant** at α = 0.05

### 2.4 Model Evaluation on Test Set
- **Predictions** generated for test data
- **Performance metrics calculated**:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² on test data
- **Comparison**: Final model vs Full model performance

### 2.5 Diagnostic Visualizations
- **Predicted vs Actual** scatter plot with 45° reference line
- **Residuals vs Fitted** plot to check patterns
- **Residual histogram** to assess distribution
- **Cook's distance plot** to identify influential observations

---

## Part 3: Logistic Regression Analysis

### 3.1 Binary Target Creation
- **Median split strategy**: HighValue = 1 if HouseValue > median, else 0
- **Balanced classes**: ~50/50 split achieved
- **Original HouseValue removed** to prevent data leakage

### 3.2 Class Distribution Analysis
- **Frequency table** and **proportions** calculated
- **Bar plot** visualizing High vs Low value houses
- **Group comparisons**:
  - Mean predictor values by class
  - Boxplots for continuous predictors
  - Chi-square tests for categorical associations

### 3.3 Log Odds Analysis

#### Full Logistic Model
- **Formula**: HighValue ~ all predictors
- **Family**: binomial (logit link)

#### Odds Ratios Interpretation
- **Calculated for significant predictors** (p < 0.05)
- **Exponential transformation** of coefficients
- **Percentage change in odds** computed

#### Interaction Testing
- Tested individual effects: zn alone, indus alone
- Tested interaction: zn × indus
- Assessed interaction significance

### 3.4 Model Selection Using BIC
- **Stepwise regression** with BIC criterion
- **Step 1**: Removed NoxGroups
- **Step 2**: Removed zn
- **Final model**: 7 predictors retained (crim, indus, rm, age, dis, rad, ptRatioBin)

#### Model Comparison Table
- **BIC values** for Full, Step1, Step2, Final models
- **Bayes Factor analysis** showed Final model strongly preferred
- **McFadden's Pseudo R²** calculated for all models

### 3.5 Logistic Model Evaluation

#### Predictions on Test Set
- **Probability predictions** generated
- **Class predictions** using threshold = 0.71 (initially)
- **Confusion matrix** created

#### Performance Metrics
- Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1-Score

#### ROC Analysis
- **ROC curve** plotted
- **AUC** calculated
- **Optimal cutoff point** identified using Youden's index

#### Precision-Recall Analysis
- **Multiple thresholds** tested (0.1 to 0.9 by 0.1)
- **Precision-Recall-F1** table created
- **PR curve** plotted

---

## Part 4: Decision Tree Models

### 4.1 Data Verification
- **Checked dimensions** of train and test sets
- **Verified no data leakage** between sets
- **Confirmed proper split** (no row overlap)

### 4.2 Simple Decision Tree (No CV)

#### Model Training
- **Method**: rpart with class classification
- **No hyperparameter tuning**
- **Tree visualization** using rpart.plot

#### Evaluation
- **Predictions** at 0.5 threshold
- **Confusion matrix** generated
- **Accuracy** reported

### 4.3 Cross-Validated Decision Tree

#### Training Setup
- **10-fold cross-validation** implemented
- **Automatic CP tuning** performed
- **Best CP selected** based on CV performance

#### Model Performance
- **Tree visualization** of final model
- **Predictions** at 0.5 threshold
- **Confusion matrix** and accuracy reported

---

## Part 5: Random Forest Models

### 5.1 Simple Random Forest (No CV)

#### Model Training
- **Default parameters**: 500 trees
- **Variable importance** calculated
- **Importance plot** created

#### Evaluation
- **Predictions** at 0.71 threshold
- **Confusion matrix** generated
- **Accuracy** reported

### 5.2 Tuned Random Forest (With CV)

#### Training Setup
- **10-fold cross-validation**
- **mtry tuning grid**: tested values 2, 3, 4, 5, 6
- **500 trees** used

#### Tuning Results
- **Best mtry** selected
- **CV results plotted**
- **Variable importance** from tuned model

#### Evaluation
- **Predictions** at 0.71 threshold
- **Confusion matrix** and accuracy reported

---

## Part 6: Unified Model Comparison

### 6.1 Standardized Metrics at Cutoff 0.5
- **All four models evaluated**:
  1. Decision Tree (Simple)
  2. Decision Tree (CV)
  3. Random Forest (Simple)
  4. Random Forest (Tuned)

#### Metrics Calculated
- Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1-Score

#### Factor Level Alignment
- **Pre-aligned factors** to prevent warnings
- **Consistent levels**: "0" and "1" across all comparisons

### 6.2 McFadden's Pseudo R²
- **Calculated for each model** to assess goodness-of-fit
- **Null model comparison** used as baseline
- **Results table** created

### 6.3 Comprehensive Precision-Recall Analysis

#### Curve Generation
- **Multiple thresholds** tested (0.1 to 0.9 by 0.05)
- **PR curves plotted** for all four models
- **Color-coded** for easy comparison

#### Visualization Improvements
- **Dynamic axis limits** based on actual data range
- **Legend positioned outside** plot area (right margin)
- **Clean line plots** (removed point markers)
- **Grid added** for readability

---

## Key Findings Summary

### Linear Regression
- **Best model**: 7 predictors explain 61% of variance
- **Rooms (rm)** strongest positive predictor
- **Distance to employment (dis)** strongest negative predictor

### Logistic Regression
- **Final model**: 7 predictors with good discrimination
- **McFadden's Pseudo R²**: Moderate fit
- **Optimal threshold**: ~0.71 for best F1-Score

### Tree-Based Models
- **Random Forests outperform** simple decision trees
- **Cross-validation improves** generalization
- **Variable importance**: rm, dis, indus most important

### Overall Best Model
- Determined by comparing F1-Scores and PR curves across all models at standardized cutoff of 0.5

---

## Technical Methodology

### Statistical Rigor
- **Set seed (123)** throughout for reproducibility
- **Proper train-test split** (80/20) with no leakage
- **Multiple evaluation metrics** for comprehensive assessment
- **Assumption testing** performed before parametric modeling

### Model Selection
- **BIC criterion** for linear and logistic regression
- **Bayes Factor analysis** for model comparison
- **Cross-validation** for tree-based models
- **Hyperparameter tuning** for random forests

### Visualization Quality
- **Professional plots** with appropriate labels
- **Color schemes** for multi-model comparison
- **Diagnostic plots** for assumption checking
- **Interactive elements** considered (e.g., legend placement)