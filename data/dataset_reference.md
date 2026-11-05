# Dataset Reference: Mall Customer Segmentation

## 1. Overview and Source
- **Dataset Title**: Mall Customer Segmentation Data
- **Original Source**: Publicly accessible via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and [Kaggle Customer Segmentation Datasets](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).
- **Collection Context**: The dataset was recorded by a major shopping mall through customer loyalty/membership card enrollments. Each record summarizes demographic attributes alongside aggregated behavioral purchasing performance.

## 2. Dataset Schema and Attributes

| Column Name | Data Type | Measurement Unit | Description | Expected Range |
| :--- | :--- | :--- | :--- | :--- |
| `CustomerID` | Integer | Unique Identifier | Anonymous sequential ID assigned to each customer | 1 to 200 |
| `Gender` | Categorical (String) | Demographics | Gender of the shopper (`Male` / `Female`) | Binary |
| `Age` | Integer | Years | Age of the customer at time of record | 18 to 70 |
| `Annual Income (k$)` | Integer / Float | Thousands of USD ($) | Computed annual earnings of the customer | $15k to $137k |
| `Spending Score (1-100)`| Integer | Normalized Score | Proprietary score assigned by the mall based on purchase frequency, basket value, and loyalty history | 1 to 99 |

## 3. Dataset Characteristics and Exploratory Baseline
- **Total Records**: 200 observations
- **Missing Values**: 0 null records (verified clean)
- **Duplicate Records**: 0 duplicates detected
- **Gender Distribution**: 112 Females (56.0%), 88 Males (44.0%)
- **Baseline Metrics**:
  - `Age`: Mean = 38.85 years, Median = 36.00 years, Std = 13.97
  - `Annual Income`: Mean = $60.56k, Median = $61.50k, Std = $26.26k
  - `Spending Score`: Mean = 50.20, Median = 50.00, Std = 25.82

## 4. Preprocessing and Feature Engineering
1. **Column Normalization**: Synonyms such as `Genre` or `Sex` are automatically converted to `Gender`. Whitespace is stripped.
2. **Missing & Duplicate Handling**: Imputation pipeline uses median for numerical features and mode for categorical features if missing values are present; duplicate rows are automatically identified and pruned.
3. **Scaling**: Standardized via Scikit-Learn `StandardScaler` ($\mu = 0, \sigma = 1$) to ensure Euclidean distance calculations in K-Means are not distorted by differences in measurement magnitude (e.g., dollars vs score points).
4. **Sample Storage**: A 15-row cleaned sample is preserved at `data/cleaned_customers_sample.csv`.
