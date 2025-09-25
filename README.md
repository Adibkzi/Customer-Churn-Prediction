# 📉 Customer Churn Prediction — End-to-End ML (Portfolio Project)

**Author:** Adib Kazi — M.S. Data Science  
**Tech Stack:** Python, pandas, scikit-learn, Streamlit  

---

## 🎯 Problem Statement

Subscription businesses such as telecoms and SaaS companies suffer significant revenue loss when customers **churn** (cancel service).  

The business challenge:  
- Retention teams can only reach out to **~10% of customers per cycle** due to budget and staff limits.  
- We must identify and prioritize **which customers are most at risk of churn** so outreach is **efficient and cost-effective**.  

---

## 🧭 Data Science Approach

Following a **CRISP-DM framework**:

### 1. Business Understanding
- Goal: Predict churn and provide an actionable list of the **top 10% riskiest customers**.  
- KPI: High **precision** at top-decile → ensures outreach targets real churners.  

### 2. Data Understanding
- Dataset: Simulated churn dataset with raw, imperfect data.  
- Key features:  
  - `tenure_months`, `monthly_charges`, `contract_type`, `internet_type`,  
  - `support_tickets_90d`, `num_logins_30d`, `payment_method`, `region`, `late_payments_12m`, `age`.  
- Target: `churn` (binary 0/1).  

### 3. Data Preparation
- Handle missing values (median for numerics, mode for categoricals).  
- One-hot encode categorical variables.  
- Standardize numeric variables.  
- All steps built into a **sklearn `Pipeline`** for reproducibility.  

### 4. Modeling
- Baseline: Logistic Regression (interpretable).  
- Final model: **RandomForestClassifier** (300 trees, random_state=42).  
- Stored in a pipeline (`preprocessor + model`).  

### 5. Evaluation
- **ROC-AUC**: measures overall separation.  
- **PR-AUC**: robust for imbalanced classes.  
- **Top-Decile Thresholding**: simulate business case (flag top 10% risk scores).  

---

## 📊 Results

### Metrics Summary

| Metric                  | Score  |
|--------------------------|--------|
| ROC-AUC                 | 0.82   |
| PR-AUC                  | 0.56   |
| Precision @ Top 10%      | 0.68   |
| Recall @ Top 10%         | 0.42   |

### Interpretation
- The model is **well-calibrated** and separates churners from non-churners.  
- At **top 10% outreach capacity**, ~68% of flagged customers are true churners.  
- Business can contact fewer customers while still capturing ~42% of churners.  



---

##  Streamlit Demo

👉 [Launch App](http://localhost:8501/)  
### About the Author

 Passionate about applied machine learning, I enjoy solving real-world business problems end-to-end — from messy raw data to deployment.

