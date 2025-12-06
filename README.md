# 📊 Customer Churn Prediction Dashboard

An interactive **Machine Learning web app** built with **Streamlit** to analyze customer behavior, understand churn factors, and predict customer retention using a trained **XGBoost model**.

---

## 🚀 Project Overview

Customer churn prediction is one of the most important use cases in customer analytics.  
This project demonstrates an **end-to-end Data Science workflow** — from data preprocessing and model training to deployment using Streamlit Cloud.

The app allows users to:
- Explore customer demographics and behavioral data  
- Understand key churn factors through visual analysis  
- Predict whether a customer will **stay or leave** using a trained ML model  

---

## 🧩 Features

✅ Interactive user interface built with **Streamlit**  
✅ Visual analytics powered by **Plotly Express**  
✅ Machine Learning pipeline with **XGBoost** and **Scikit-Learn**  
✅ Real-time churn prediction form  
✅ Animated **Lottie illustrations** for a modern experience  

---

## 🧠 Model Information

- **Algorithm Used:** XGBoost Classifier  
- **Evaluation Metric:** F1-Score (Cross Validation)  
- **Preprocessing:**
  - OneHotEncoding for categorical features  
  - StandardScaler for numerical features  
  - Combined via `ColumnTransformer`  
- **Saved Model:** `customer_churn_model.pkl`

---

## 📁 Dataset Information

**File:** `Customer_Churn_Cl.csv`

Contains:
- **Demographics:** Age, Gender  
- **Account Details:** Contract Length, Subscription Type  
- **Behavioral Data:** Usage Frequency, Support Calls, Payment Delay, Tenure, Total Spend, Last Interaction  
- **Target Variable:** Churn (0 = Stay, 1 = Leave)

---

## 🛠️ Technologies Used

| Category | Tools |
|-----------|-------|
| **Frontend** | Streamlit, Plotly |
| **Backend / ML** | Scikit-learn, XGBoost |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Plotly Express |
| **Animation** | Streamlit-Lottie |
| **Version Control** | Git & GitHub |

---

## ⚙️ Installation & Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/ahmedshlaby/customer-churn-prediction.git
   cd customer-churn-prediction

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Run the Streamlit app**
   ```bash
    streamlit run customer_churn.py

---

## 🌐 Deployment

This app is **fully deployed on Streamlit Cloud** for public access.

🔗 **Live Demo:** [Click Here to View App](https://ahmedshlaby-customer-churn-prediction-customer-churn-bv74t0.streamlit.app/)  
🔗 **GitHub Repository:** [https://github.com/ahmedshlaby/customer-churn-prediction](https://github.com/ahmedshlaby/customer-churn-prediction)

---

## 👨‍💻 Author

**Ahmed Ahmed Mohamed Shlaby**  
📍 Egypt, Cairo  
📧 [shalabyahmed299@gmail.com](mailto:shalabyahmed299@gmail.com)  

🔗 [GitHub](https://github.com/ahmedshlaby) | [LinkedIn](https://www.linkedin.com/in/ahmed-shlaby22/)

---

## 🏁 Conclusion

This project demonstrates a **complete customer churn prediction pipeline**, combining:

- 🧹 **Data Cleaning & Preprocessing**  
- 🧠 **Feature Engineering & Model Selection**  
- 📊 **Exploratory Data Analysis & Visualization**  
- 🚀 **Deployment with Streamlit Cloud**

---

⭐ *If you like this project, don’t forget to give it a star on GitHub!*


   
