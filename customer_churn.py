

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import requests
from streamlit_lottie import st_lottie
import json

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Customer Churn Analysis Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")



# ------------------- LOADDATA -------------------
@st.cache_data
def load_data():
    return pd.read_csv("Customer_Churn_Cl.csv")

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")

df = load_data()
model = load_model()



# ------------------- LOTTIE ANIMATION -------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_success = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_touohxv0.json")
lottie_warning = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_twijbubv.json")
lottie_home = load_lottieurl("https://lottie.host/d3a0bb10-39c7-4613-ac3a-8834837a6605/QBZaQK1gPN.json")
lottie_project = load_lottieurl("https://lottie.host/a69c1ea3-db70-4c55-98b5-c28d7601a5fe/LOegsJACJK.json")






# ------------------- SIDEBAR NAVIGATION -------------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio( "Go to",
                         ["🏠 Home",
                          "📊 EDA",
                          "🤖 Prediction",
                          "🎯 Presentation"]
                        )




# ------------------- HOME PAGE -------------------
if page == "🏠 Home":
    # Title & intro
    st.markdown("""
        <h1 style='text-align:center; color:#008C9E;'>📊 Customer Churn Prediction Dashboard</h1>
        <p style='text-align:center; font-size:18px;'>
            A professional machine learning web app to analyze customer behavior, understand churn factors, and predict customer retention.
        </p>""", unsafe_allow_html=True)
    
    st_lottie(lottie_home, height=350, key="home_animation")

    st.markdown("---")

    # Project overview
    st.markdown("<h3 style='color:#008C9E;'>💡 Project Overview</h3>", unsafe_allow_html=True)
    st.write("""
    This dashboard is part of a data science project aimed at predicting **customer churn** using multiple machine learning algorithms.
    
    It helps identify key factors that influence customer decisions to stay or leave a service, using **exploratory data analysis**, **visual analytics**, and **real-time prediction**.
    """)

    st.info("""
    **Goal:** Predict whether a customer will churn or stay based on historical and behavioral data.
    """)

    # Key project info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='color:#008C9E;'>🧠 Project Details</h3>", unsafe_allow_html=True)
        st.write("""
        - **Dataset:** Customer_Churn_Cl.csv  
        - **Models Used:** Logistic Regression, Random Forest, KNN, Decision Tree, CatBoost, GaussianNB, LightGBM, XGBoost, SVM  
        - **Evaluation Metric:** F1 Score via Cross Validation  
        """)
    with col2:
        st_lottie(lottie_project, speed=1, reverse=False, loop=True, quality="high", height=230, key="project_animation")

    st.markdown("---")


    # Feature Highlights
    st.markdown("<h3 style='color:#008C9E;'>🚀 Dashboard Features</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background-color:#f8f9fa; padding:15px; border-radius:10px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
        <h3 style='color:#008C9E;'>📊 EDA Insights</h3>
        <p style='color:#000000;'>Explore customer demographics and churn distribution through dynamic visuals.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background-color:#f8f9fa; padding:15px; border-radius:10px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
        <h3 style='color:#008C9E;'>🤖 Machine Learning</h3>
        <p style='color:#000000;'>Compare and evaluate multiple ML models using cross-validation metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background-color:#f8f9fa; padding:15px; border-radius:10px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
        <h3 style='color:#008C9E;'>🔮 Predictions</h3>
        <p style='color:#000000;'>Get real-time churn predictions by entering customer details.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Contact Section
    st.markdown("<h3 style='color:#008C9E;'>🔗 Connect with Me</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; font-size:18px;'>
    🌐 <a href='https://github.com/ahmedshlaby' target='_blank'>GitHub</a> | 
    💼 <a href='https://www.linkedin.com/in/ahmed-shlaby22' target='_blank'>LinkedIn</a> | 
    📧 <a href='mailto:shalabyahmed299@gmail.com'>Email</a>
    </div>
    """, unsafe_allow_html=True)


# ------------------- EDA PAGE -------------------
elif page == "📊 EDA":
    st.markdown("<h1 style='color:#008C9E; text-align:center;'>📊 Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style=text-align:center;'> Explore the customer churn dataset with interactive visualizations and insights. </h3>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Dataset Overview 
    st.markdown("### 🧩 Dataset Overview")

    kpi_style = """
    <style>
    .kpi-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .kpi-title {
        font-size: 18px;
        color: #008C9E;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 22px;
        color: #333333;
        font-weight: bold;
    }
    </style>
    """

    st.markdown(kpi_style, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Total Rows</div>
            <div class='kpi-value'>{df.shape[0]:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Total Columns</div>
            <div class='kpi-value'>{df.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Churned Customers</div>
            <div class='kpi-value'>{df[df['Churn'] == 1].shape[0]:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Retention Rate</div>
            <div class='kpi-value'>{100 - df['Churn'].mean()*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("📄 View Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Data Understanding Section
    st.markdown("### 🧾 Data Understanding")
    st.write("Below is a summary of the main dataset columns and what each represents:")

    with st.expander("📘 Click to view column meanings"):
        data_info = {
            "CustomerID": "Unique identifier for each customer",
            "Gender": "Customer gender (Male/Female)",
            "Age": "Customer age in years",
            "Tenure": "Number of months the customer has stayed with the company.",
            "Usage Frequency": "Frequency of service usage by the customer",
            "Subscription Type": "Type of subscription plan (Basic, Standard, Premium)",
            "Contract Length": "Length of the contract (Monthly, Quarterly, Annual)",
            "Total Spend": "Total amount spent by the customer",
            "Last Interaction": "Days since last customer interaction",
            "Support Calls": "Number of customer support calls made",
            "Payment Delay": "Number of times or days the customer delayed payments",
            "Churn": "Target variable (1) if customer churned, (0) if stayed"
        }

        df_info = pd.DataFrame(list(data_info.items()), columns=["Column Name", "Description"])
        st.dataframe(df_info, use_container_width=True)

    st.markdown("---")

    # --- Churn Distribution ---
    st.markdown("### 💬 Churn Distribution Overview")
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn', 'Count']
    fig_pie = px.pie(
        churn_counts,
        names='Churn',
        values='Count',
        color='Churn',
        color_discrete_map={0: '#00B09B', 1: '#FF4B2B'},
        hole=0.4,
        title="Customer Churn Ratio"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Feature Distribution ---
    st.markdown("### 🧠 Feature Distribution by Churn")
    feature = st.selectbox("Select a feature to explore:", df.columns, index=0)
    fig_hist = px.histogram(
        df,
        x=feature,
        color="Churn",
        barmode="group",
        color_discrete_map={0: '#00B09B', 1: '#FF4B2B'},
        title=f"{feature} Distribution by Churn"
    )
    st.plotly_chart(fig_hist, use_container_width=True)



    # --- Correlation Heatmap ---
    st.markdown("### 🔥 Correlation Heatmap")

    num_df = df.select_dtypes(include=["int64", "float64"])

    custom_colors = [
        [0.0, "#FF4B2B"],   # Strong negative correlation (red)
        [0.5, "#f8f9fa"],   # Neutral (white/light gray)
        [1.0, "#00B09B"]    # Strong positive correlation (greenish/teal)
    ]

    fig_corr = px.imshow(
        num_df.corr(),
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=custom_colors,
        title="Feature Correlation Heatmap"
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)


    # --- Key Insights ---
    st.markdown("### 🎯 Key Insights")

    insight_style = """
    <style>
    .insight-card {
        background-color: #f8f9fa;
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        margin-bottom: 12px;
    }
    .insight-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    .insight-title {
        margin-bottom: 5px;
        color: #008C9E;
        font-size: 18px;
        font-weight: bold;
    }
    .insight-text {
        color: #333;
        font-size: 16px;
        margin-top: 0;
    }
    </style>
    """

    st.markdown(insight_style, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='insight-card'>
            <div class='insight-title'>📉 Tenure & Support Calls</div>
            <div class='insight-text'>Customers with <b>shorter tenure</b> and <b>frequent support calls</b> are more likely to churn.</div>
        </div>

        <div class='insight-card'>
            <div class='insight-title'>💰 Payment Behavior</div>
            <div class='insight-text'><b>Payment delays</b> show a strong correlation with churn, often signaling dissatisfaction.</div>
        </div>

        <div class='insight-card'>
            <div class='insight-title'>📊 Spending & Subscription</div>
            <div class='insight-text'><b>Premium</b> and <b>high-spending</b> users tend to stay longer compared to basic plan customers.</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='insight-card'>
            <div class='insight-title'>📆 Contract Type Impact</div>
            <div class='insight-text'>Longer <b>contract lengths</b> lead to better retention, while gender shows minor impact.</div>
        </div>

        <div class='insight-card'>
            <div class='insight-title'>🔗 Feature Correlations</div>
            <div class='insight-text'>Moderate relationships exist between <b>tenure</b>, <b>usage frequency</b>, and <b>payment delay</b>.</div>
        </div>
        """, unsafe_allow_html=True)




# ------------------- PREDICTION PAGE -------------------
elif page == "🤖 Prediction":
    st.markdown("""
    <h1 style='text-align:center; color:#008C9E;'>
    🤖 Customer Churn Prediction
    </h1>
    """, unsafe_allow_html=True)

    
    st.markdown('''<p style="text-align:center; font-size: 18px; color: #333333; line-height: 1.6; margin: 0;">
                        Predict whether a customer will <b>stay</b> or <b>churn</b> using the trained <b>Machine Learning model</b>.
                    </p>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    # --- Input Container (Glass Effect) ---
    st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(155,89,182,0.15), rgba(52,152,219,0.15));
                    border-radius: 20px;
                    padding: 25px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    backdrop-filter: blur(8px);
                    text-align: center;
                    margin-top: 15px;
                    margin-bottom: 25px;
                    ">
                    <h2 style="
                        font-size: 34px;
                        text-align:center;
                        color: #008C9E;
                        line-height: 1.6;
                        margin: 0;
                    ">
                        🧾 Customer Information
                    </h2>
                </div>
                """, unsafe_allow_html=True)


    

    # --- Demographics ---
    st.markdown("<h3 style='color:#008C9E;'>👤 Demographics</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, step= 1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    # --- Account Details ---
    st.markdown("<h3 style='color:#008C9E;'>💼 Account Details</h3>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    with col4:
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

    # --- Behavioral Data ---
    st.markdown("<h3 style='color:#008C9E;'>📊 Behavioral Data</h3>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        total_spend = st.number_input("Total Spend ($)", min_value=0.0, step=10.0)
        support_calls = st.slider("Support Calls", 0, 10, 2)
        payment_delay = st.slider("Payment Delay (times)", 0, 50, 1)
    with col6:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        usage_frequency = st.slider("Usage Frequency", 0, 100, 50)
        last_interaction = st.slider("Last Interaction (days)", 0, 100, 10)

    # --- Combine all inputs ---
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Contract Length": [contract_length],
        "Subscription Type": [subscription_type],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Tenure": [tenure],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Predict Button (Animated style) ---
    btn_style = """
                    <style>
                    div.stButton > button:first-child {
                        background: linear-gradient(90deg, #008C9E, #00C9A7);
                        background-size: 200% auto;
                        color: white;
                        border: none;
                        border-radius: 12px;
                        font-size: 1.1em;
                        font-weight: 600;
                        width: 100%;
                        padding: 0.8em 2em;
                        transition: 0.5s;
                        box-shadow: 0 0 12px rgba(0, 140, 158, 0.3);
                    }
                    div.stButton > button:hover {
                        background-position: right center;
                        transform: scale(1.05);
                        box-shadow: 0 0 20px rgba(0, 201, 167, 0.6);
                    }
                    </style>
                """

    st.markdown(btn_style, unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    predict_button = st.button("🔍 Predict Churn Probability")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Prediction Logic ---
    if predict_button:
        try:
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #FF416C, #FF4B2B);
                    border-radius: 20px;
                    padding: 25px;
                    color: white;
                    text-align:center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                ">
                <h2>❌ Customer Likely to CHURN</h2>
                </div>
                """, unsafe_allow_html=True)
                if lottie_warning:
                    st_lottie(lottie_warning, height=160)
            else:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #00B09B, #96C93D);
                    border-radius: 20px;
                    padding: 25px;
                    color: white;
                    text-align:center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                ">
                <h2>✅ Customer Likely to STAY</h2>
                </div>
                """, unsafe_allow_html=True)
                if lottie_success:
                    st_lottie(lottie_success, height=160)

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Model: XGBoost Pipeline | Developed by Ahmed Shlaby", icon="🤖")
    



    

# ------------------- PRESENTATION PAGE -------------------
elif page == "🎯 Presentation":
    st.markdown("<h1 style='color:#008C9E; text-align:center;'>🎯 Project Presentation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'> This page summarizes all steps performed throughout the project, from start to deployment.</p>", unsafe_allow_html=True)

    st.markdown("---")
    
    # ----------------- Problem Definition -------------
    st.markdown("<h3 style='color:#008C9E;'>🧭 Step 1: Problem Definition</h3>", unsafe_allow_html=True)
    st.write("""
    The goal of this project was to **predict customer churn**, identifying which customers are most likely to leave the service based on their demographic and behavioral data.
    """)

    # ----------------- Data Collection -------------
    st.markdown("<h3 style='color:#008C9E;'>📂 Step 2: Data Collection</h3>", unsafe_allow_html=True)
    st.write("""
    The dataset **Customer_Churn_Cl.csv** includes:
    - Demographics (Age, Gender)
    - Account details (Contract Length, Subscription Type)
    - Behavioral data (Usage Frequency, Support Calls, Payment Delay, Tenure, Total Spend, Last Interaction)
    - Target variable: **Churn (0 = Stay, 1 = Leave)**
    """)

    # ----------------- Data Cleaning & Preprocessing -------------
    st.markdown("<h3 style='color:#008C9E;'>🧹 Step 3: Data Cleaning & Preprocessing</h3>", unsafe_allow_html=True)
    st.write("""
    - Removed duplicates and handled missing values  
    - Encoded categorical variables using **OneHotEncoder**  
    - Scaled numerical features with **StandardScaler**  
    - Combined both transformations using **ColumnTransformer**  
    - Applied preprocessing pipelines to prepare the data for multiple machine learning models  
    """)


    # Info For Code
    st.info("""
    🧠 **Model Training Setup:**  
    During experimentation, several models were trained and compared using **cross-validation (F1-score)**, including:
    - Logistic Regression  
    - Random Forest  
    - K-Nearest Neighbors (KNN)  
    - Decision Tree  
    - Gaussian Naive Bayes  
    - CatBoost  
    - LightGBM  
    - XGBoost  
    - Support Vector Machine (SVC)
    """)

    
    # Code
    st.code("""
    from sklearn.model_selection import cross_validate
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC

    models = [
        ('Logistic Regression', LogisticRegression(random_state=42, n_jobs=-1)),
        ('Random Forest', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('CatBoost', CatBoostClassifier(verbose=0)),
        ('Gaussian NB', GaussianNB()),
        ('LightGBM', LGBMClassifier()),
        ('XGBoost', XGBClassifier()),
        ('SVM', SVC())
    ]

    for model in models:
        model_pipeline = Pipeline([
            ('Preprocessing', preprocessing),
            ('Model', model[1])
        ])
        results = cross_validate(model_pipeline, X, y, scoring='f1', return_train_score=True, n_jobs=-1)
        print(model[0])
        print('Train F1 Score :', round(results['train_score'].mean() * 100, 2))
        print('Test F1 Score :', round(results['test_score'].mean() * 100, 2))
        print('AVG Fit Time :', round(results['fit_time'].mean(), 2))
        print('-' * 100)
    """, language="python")


    # ----------------- Model Development & Selection -------------
    st.markdown("<h3 style='color:#008C9E;'>🧠 Step 5: Model Development & Selection</h3>", unsafe_allow_html=True)
    st.write("""
    - Compared all models based on **F1-score** and training time  
    - Selected **XGBoost Classifier** as the final model due to its best trade-off between accuracy and performance  
    - Integrated XGBoost into a **Scikit-learn Pipeline** for reproducibility  
    - Saved the final trained pipeline as `customer_churn_model.pkl`
    """)


    # -----------------  Model Evaluation -------------
    st.markdown("<h3 style='color:#008C9E;'>📊 Step 6: Model Evaluation</h3>", unsafe_allow_html=True)
    st.write("""
    - Evaluated model using metrics such as F1-score  
    - Verified model generalization with cross-validation  
    - Confirmed stability and interpretability for production use
    """)


    # ----------------- Deployment with Streamlit -------------
    st.markdown("<h3 style='color:#008C9E;'>🚀 Step 7: Deployment with Streamlit</h3>", unsafe_allow_html=True)
    st.write("""
    The final pipeline was deployed using **Streamlit**, enabling users to:
    - Explore and visualize customer data  
    - Interactively predict churn probabilities  
    - Gain actionable insights through an intuitive dashboard
    """)


    # ----------------- Conclusion & Insights -------------
    st.markdown("<h3 style='color:#008C9E;'>🎓 Step 8: Conclusion & Insights</h3>", unsafe_allow_html=True)
    st.write("""
            - The analysis identifies high-risk churn customers for proactive retention  
            - Demonstrates a complete **end-to-end Data Science workflow** from data to deployment  
            - Combines **EDA, feature engineering, model comparison, and visualization** into one integrated system
            """)


    ##------ Video ------
    st.markdown("""
    <div style='background: #f0f8ff; padding: 20px; border-radius: 15px; text-align:center'>
        <h3 style='color:#008C9E;'>🎥 Watch the Project Demo</h3>
    </div>
    """, unsafe_allow_html=True)

    
    video_url = "https://www.dropbox.com/scl/fi/jyvd9nlmy7flh2k2q9dqz/Customer_Churn.ipynb-Final-Project-Visual-Studio-Code-2025-10-14-21-31-50.mp4?rlkey=acoq1dyb6ek8f0ea29f69grvn&dl=1"
    st.video(video_url)


    
    # Footer
    st.markdown("""---""")
    st.markdown("""
                <p style='text-align: center; font-size: 14px;'>
                    © 2025 | Developed by <strong>Ahmed Shlaby</strong> | 📧 <a href="mailto:shalabyahmed299@gmail.com">Contact</a>
                </p>
                """, unsafe_allow_html=True)


















