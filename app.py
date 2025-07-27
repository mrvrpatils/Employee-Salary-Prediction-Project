
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Emplyee Salary Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #1E3A8A; /* A deep blue color */
        text-align: center;
        font-family: 'Arial Black', sans-serif;
    }
    .st-emotion-cache-16txtl3 {
        background-color: #DBEAFE; /* A light blue color */
    }
    .stButton>button {
        background-color: #2563EB; /* A vibrant blue */
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(0, 118, 255, 0.39);
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        color: white;
    }
    .st-emotion-cache-9115Gi {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Model and Data Loading 
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model.pkl")
        return model
    except FileNotFoundError:
        st.error("🚨 Model file not found! Please ensure 'best_model.pkl' is in the same directory.")
        return None

model = load_model()

# Encoding Maps (from your notebook) 
workclass_map = {'?': 0, 'Federal-gov': 1, 'Local-gov': 2, 'Never-worked': 3, 'Private': 4, 'Self-emp-inc': 5, 'Self-emp-not-inc': 6, 'State-gov': 7, 'Without-pay': 8}
marital_status_map = {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6}
occupation_map = {'?': 0, 'Adm-clerical': 1, 'Armed-Forces': 2, 'Craft-repair': 3, 'Exec-managerial': 4, 'Farming-fishing': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Other-service': 8, 'Priv-house-serv': 9, 'Prof-specialty': 10, 'Protective-serv': 11, 'Sales': 12, 'Tech-support': 13, 'Transport-moving': 14}
relationship_map = {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5}
race_map = {'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4}
gender_map = {'Female': 0, 'Male': 1}
native_country_map = {'United-States': 39, 'Mexico': 26, 'Philippines': 30, 'Germany': 11, 'Canada': 2, 'Puerto-Rico': 33, 'El-Salvador': 8, 'India': 19, 'Cuba': 5, 'England': 9, '?': 0}


st.title("Employee Salary Prediction App 📈")
st.markdown("<h3 style='text-align: center; color: #555;'>Employees Salary Prediction by Their Income <=50k or >=50k with Machine Learning</h3>", unsafe_allow_html=True)
st.markdown("---")


with st.sidebar:
    st.header("👤 Employee Profile")
    st.markdown("Adjust the features below to get a salary prediction.")

    age = st.slider("Age", 17, 75, 35, help="Employee's age.")
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=12000, max_value=1500000, value=180000, help="Census Bureau weighting factor.")
    educational_num = st.slider("Education Level (Numeric)", 5, 16, 10, help="Numeric representation of education.")
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, help="Income from investments.")
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, help="Losses from investments.")
    hours_per_week = st.slider("Hours per Week", 1, 99, 40, help="Number of hours worked per week.")

    workclass = st.selectbox("Work Class", list(workclass_map.keys()), index=4)
    marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
    occupation = st.selectbox("Occupation", list(occupation_map.keys()))
    relationship = st.selectbox("Relationship", list(relationship_map.keys()))
    race = st.selectbox("Race", list(race_map.keys()))
    gender = st.selectbox("Gender", list(gender_map.keys()))
    native_country = st.selectbox("Native Country", list(native_country_map.keys()))

# Data Preparation for Model 
input_data = {
    'age': age, 'workclass': workclass_map[workclass], 'fnlwgt': fnlwgt,
    'educational-num': educational_num, 'marital-status': marital_status_map[marital_status],
    'occupation': occupation_map[occupation], 'relationship': relationship_map[relationship],
    'race': race_map[race], 'gender': gender_map[gender], 'capital-gain': capital_gain,
    'capital-loss': capital_loss, 'hours-per-week': hours_per_week,
    'native-country': native_country_map[native_country]
}
input_df = pd.DataFrame([input_data])
model_features = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
                  'occupation', 'relationship', 'race', 'gender', 'capital-gain',
                  'capital-loss', 'hours-per-week', 'native-country']
input_df = input_df[model_features]

# UI: Main Panel for Prediction 
if model is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Prediction Output")

        if st.button("📈 Predict Employee Salary Class", use_container_width=True, type="primary"):
            prediction = model.predict(input_df)
            pred_proba = model.predict_proba(input_df)

            if prediction[0] == '>50K':
                st.metric(label="Predicted Salary Class", value="> 50K", delta="High Income Potential", delta_color="normal")
                st.success("The model predicts this individual has a high probability of earning over $50,000 annually.")
            else:
                st.metric(label="Predicted Salary Class", value="<= 50K", delta="Standard Income Potential", delta_color="inverse")
                st.info("The model predicts this individual likely earns at or below $50,000 annually.")

    with col2:
        st.subheader("Confidence Score")

        if 'pred_proba' in locals():
            proba_df = pd.DataFrame(pred_proba, columns=model.classes_).T
            proba_df = proba_df.rename(columns={0: 'Probability'})

            fig = go.Figure(data=[go.Bar(
                x=proba_df.index,
                y=proba_df['Probability'],
                marker_color=['#60A5FA', '#2563EB'],
                text=proba_df['Probability'].apply(lambda x: f'{x:.1%}'),
                textposition='auto',
            )])
            fig.update_layout(
                title_text='Prediction Confidence',
                yaxis_title='Probability',
                xaxis_title='Salary Class',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Click the 'Predict' button to see the confidence score.")

    with st.expander("Show Input Data (Encoded for Model)"):
        st.dataframe(input_df)

# Footer 
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed  By M Vittal Rao By Using Streamlit & Scikit-learn <br> Thankyou For Using This App....</p>", unsafe_allow_html=True)
