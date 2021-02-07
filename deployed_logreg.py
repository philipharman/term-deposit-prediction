import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.write("""
# Term Deposit Prediction App
This app predicts if a client will subscribe to a term deposit.
""")
st.sidebar.header('Client Data')

# Categorical selections
job_statuses = ['blue-collar', 'technician', 'management', 'services', 'retired','admin.', 'housemaid', 'unemployed', 'entrepreneur','self-employed', 'student', 'unknown']
marital_statuses = ['married', 'single', 'divorced','unknown']
education_statuses = ['university.degree', 'professional.course' , 'high.school', 'basic', 'illiterate', 'unknown']
default_statuses = ['yes','no','unknown']
housing_statuses = ['yes','no','unknown']
loan_statuses = ['yes','no','unknown']
contact_types = ['cellular', 'telephone']
months = ['mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
days_of_week = ['mon','tue','wed','thu','fri']
poutcomes = ['success', 'failure','nonexistent']

# User select parameters function
def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 50)
    job = st.sidebar.selectbox('Job',(job_statuses))
    marital = st.sidebar.selectbox('Marital status',(marital_statuses))
    education = st.sidebar.selectbox('Education',(education_statuses))
    default = st.sidebar.selectbox('Credit in default',(default_statuses))
    housing = st.sidebar.selectbox('Housing loan', housing_statuses)
    loan = st.sidebar.selectbox('Loan (Other)',(loan_statuses))
    contact = st.sidebar.selectbox('Method of contact',(contact_types))
    month = st.sidebar.selectbox('Month of contact',(months))
    day_of_week = st.sidebar.selectbox('Weekday of contact',(days_of_week))
    duration = st.sidebar.slider('Duration of last contact (sec)', 0, 5000, 250)
    campaign = st.sidebar.slider('Number of contacts during current campaign', 0, 60, 3)
    pdays = st.sidebar.slider('Days passed since last contact (999 if not contacted)', 0, 999, 15)
    previous = st.sidebar.slider('Number of contacts before current campaign', 0, 7, 0)
    poutcome = st.sidebar.selectbox('Outcome of previous campaign',(poutcomes))
    emp_var_rate = st.sidebar.slider('Emloyment variability rate',-3.5,1.5,-1.0) # Need to increment by 1/10
    cons_price_idx = st.sidebar.slider('Consumer price index',92.00, 95.00 , 93.50) # Need to increment by 1/100
    cons_conf_idx = st.sidebar.slider('Consumer confidence index',-51.0,-26.0,-40.0) # Need to increment by 1/10
    euribor3m = st.sidebar.slider('Euribor3m',0.60,5.10,4.85) # increment 1/100
    nr_employed = st.sidebar.slider('No. Employed',4900,5300,5200)

    data = {'age' : age, 'job' : job, 'marital' : marital, 'education' : education,
            'default' : default, 'housing' : housing, 'loan' : loan ,
            'contact' : contact, 'month' : month, 'day_of_week' : day_of_week,
            'duration' : duration, 'campaign' : campaign, 'pdays' : pdays,
            'previous' : previous, 'poutcome' : poutcome, 'emp_var_rate' : emp_var_rate,
            'cons_price_idx' : cons_price_idx, 'cons_conf_idx' : cons_conf_idx,
            'euribor3m' : euribor3m, 'nr_employed' : nr_employed}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


# Display input parameters
st.subheader('User Input parameters')
st.write(df)

# Modify df to have dummies such that it can be fed into the loaded model.
#### Dummy variables made from the DF above will ONLY include columns for the
#### parameters that were selected. The function below creates dummies from user
#### input df AND adds the other required columns to fit the model input shape.
def mod_df(df):
    all_categ_dummy_cols = pd.DataFrame(columns = ['job_admin.', 'job_blue-collar', 'job_entrepreneur',
        'job_housemaid', 'job_management', 'job_retired', 'job_self-employed',
        'job_services', 'job_student', 'job_technician', 'job_unemployed',
        'job_unknown', 'marital_divorced', 'marital_married', 'marital_single',
        'marital_unknown', 'education_basic', 'education_high.school', 'education_illiterate',
        'education_professional.course', 'education_university.degree',
        'education_unknown', 'default_no', 'default_unknown', 'default_yes',
        'housing_no', 'housing_unknown', 'housing_yes', 'loan_no',
        'loan_unknown', 'loan_yes', 'contact_cellular', 'contact_telephone',
        'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun',
        'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
        'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu',
        'day_of_week_tue', 'day_of_week_wed', 'poutcome_failure',
        'poutcome_nonexistent', 'poutcome_success'], data = [[0] * 51])

    num_vars = ['age','duration','campaign','pdays','previous','emp_var_rate','cons_price_idx',
        'cons_conf_idx','euribor3m','nr_employed']

    column_order = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate',
       'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'marital_unknown', 'education_basic', 'education_high.school',
       'education_illiterate', 'education_professional.course',
       'education_university.degree', 'education_unknown', 'default_no',
       'default_unknown', 'default_yes', 'housing_no', 'housing_unknown',
       'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes',
       'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug',
       'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may',
       'month_nov', 'month_oct', 'month_sep', 'day_of_week_fri',
       'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',
       'day_of_week_wed', 'poutcome_failure', 'poutcome_nonexistent',
       'poutcome_success']

    df2 = pd.get_dummies(df, drop_first = False)
    cols_to_append = all_categ_dummy_cols.drop(columns =df2.drop(columns = num_vars).columns)
    df2[cols_to_append.columns] = cols_to_append
    df2 = df2[column_order]

    return pd.DataFrame(df2)

# Construct new dataframe
df2 = mod_df(df)

# Load model
filename = 'logreg_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Make predictions on df2 inputs
prediction = loaded_model.predict(df2)
prediction_proba = loaded_model.predict_proba(df2)
prob_print = pd.DataFrame(
            columns = ["Will subscribe","Will not subcribe"],
            data = [[str(round(prediction_proba[0,1]*100,2))+'%',
            str(round(prediction_proba[0,0]*100,2))+'%']])

# Printout results
st.subheader('Prediction Results')
if prediction[0] == 1:
    st.write('Client WILL subcribe to term deposit.')
else:
    st.write('Client WILL NOT subscribe to term deposit.')

st.write('Prediction Probability')
st.write(prob_print)
