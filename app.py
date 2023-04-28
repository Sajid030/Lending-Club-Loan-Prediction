# Importing necessary libraries
import pickle
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings(action='ignore')

# Importing the models
df = pickle.load(open('model/dataset.pkl','rb'))
model = load_model('model/my_model.h5')

# Initialize the session state variable
if 'loan_details' not in st.session_state:
    st.session_state['loan_details'] = {
        'loan_amnt': 2400,
        'term': 36,
        'int_rate': 15.96,
        'installment': 84.33,
        'grade': 'B',
        'sub_grade': 'C5',
        'last_pymnt_amnt': 649.91,
    }

# Initialize the session state variable
if 'borrower_details' not in st.session_state:
    st.session_state['borrower_details'] = {
        'emp_length': '10+',
        'home_ownership': 'RENT',
        'annual_inc': 12252.0,
        'verification_status': 'Verified',
        'purpose': 'small_business',
        'dti': 8.72,
        'delinq_2yrs': 0.0,
        'inq_last_6mths': 2.0,
        'open_acc': 2.0,
        'pub_rec': 0.0,
        'revol_bal': 2956.0	,
        'revol_util': 98.5,
        'total_acc': 10.0
    }

# Create a container for the sidebar
sidebar = st.sidebar.container()

# Add widgets to the sidebar
with sidebar:
    st.header('Navigation')
    nav_item = st.radio('Go to', ('Home', 'Loan Details', 'Borrower Details'))

# Create a container for the main content
content = st.container()

# Add content to the main container
with content:
    if nav_item == 'Home':
      # st.title('Lending-Club Loan Prediction App')
      st.markdown("<h1 style='font-size: 35px;'><u>Lending-Club Loan Prediction App</u></h1>", unsafe_allow_html=True)
      st.markdown("<h4>Using the power of Artificial Neural Networks to make informed financial decisions</h4>", unsafe_allow_html=True)
      message = """
      **Welcome to the Loan Status Predictor App!🥳**\n
      With just few inputs, our model can predict whether a loan is Fully Paid or Charged Off.\n
      To get started, simply fill in the Loan Details and Borrower Details sections accessible from the left-side navigation bar. 
      Don't worry about missing inputs as default values have been set for each feature🤙. 
      However, for optimal results, we recommend filling in all inputs.
      Once you've entered all necessary details, hit the Predict button available in all three sections to see the prediction.
      We hope this app helps you make informed financial decisions.\n
      Thank you for using our Loan Status Predictor App!🤞
      """
      st.markdown(message)
      
    elif nav_item == 'Loan Details':
      st.title('Loan Details')
      st.write('Enter Loan details here.')
      
      loan_amnt = st.number_input('Loan Amount', min_value=0, step=1000)
      term = st.selectbox('Term', [36,60])
      int_rate = st.number_input('Interest Rate', min_value=0.0, step=5.0, format="%.2f")
      installment = st.number_input('Monthly Payment', min_value=0.0, step=50.0, format="%.2f")
      grade = st.selectbox('Grade', np.sort(df['grade'].unique(), kind='mergesort'))
      sub_grade = st.selectbox('Sub-Grade', np.sort(df['sub_grade'].unique(), kind='mergesort'))
      last_pymnt_amnt = st.number_input('Loan Payment Amount', min_value=0.0, step=100.0, format="%.2f")
      
      st.session_state['loan_details'] = {
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'installment': installment,
        'grade': grade,
        'sub_grade': sub_grade,
        'last_pymnt_amnt': last_pymnt_amnt,
        }


    elif nav_item == 'Borrower Details':
      st.title('Borrower Details')
      st.write('Enter Borrower details here.')
      
      emp_length = st.selectbox('Employment Length(in year)', ['<1', '1', '2', '3',
                                                '4', '5', '6', '7',
                                                '8', '9', '10+'])
      home_ownership = st.selectbox('Home Ownership', np.sort(df['home_ownership'].unique(),kind='mergesort'))
      # Preprocess input to match format used in model
      home_ownership = home_ownership.replace(" ", "_")
      annual_inc = st.number_input('Annual Income', min_value=0.0, step=1000.0)
      verification_status = st.selectbox('Income Verification Status', np.sort(df['verification_status'].unique(),kind='mergesort'))
      purpose = st.selectbox('Purpose for the loan',np.sort(df['purpose'].unique(),kind='mergesort'))
      
      dti = st.number_input('Debt-to-Income Ratio (DTI)', min_value=0.0, step=5.0)
      delinq_2yrs = st.number_input('Delinquency count in the past 2 years', min_value=0.0, step=1.0, max_value=50.0)
      inq_last_6mths = st.number_input('Credit inquiry count in the last 6 months:', min_value=0.0, step=1.0, max_value=30.0)
      open_acc = st.number_input('Open credit account count', min_value=0.0, step=1.0, max_value=500.0)
      pub_rec = st.number_input('Derogatory public record count', min_value=0.0, step=1.0, max_value=30.0)
      revol_bal = st.number_input('Revolving balance', min_value=0.0, step=1000.0)
      revol_util = st.number_input("Revolving credit utilization(%):", min_value=0.0, step=5.0)
      total_acc = st.number_input("Enter your total number of credit accounts:", min_value=0.0, value=0.0, step=1.0)
      
      st.session_state['borrower_details'] = {
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'verification_status': verification_status,
        'purpose': purpose,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'inq_last_6mths': inq_last_6mths,
        'open_acc': open_acc,
        'pub_rec': pub_rec,
        'revol_bal': revol_bal,
        'revol_util': revol_util,
        'total_acc': total_acc
        }

if st.button('Predict'):
    # Access loan details and borrower details from session state
    loan_details = st.session_state.get('loan_details')
    borrower_details = st.session_state.get('borrower_details')
    
    loan_amnt = loan_details['loan_amnt']
    term = loan_details['term']
    int_rate = loan_details['int_rate']
    installment = loan_details['installment']
    grade = loan_details['grade']
    # Create a dictionary that maps each grade to its corresponding number
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    # Map the selected grade to its corresponding number
    grade = grade_map.get(grade, -1)
    
    sub_grade = loan_details['sub_grade']
    # Create a dictionary that maps each sub_grade to its corresponding number
    sub_grade_map= {'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4, 
                    'B1': 5, 'B2': 6, 'B3': 7, 'B4': 8, 'B5': 9, 
                    'C1': 10, 'C2': 11, 'C3': 12, 'C4': 13, 'C5': 14, 
                    'D1': 15, 'D2': 16, 'D3': 17, 'D4': 18, 'D5': 19, 
                    'E1': 20, 'E2': 21, 'E3': 22, 'E4': 23, 'E5': 24, 
                    'F1': 25, 'F2': 26, 'F3': 27, 'F4': 28, 'F5': 29, 
                    'G1': 30, 'G2': 31, 'G3': 32, 'G4': 33, 'G5': 34}
    # Map the selected sub_grade to its corresponding number
    sub_grade = sub_grade_map.get(sub_grade, -1)
    
    last_pymnt_amnt = float(loan_details['last_pymnt_amnt'])
    
    emp_length = borrower_details['emp_length']
    # Create a dictionary that maps each emp_length to its corresponding number
    emp_length_map= {'<1': 0,'1': 1,'2': 2,'3': 3,
                    '4': 4, '5': 5,'6': 6,'7': 7,
                    '8': 8,'9': 9, '10+': 10}
    # Map the selected sub_grade to its corresponding number
    emp_length = emp_length_map.get(emp_length, -1)
    
    home_ownership = borrower_details['home_ownership']
    annual_inc = borrower_details['annual_inc']
    verification_status = borrower_details['verification_status']
    purpose = borrower_details['purpose']
    dti = borrower_details['dti']
    delinq_2yrs = borrower_details['delinq_2yrs']
    inq_last_6mths = borrower_details['inq_last_6mths']
    open_acc = borrower_details['open_acc']
    pub_rec = borrower_details['pub_rec']
    revol_bal = borrower_details['revol_bal']
    revol_util = borrower_details['revol_util']
    total_acc = borrower_details['total_acc']
    
    if home_ownership =='MORTGAGE':
      home_ownership_MORTGAGE=1
      home_ownership_NONE=0
      home_ownership_OTHER=0
      home_ownership_OWN=0
      home_ownership_RENT=0
    elif home_ownership =='NONE':
      home_ownership_MORTGAGE=0
      home_ownership_NONE=1
      home_ownership_OTHER=0
      home_ownership_OWN=0
      home_ownership_RENT=0
    elif home_ownership =='OTHER':
      home_ownership_MORTGAGE=0
      home_ownership_NONE=0
      home_ownership_OTHER=1
      home_ownership_OWN=0
      home_ownership_RENT=0
    elif home_ownership =='OWN':
      home_ownership_MORTGAGE=0
      home_ownership_NONE=0
      home_ownership_OTHER=0
      home_ownership_OWN=1
      home_ownership_RENT=0
    elif home_ownership =='RENT':
      home_ownership_MORTGAGE=0
      home_ownership_NONE=0
      home_ownership_OTHER=0
      home_ownership_OWN=0
      home_ownership_RENT=1
    else:
      home_ownership_MORTGAGE=0
      home_ownership_NONE=0
      home_ownership_OTHER=0
      home_ownership_OWN=0
      home_ownership_RENT=0
      
    if verification_status =='Source Verified':
      verification_status_Source_Verified=1
      verification_status_Verified=0
    elif verification_status =='Verified':
      verification_status_Source_Verified=0
      verification_status_Verified=1
    else:
      verification_status_Source_Verified=0
      verification_status_Verified=0
    
    if purpose =='credit_card':
      purpose_credit_card=1
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='debt_consolidation':
      purpose_credit_card=0
      purpose_debt_consolidation=1
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='educational':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=1
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='home_improvement':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=1
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='house':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=1
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='major_purchase':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=1
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='medical':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=1
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='moving':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=1
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='other':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=1
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='renewable_energy':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=1
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='small_business':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=1
      purpose_vacation=0
      purpose_wedding=0
    elif purpose =='vacation':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=1
      purpose_wedding=0
    elif purpose =='wedding':
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=1
    else:
      purpose_credit_card=0
      purpose_debt_consolidation=0
      purpose_educational=0
      purpose_home_improvement=0
      purpose_house=0
      purpose_major_purchase=0
      purpose_medical=0
      purpose_moving=0
      purpose_other=0
      purpose_renewable_energy=0
      purpose_small_business=0
      purpose_vacation=0
      purpose_wedding=0
    
    query = np.array([[loan_amnt, term, int_rate, installment, grade, sub_grade, emp_length, annual_inc, dti, delinq_2yrs,
              inq_last_6mths, open_acc,	pub_rec, revol_bal,	revol_util,	total_acc, last_pymnt_amnt, home_ownership_MORTGAGE,
              home_ownership_NONE,	home_ownership_OTHER, home_ownership_OWN, home_ownership_RENT, verification_status_Source_Verified,
              verification_status_Verified,	purpose_credit_card, purpose_debt_consolidation, purpose_educational,
              purpose_home_improvement,	purpose_house, purpose_major_purchase, purpose_medical,	purpose_moving,
              purpose_other, purpose_renewable_energy, purpose_small_business, purpose_vacation, purpose_wedding]])
    query = query.reshape(1,37)
    
    prediction=int(model.predict(query)[0])
    st.title("Prediction:")
    if prediction ==0:
      st.subheader("Fully Paid🙂")
    else:
      st.subheader("Charged Off☹️")
