import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Software Bug Predictor",
                   page_icon="📵", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
X=pd.read_csv("bug_detection.csv")
st.write("""
# Software Bug Prediction

This app predicts occurrence of a bug in a software application
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
pred= pickle.load(open('bug_clf.pkl', 'rb'))
def user_input_features():
    with st.form("Form1"):
        loc = st.sidebar.number_input("Lines of code [loc]: ",format="%.5f")
        v_g = st.sidebar.number_input("Cyclomatic complexity [v(g)]: ",format="%.5f")
        ev = st.sidebar.number_input("Essential complexity [ev(g)]: ",format="%.5f")
        iv = st.sidebar.number_input("Design complexity [iv(g)]: ",format="%.5f")
        n = st.sidebar.number_input("Total operators and operands [n]: ",format="%.5f")
        v = st.sidebar.number_input("Volume [v]:",format="%.5f")
        l = st.sidebar.number_input("Program length [l]:",format="%.5f")
        d = st.sidebar.number_input("Difficulty [d]: ",format="%.5f")
        i = st.sidebar.number_input("Intelligence [i]:",format="%.5f")
        e = st.sidebar.number_input("Effort [e]:",format="%.5f")
        b = st.sidebar.number_input("Delivered bugs [b]:",format="%.5f")
        t = st.sidebar.number_input("Time estimator [t]:",format="%.5f")
        lOCode = st.sidebar.number_input("Line count [lOCode]:",format="%.5f")
        lOBlank = st.sidebar.number_input("Count of blank lines [lOBlank]:",format="%.5f")
        lOComment = st.sidebar.number_input("Count of lines of comments [lOComment]:",format="%.5f")
        uniq_Op = st.sidebar.number_input("Unique operators [uniq_Op]:",format="%.5f")
        uniq_Opnd = st.sidebar.number_input("Unique operands [uniq_Opnd]:",format="%.5f")
        total_Op = st.sidebar.number_input("Total operators [total_Op]:",format="%.5f")
        total_Opnd = st.sidebar.number_input("Total operands [total_Opnd]:",format="%.5f")
        branchCount = st.sidebar.number_input("Branch count [branchCount]:",format="%.5f")
        submit_val = st.form_submit_button("Predict Defect Possibility")

    data = {'LOC': loc,
            'v_g': v_g,
            'ev': ev,
            'iv': iv,
            'n': n,
            'v': v,
            'l': l,
            'd': d,
            'i': i,
            'e': e,
            'b': b,
            't': t,
            'lOCode': lOCode,
            'lOBlank':lOBlank,
            'lOComment':lOComment,
            'uniq_Op':uniq_Op,
            'uniq_Opnd':uniq_Opnd,
            'total_Op':total_Op,
            'total_Opnd':total_Opnd,
            'branchCount':branchCount}
    features = pd.DataFrame(data, index=[0])
    if submit_val:
        #attributes = np.array(df)

        attributes = np.array([loc,v_g,ev,iv,n,v,l,d,i,e,b,t,lOCode,lOComment,lOBlank,uniq_Op,uniq_Opnd,total_Op,total_Opnd,branchCount])
        print("attributes value")
        status = pred.predict(attributes.reshape(1, -1))
        st.write(status)
        if status:
            st.error("Defect Probable")
        else:
            st.success("No defect")
    return features


df_ft = user_input_features()

# Main Panel
prediction_proba = pred.predict_proba(df_ft)
st.subheader('Prediction Probability')
st.write(prediction_proba)
# Print specified input parameters
st.header('Specified Input parameters')
st.write(df_ft)
st.write('---')


