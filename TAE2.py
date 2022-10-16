import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
ref_categories = ['mths_since_last_credit_pull_d:>75', 'mths_since_issue_d:>122', 'mths_since_earliest_cr_line:>434', 'total_rev_hi_lim:>79,780',
                  'total_rec_int:>7,260', 'total_pymnt:>25,000', 'out_prncp:>15,437', 'revol_util:>1.0', 'inq_last_6mths:>4', 'dti:>35.191',
                  'annual_inc:>150K', 'int_rate:>20.281', 'term:60', 'purpose:major_purch__car__home_impr', 'verification_status:Not Verified',
                  'home_ownership:MORTGAGE', 'grade:G']

data1=[{'term':36,
   'int_rate':14.64,
   'grade':'C',
   'emp_length':10.0,
   'home_ownership' :'OWN',
   'annual_inc':50000.0,
   'verification_status':'Source Verified',
   'purpose':'home_improvement',
   'dti':19.11,
   'inq_last_6mths':4.0,
   'revol_util':49.0,
   'total_acc':32.0,
   'out_prncp':0.0,
   'total_pymnt':2090.92,
   'total_rec_int':290.92,
   'last_pymnt_amnt':1159.57,
   'tot_cur_bal':107437.0,
   'total_rev_hi_lim':2450.0,
   'mths_since_earliest_cr_line':347.0,
   'mths_since_issue_d':75.0,
   'mths_since_last_pymnt_d':59.0,
   'mths_since_last_credit_pull_d':55.0,
   'grade:A':0,
   'grade:B':0,
   'grade:C':1,
   'grade:D':0,
   'grade:E':0,
   'grade:F':0,
   'grade:G':0,
   'home_ownership:MORTGAGE':0,
   'home_ownership:NONE':0,
   'home_ownership:OTHER':0,
    'home_ownership:OWN':1,
   'home_ownership:RENT':0,
   'verification_status:Not Verified':0,
   'verification_status:Source Verified':1,
   'verification_status:Verified':0,
   'purpose:car':0,
   'purpose:credit_card':0,
   'purpose:debt_consolidation':0,
   'purpose:educational':0,
   'purpose:home_improvement':1,
   'purpose:house':0,
   'purpose:major_purchase':0,
   'purpose:medical':0,
   'purpose:moving':0,
   'purpose:other':0,
   'purpose:renewable_energy':0,
   'purpose:small_business':0,
   'purpose:vacation':0,
   'purpose:wedding':0
}]
data2=[{'term':36,
   'int_rate':7.12,
   'grade':'A',
   'emp_length':1.0,
   'home_ownership' :'RENT',
   'annual_inc':63000.0,
   'verification_status':'Not Verified',
   'purpose':'debt_consolidation',
   'dti':7.98,
   'inq_last_6mths':0.0,
   'revol_util':7.1,
   'total_acc':24.0,
   'out_prncp':2992.57,
   'total_pymnt':3526.4,
   'total_rec_int':518.97,
   'last_pymnt_amnt':185.6,
   'tot_cur_bal':4413.0,
   'total_rev_hi_lim':27000.0,
   'mths_since_earliest_cr_line':265.0,
   'mths_since_issue_d':74.0,
   'mths_since_last_pymnt_d':55.0,
   'mths_since_last_credit_pull_d':55.0,
   'grade:A':1,
   'grade:B':0,
   'grade:C':0,
   'grade:D':0,
   'grade:E':0,
   'grade:F':0,
   'grade:G':0,
   'home_ownership:MORTGAGE':0,
   'home_ownership:NONE':0,
   'home_ownership:OTHER':0,
    'home_ownership:OWN':0,
   'home_ownership:RENT':1,
   'verification_status:Not Verified':1,
   'verification_status:Source Verified':0,
   'verification_status:Verified':0,
   'purpose:car':0,
   'purpose:credit_card':0,
   'purpose:debt_consolidation':1,
   'purpose:educational':0,
   'purpose:home_improvement':0,
   'purpose:house':0,
   'purpose:major_purchase':0,
   'purpose:medical':0,
   'purpose:moving':0,
   'purpose:other':0,
   'purpose:renewable_energy':0,
   'purpose:small_business':0,
   'purpose:vacation':0,
   'purpose:wedding':0
}]

dfxxx = pd.DataFrame(data2)
import lzma

import dill as pickle
with lzma.open('prediccion_compressed.pkl', 'rb') as file:
    B = pickle.load(file)

S=B.predict(dfxxx)
print(S[0])

import streamlit as st

gradeA=0
gradeB=0
gradeC=1
gradeD=0
gradeE=0
gradeF=0
gradeG=0
verification_status1=0
verification_status2=1
verification_status3=0
home_ownership_mortage=0
home_ownership_rent=0
home_ownership_own=0
home_ownership_other=1
home_ownership_none=0
purpose_car=0
purpose_credit=0
purpose_debt=0
purpose_education=0
purpose_home=1
purpose_house=0
purpose_major=0
purpose_medical=0
purpose_moving=0
purpose_other=0
purpose_energy=0
purpose_business=0
purpose_vacation=0
purpose_wedding=0

def main():


   st.title("Encuesta de credito")
   termino=st.radio("Escoja el termino",(36,60))
   int_ratee=st.number_input('Ingrese el numero')
   grado=st.radio("Escoja el grado",('A','B','C','D','E','F','G'))
   emp_l=st.slider("Numero de años trabajando",0,10,1)
   ingresos_anuales=st.number_input('Ingrese la cantidad de sus ingresos anuales (En dolares)')
   dtii=st.number_input("meta su dti ?")
   casa=st.radio("Cual es su estado de posesion de vivienda",('Hipoteca','No posee','Otro','Casa propia','Arrendado'))
   ver_sta=st.radio("ss",('No verificado','Fuente verificada','Verificado'))
   proposito=st.radio("Proposito del credito ",
                      ('Carro','Tarjeta de credito','Consolidacion de la deuda','Educativo',
                       'Mejora de la vivienda','Casa','Compra grande',
                       'Motivos medicos','Mudanza','Otro','Energia renovable',
                       'Pequeño negocio','Vacaciones','Boda'))
   inq_6meses = st.number_input("meta su dti1 ?")
   revol = st.number_input("meta su dti2 ?")
   totalacc = st.number_input("meta su dti3 ?")
   out_pri = st.number_input("meta su dti4 ?")
   pago_total = st.number_input("meta su dti 5?")
   pago_reciente = st.number_input("meta su dti6 ?")
   ultimo_pago = st.number_input("meta su dti7 ?")
   total_cur = st.number_input("meta su dt8i ?")
   total_rev_lims=st.number_input("ss")
   meses_cr = st.number_input("meta su dti9 ?")
   meses_issue = st.number_input("meta su dti p?")
   meses_ultimo_pago = st.number_input("meta su dtia ?")
   meses_ultimo_credito = st.number_input("meta su dsti ?")

   S = B.predict(dfxxx)
   st.title(str(S[0]))




if __name__=="__main__":
   main()