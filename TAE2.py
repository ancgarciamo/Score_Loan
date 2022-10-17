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

with lzma.open('df_scorecard.pkl', 'rb') as file:
   A = pickle.load(file)

with lzma.open('woe_transform.pkl', 'rb') as file:
   C = pickle.load(file)

X_test_woe_transformed = C.fit_transform(dfxxx)
X_test_woe_transformed.insert(0, 'Intercept', 1)
scorecard_scores = A['Score - Final']
X_test_woe_transformed = pd.concat([X_test_woe_transformed, pd.DataFrame(dict.fromkeys(ref_categories, [0] * len(X_test_woe_transformed)),
                                                                         index = X_test_woe_transformed.index)], axis = 1)
# Need to reshape scorecard_scores so that it is (102,1) to allow for matrix dot multiplication
scorecard_scores = scorecard_scores.values.reshape(102, 1)
y_scores = X_test_woe_transformed.dot(scorecard_scores)
print("Su puntaje es :" + str(y_scores[0][0]))

import streamlit as st


def main():
   gradeA = 0
   gradeB = 0
   gradeC = 0
   gradeD = 0
   gradeE = 0
   gradeF = 0
   gradeG = 0
   verification_status_info="placeholder"
   verification_status1 = 0
   verification_status2 = 0
   verification_status3 = 0
   home_ownership_info="placeholder"
   home_ownership_mortage = 0
   home_ownership_rent = 0
   home_ownership_own = 0
   home_ownership_other = 0
   home_ownership_none = 0
   purpose_info="placeholder"
   purpose_car = 0
   purpose_credit = 0
   purpose_debt = 0
   purpose_education = 0
   purpose_home = 0
   purpose_house = 0
   purpose_major = 0
   purpose_medical = 0
   purpose_moving = 0
   purpose_other = 0
   purpose_energy = 0
   purpose_business = 0
   purpose_vacation = 0
   purpose_wedding = 0

   st.title("Encuesta de credito")
   termino=st.radio("Escoja el termino",(36,60))
   int_ratee=st.number_input('Ingrese la tasa de interes del prestamo')
   grado=st.radio("Escoja el grado",('A','B','C','D','E','F','G'))
   emp_l=st.slider("Numero de años trabajando",0,10,1)
   ingresos_anuales=st.number_input('Ingrese la cantidad de sus ingresos anuales (En dolares)')
   casa=st.radio("Cual es su estado de posesion de vivienda",('Hipoteca','No posee','Otro','Casa propia','Arrendado'))
   ver_sta=st.radio("ss",('No verificado','Fuente verificada','Verificado'))
   proposito=st.radio("Proposito del credito ",
                      ('Carro','Tarjeta de credito','Consolidacion de la deuda','Educativo',
                       'Mejora de la vivienda','Casa','Compra grande',
                       'Motivos medicos','Mudanza','Otro','Energia renovable',
                       'Pequeño negocio','Vacaciones','Boda'))

   dtii = st.number_input("Ingrese su dti (Una relación calculada utilizando los pagos de deuda mensuales totales del prestatario sobre las obligaciones de deuda total, excluyendo la hipoteca y el préstamo LC solicitado, dividido por los ingresos mensuales autoinformados del prestatario.)")
   inq_6meses = st.number_input("Ingrese el número de consultas en los últimos 6 meses (excluyendo consultas de automóviles e hipotecas")
   revol = st.number_input("Cantidad del credito que el prestatartio esta utilizando en relacion con todo el credito giratorio disponible")
   totalacc = st.number_input("Ingrese el número total de líneas de crédito actualmente en el archivo de crédito del prestatario")
   out_pri = st.number_input("Capital restante pendiente por el monto total financiado")
   pago_total = st.number_input("Ingrese el total de  pagos hasta la fecha")
   pago_reciente = st.number_input("Ingrese el total los intereses hasta la fecha")
   ultimo_pago = st.number_input("Ingrese el último monto total de pago recibido")
   total_cur = st.number_input("Ingrese su saldo actual total de todas las cuentas")
   total_rev_lims = st.number_input("Límite total de crédito/crédito de alto aumento giratorio")
   meses_cr = st.number_input("Numero de meses desde que se abrió la línea de crédito más temprana del prestatario")
   meses_issue = st.number_input("Numero de meses desde que el prestamo fue hecho")
   meses_ultimo_pago = st.number_input("Numero de  meses desde efectuo su ultimo pago?")
   meses_ultimo_credito = st.number_input("Numero de meses desde que LC obtuvo crédito por este préstamo")

   if grado=='A':
      gradeA=1
   elif grado=='B':
      gradeB=1
   elif grado=='C':
      gradeC=1
   elif grado=='D':
      gradeD=1
   elif grado=='E':
      gradeE=1
   elif grado=='F':
      gradeF=1
   elif grado=='G':
      gradeG=1


   if ver_sta=='No verificado':
      verification_status1=1
      verification_status_info='Not Verified'
   elif ver_sta=='Fuente verificada':
      verification_status2=1
      verification_status_info = 'Source Verified'
   elif ver_sta=='Verificado':
      verification_status2=1
      verification_status_info = 'Verified'

   if casa=='Hipoteca':
      home_ownership_mortage=1
      home_ownership_info='MORTGAGE'
   elif casa=='No posee':
      home_ownership_none=1
      home_ownership_info='NONE'
   elif casa == 'Otro':
      home_ownership_other = 1
      home_ownership_info = 'OTHER'
   elif casa == 'Casa propia':
      home_ownership_own = 1
      home_ownership_info = 'OWN'
   elif casa == 'Arrendado':
      home_ownership_rent = 1
      home_ownership_info = 'RENT'


   if proposito=='Carro':
      purpose_info='car'
      purpose_car=1
   elif proposito=='Tarjeta de credito':
      purpose_credit=1
      purpose_info='credit_card'
   elif proposito == 'Consolidacion de la deuda':
      purpose_debt = 1
      purpose_info = 'debt_consolidation'
   elif proposito == 'Educativo':
      purpose_education = 1
      purpose_info = 'educational'
   elif proposito == 'Mejora de la vivienda':
      purpose_home = 1
      purpose_info = 'home_improvement'
   elif proposito == 'Casa':
      purpose_house = 1
      purpose_info = 'house'
   elif proposito == 'Compra grande':
      purpose_major = 1
      purpose_info = 'major_purchase'
   elif proposito == 'Motivos medicos':
      purpose_medical = 1
      purpose_info = 'medical'
   elif proposito == 'Mudanza':
      purpose_moving = 1
      purpose_info = 'moving'
   elif proposito == 'Otro':
      purpose_other = 1
      purpose_info = 'other'
   elif proposito == 'Energia renovable':
      purpose_energy = 1
      purpose_info = 'renewable_energy'
   elif proposito == 'Pequeño negocio':
      purpose_business = 1
      purpose_info = 'small_business'
   elif proposito == 'Vacaciones':
      purpose_vacation = 1
      purpose_info = 'vacation'
   elif proposito == 'Boda':
      purpose_wedding = 1
      purpose_info = 'wedding'

   data_predictiva = [{'term': termino,
                       'int_rate': int_ratee,
                       'grade': grado,
                       'emp_length': emp_l,
                       'home_ownership': home_ownership_info,
                       'annual_inc': ingresos_anuales,
                       'verification_status': verification_status_info,
                       'purpose': purpose_info,
                       'dti': dtii,
                       'inq_last_6mths': inq_6meses,
                       'revol_util': revol,
                       'total_acc': totalacc,
                       'out_prncp': out_pri,
                       'total_pymnt': pago_total,
                       'total_rec_int': pago_reciente,
                       'last_pymnt_amnt': ultimo_pago,
                       'tot_cur_bal': total_cur,
                       'total_rev_hi_lim': total_rev_lims,
                       'mths_since_earliest_cr_line': meses_cr,
                       'mths_since_issue_d': meses_issue,
                       'mths_since_last_pymnt_d': meses_ultimo_pago,
                       'mths_since_last_credit_pull_d': meses_ultimo_credito,
                       'grade:A': gradeA,
                       'grade:B': gradeB,
                       'grade:C': gradeC,
                       'grade:D': gradeD,
                       'grade:E': gradeE,
                       'grade:F': gradeF,
                       'grade:G': gradeG,
                       'home_ownership:MORTGAGE': home_ownership_mortage,
                       'home_ownership:NONE': home_ownership_none,
                       'home_ownership:OTHER': home_ownership_other,
                       'home_ownership:OWN': home_ownership_own,
                       'home_ownership:RENT': home_ownership_rent,
                       'verification_status:Not Verified': verification_status1,
                       'verification_status:Source Verified': verification_status2,
                       'verification_status:Verified': verification_status2,
                       'purpose:car': purpose_car,
                       'purpose:credit_card': purpose_credit,
                       'purpose:debt_consolidation': purpose_debt,
                       'purpose:educational': purpose_education,
                       'purpose:home_improvement': purpose_home,
                       'purpose:house': purpose_house,
                       'purpose:major_purchase': purpose_major,
                       'purpose:medical': purpose_medical,
                       'purpose:moving': purpose_moving,
                       'purpose:other': purpose_other,
                       'purpose:renewable_energy': purpose_energy,
                       'purpose:small_business': purpose_business,
                       'purpose:vacation': purpose_vacation,
                       'purpose:wedding': purpose_wedding
                       }]
   dataframe_predecir = pd.DataFrame(data_predictiva)

   X_test_woe_transformed = C.fit_transform(dataframe_predecir)
   X_test_woe_transformed.insert(0, 'Intercept', 1)
   scorecard_scores = A['Score - Final']
   X_test_woe_transformed = pd.concat(
       [X_test_woe_transformed, pd.DataFrame(dict.fromkeys(ref_categories, [0] * len(X_test_woe_transformed)),
                                             index=X_test_woe_transformed.index)], axis=1)

   scorecard_scores = scorecard_scores.values.reshape(102, 1)
   y_scores = X_test_woe_transformed.dot(scorecard_scores)
   st.sidebar.title("Su puntaje es : " + str(y_scores[0][0]))
   puntaje_promedio=549.40798
   st.sidebar.title("El puntaje promedio de los usuarios es de "+str(puntaje_promedio))
   if y_scores[0][0]>=puntaje_promedio:
       st.sidebar.title("Su puntaje es superior o igual a la media")
   else:
       st.sidebar.title("Su puntaje es inferior a la media")









if __name__=="__main__":
   main()