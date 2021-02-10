#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:56:18 2020

@author: anushidoshi
"""

import streamlit as st
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import apyori as ap
from apyori import apriori #Apriori Algorithm
import mlxtend as ml
from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
import math
import os
import altair as alt
import pickle
with open('accuracy_df.pickle', 'rb') as f:
    el_df=pickle.load(f)
    rfc_df=pickle.load(f)
    lsvm_df=pickle.load(f)
    nlsvm_df=pickle.load(f)
    dnn2_df=pickle.load(f)
    dt_df=pickle.load(f)
    lr_df = pickle.load(f)
with open('bestmodel.pickle', 'rb') as f:
    bestmodel_df = pickle.load(f)


def main():
    eda_data = pd.DataFrame(pd.read_csv("PreprocessedEDA_data.csv"))
    arm_data = pd.DataFrame(pd.read_csv("PreprocessedARM_data.csv"))
    rf_data = rfc_df
    el_data = el_df
    lsvm_data = lsvm_df
    nlsvm_data = nlsvm_df
    dt_data = dt_df
    dnn2_data = dnn2_df
    lr_data = lr_df
    
    page = st.sidebar.selectbox("Choose a page", ['Exploratory Data Analysis', 'Classification Models','Deep Learning','Association Rule Mining', 'Best Model Performances'])
    
    if page == 'Exploratory Data Analysis':
        
        st.title('Explore the Diabetic Patient Readmission Dataset')
        sns.countplot(eda_data['readmitted']).set_title('Distribution of readmitted patients')
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # time in hospital VS readmission
        fig, ax = plt.subplots(figsize = (13,7),)
        sns.kdeplot(eda_data.loc[(eda_data['readmitted'] == 0), 'time_in_hospital'], color = 'b', shade = True, label = 'Not Readmitted')
        sns.kdeplot(eda_data.loc[(eda_data['readmitted'] == 1), 'time_in_hospital'] , color = 'r',shade = True, label = 'Readmitted')
        ax.set(xlabel = 'Time in Hospital', ylabel = 'Frequency')
        plt.title('Time in Hospital VS. Readmission')
        st.pyplot(fig)
        # age VS readmission
        fig = plt.figure(figsize = (15, 10))
        sns.countplot(y = eda_data['age'], hue = eda_data['readmitted']).set_title('Age of Patient VS. Readmission')
        st.pyplot()
        # race VS readmission
        fig = plt.figure(figsize = (8, 8))
        sns.countplot(y = eda_data['race'], hue = eda_data['readmitted'])
        st.pyplot()
        # number of medication VS readmission
        fig = plt.figure(figsize = (8, 8))
        sns.barplot(x = eda_data['readmitted'], 
                    y = eda_data['num_medications']).set_title("Number of medication used VS. Readmission")
        st.pyplot()
        # gender VS readmission
        fig=plt.figure(figsize = (8, 8))
        sns.countplot(eda_data['gender'], hue = eda_data['readmitted']).set_title("Gender VS. Readmission")
        st.pyplot()
        # change in medication VS readmission
        fig = plt.figure(figsize = (8, 8))
        sns.countplot(eda_data['change'], 
                      hue = eda_data['readmitted']).set_title('Change of Medication VS. Readmission')
        st.pyplot()
        # service utilization VS readmission
        fig = plt.figure(figsize = (8, 8))
        sns.barplot( y = eda_data['service_utilization'], 
                    x = eda_data['readmitted']).set_title('Service Utilization VS. Readmission')
        st.pyplot()
        # glucose serum test result VS readmission
        fig = plt.figure(figsize = (8, 8))
        sns.countplot(y = eda_data['max_glu_serum'], 
                      hue = eda_data['readmitted']).set_title('Glucose test serum test result VS. Readmission')
        st.pyplot()
        # A1C test result VS readmission
        fig = plt.figure(figsize = (8, 8))
        sns.countplot(y = eda_data['A1Cresult'],hue = eda_data['readmitted']).set_title('A1C test result VS. Readmission')
        st.pyplot()
        # number of lab procedures VS readmission
        fig, ax = plt.subplots(figsize = (15,6),)
        ax = sns.kdeplot(eda_data.loc[(eda_data['readmitted'] == 0), 'num_lab_procedures'] , color = 'b', shade = True, label = 'Not readmitted')
        ax = sns.kdeplot(eda_data.loc[(eda_data['readmitted'] == 1), 'num_lab_procedures'] , color = 'r', shade = True, label = 'readmitted')
        ax.set(xlabel = 'Number of lab procedure', ylabel = 'Frequency')
        plt.title('Number of lab procedure VS. Readmission')
        st.pyplot(fig)
    
    
    elif page == 'Classification Models':
        add_selectbox = st.sidebar.selectbox("Choose a Model",("Ensemble Learning","Random Forest Classifier", "Linear SVM","Non-Linear SVM","Decision Tree","Logistic Regression"))
        st.title("Classification Model:")
        #Attrition status
        if add_selectbox == 'Random Forest Classifier':
            st.title('Random Forest Classifier')
            rf_criterion = ["gini","entropy"]
            rf_maxfeatures = ["2","5","7"]
            rf_nestimators = ["100","200","500"]
            rf_criterion_choice = st.sidebar.selectbox('Select Criteria', rf_criterion)
            rf_criterion_df = rf_data[rf_data['param_criterion']==(rf_criterion_choice)]            
            rf_maxfeatures_choice = st.sidebar.selectbox('Select Maximum Features', rf_maxfeatures)
            rf_mf_df = rf_criterion_df[rf_criterion_df['param_max_features']==int(rf_maxfeatures_choice)]            
            rf_nestimators_choice = st.sidebar.selectbox('Select no. of Estimators', rf_nestimators)
            rf_result1 = rf_mf_df[rf_mf_df['param_n_estimators']==int(rf_nestimators_choice)]           
            st.write('Mean Fit Time: ',rf_result1.mean_fit_time)
            st.write('Standard Fit Time: ', rf_result1.std_fit_time)
            st.write('Mean Score Time: ', rf_result1.mean_score_time)
            st.write('Standard Score Time: ', rf_result1.std_score_time)
            st.write('Parameters: ', rf_result1.params)
            st.write('Split 0 Test Score: ', rf_result1.split0_test_score)
            st.write('Split 1 Test Score: ',rf_result1.split1_test_score )
            st.write('Split 2 Test Score: ',rf_result1.split2_test_score )
            st.write('Mean Test Score: ',rf_result1.mean_test_score )
            st.write('Standard Test Score: ',rf_result1.std_test_score )
            st.write('Rank Test Score: ',rf_result1.rank_test_score )
        elif add_selectbox == 'Ensemble Learning':
            st.title('Ensemble Learning')
            el_learningrate = ["0.05","0.1","0.5","1"]
            el_maxdepth = ["3","5","10"]
            el_nestimators = ["100","200","500"]
            el_learningrate_choice = st.sidebar.selectbox('Select Learning Rate', el_learningrate)
            el_lr_df = el_data[el_data['param_learning_rate']==float(el_learningrate_choice)]
            el_maxdepth_choice = st.sidebar.selectbox('Select Maximum Depth', el_maxdepth)
            el_md_df = el_lr_df[el_lr_df['param_max_depth']==int(el_maxdepth_choice)]           
            el_nestimators_choice = st.sidebar.selectbox('Select no. of Estimators', el_nestimators)
            el_result1 = el_md_df[el_md_df['param_n_estimators']==int(el_nestimators_choice)]            
            st.write('Mean Fit Time: ',el_result1.mean_fit_time)
            st.write('Standard Fit Time: ', el_result1.std_fit_time)
            st.write('Mean Score Time: ', el_result1.mean_score_time)
            st.write('Standard Score Time: ', el_result1.std_score_time)
            st.write('Parameters: ', el_result1.params)
            st.write('Split 0 Test Score: ', el_result1.split0_test_score)
            st.write('Split 1 Test Score: ',el_result1.split1_test_score )
            st.write('Split 2 Test Score: ',el_result1.split2_test_score )
            st.write('Mean Test Score: ',el_result1.mean_test_score )
            st.write('Standard Test Score: ',el_result1.std_test_score )
            st.write('Rank Test Score: ',el_result1.rank_test_score )
        elif add_selectbox == 'Linear SVM':
            st.title('Linear Support Vector Machine')
            lsvm_C = ["0.01","0.10","1.00"]
            lsvm_loss = ["squared_hinge","hinge"]
            lsvm_maxiter = ["1000","3000","5000"]
            lsvm_penalty = ["l2"]
            lsvm_C_choice = st.sidebar.selectbox('Select C', lsvm_C)
            lsvm_C_df = lsvm_data[lsvm_data['param_C']== float(lsvm_C_choice)]           
            lsvm_loss_choice = st.sidebar.selectbox('Select Loss Function', lsvm_loss)
            lsvm_loss_df = lsvm_C_df[lsvm_C_df['param_loss']==(lsvm_loss_choice)]               
            lsvm_maxiter_choice = st.sidebar.selectbox('Select Maximum Iteration', lsvm_maxiter)
            lsvm_mi_df = lsvm_loss_df[lsvm_loss_df['param_max_iter']==int(lsvm_maxiter_choice)]             
            lsvm_penalty_choice = st.sidebar.selectbox('Select Penalty', lsvm_penalty)
            lsvm_result1 = lsvm_mi_df[lsvm_mi_df['param_penalty']==lsvm_penalty_choice]                
            st.write('Mean Fit Time: ',lsvm_result1.mean_fit_time)
            st.write('Standard Fit Time: ', lsvm_result1.std_fit_time)
            st.write('Mean Score Time: ', lsvm_result1.mean_score_time)
            st.write('Standard Score Time: ', lsvm_result1.std_score_time)
            st.write('Parameters: ', lsvm_result1.params)
            st.write('Split 0 Test Score: ', lsvm_result1.split0_test_score)
            st.write('Split 1 Test Score: ',lsvm_result1.split1_test_score )
            st.write('Split 2 Test Score: ',lsvm_result1.split2_test_score )
            st.write('Mean Test Score: ',lsvm_result1.mean_test_score )
            st.write('Standard Test Score: ',lsvm_result1.std_test_score )
            st.write('Rank Test Score: ',lsvm_result1.rank_test_score )
        elif add_selectbox == 'Non-Linear SVM':
            st.title('Non-Linear Support Vector Machine')
            nlsvm_C = ["0.01","0.10","1.00"]
            nlsvm_degree = ["2","3"]
            nlsvm_gamma = ["0.01","0.10","1.00"]
            nlsvm_kernel = ["poly","rbf"]
            nlsvm_maxiter = ["500","1000","10000"]
            nlsvm_C_choice = st.sidebar.selectbox('Select C', nlsvm_C)
            nlsvm_C_df = nlsvm_data[nlsvm_data['param_C']== float(nlsvm_C_choice)]  
            nlsvm_degree_choice = st.sidebar.selectbox('Select Degree', nlsvm_degree)
            nlsvm_degree_df = nlsvm_C_df[nlsvm_C_df['param_degree']==int(nlsvm_degree_choice)]               
            nlsvm_gamma_choice = st.sidebar.selectbox('Select Gamma Value', nlsvm_gamma)
            nlsvm_gamma_df = nlsvm_degree_df[nlsvm_degree_df['param_gamma']==float(nlsvm_gamma_choice)]             
            nlsvm_kernel_choice = st.sidebar.selectbox('Select Kernel', nlsvm_kernel)
            nlsvm_kernel_df = nlsvm_gamma_df[nlsvm_gamma_df['param_kernel']==nlsvm_kernel_choice]              
            nlsvm_maxiter_choice = st.sidebar.selectbox('Select Maximum Iteration', nlsvm_maxiter)
            nlsvm_result1 = nlsvm_kernel_df[nlsvm_kernel_df['param_max_iter']==int(nlsvm_maxiter_choice)]               
            st.write('Mean Fit Time: ',nlsvm_result1.mean_fit_time)
            st.write('Standard Fit Time: ', nlsvm_result1.std_fit_time)
            st.write('Mean Score Time: ', nlsvm_result1.mean_score_time)
            st.write('Standard Score Time: ', nlsvm_result1.std_score_time)
            st.write('Parameters: ', nlsvm_result1.params)
            st.write('Split 0 Test Score: ', nlsvm_result1.split0_test_score)
            st.write('Split 1 Test Score: ',nlsvm_result1.split1_test_score )
            st.write('Split 2 Test Score: ',nlsvm_result1.split2_test_score )
            st.write('Mean Test Score: ',nlsvm_result1.mean_test_score )
            st.write('Standard Test Score: ',nlsvm_result1.std_test_score )
            st.write('Rank Test Score: ',nlsvm_result1.rank_test_score )  
        elif add_selectbox == 'Decision Tree':
            st.title('Decision Tree')
            dt_criterion = ["gini","entropy"]
            dt_maxdepth = ["5","10"]
            dt_maxleafnodes = ["5","10"]
            dt_minsamplesleaf = ["3","5"]
            dt_criterion_choice = st.sidebar.selectbox('Select Criteria', dt_criterion)
            dt_criterion_df = dt_data[dt_data['param_criterion']== (dt_criterion_choice)]              
            dt_maxdepth_choice = st.sidebar.selectbox('Select Maximum Depth', dt_maxdepth)
            dt_maxdepth_df = dt_criterion_df[dt_criterion_df['param_max_depth']==int(dt_maxdepth_choice)]               
            dt_maxleafnodes_choice = st.sidebar.selectbox('Select Maximum Leaf Node', dt_maxleafnodes)
            dt_maxleafnodes_df = dt_maxdepth_df[dt_maxdepth_df['param_max_leaf_nodes']==int(dt_maxleafnodes_choice)]              
            dt_minsamplesleaf_choice = st.sidebar.selectbox('Select Minimum Samples Leaf', dt_minsamplesleaf)
            dt_result1 = dt_maxleafnodes_df[dt_maxleafnodes_df['param_min_samples_leaf']==int(dt_minsamplesleaf_choice)]               
            st.write('Mean Fit Time: ',dt_result1.mean_fit_time)
            st.write('Standard Fit Time: ', dt_result1.std_fit_time)
            st.write('Mean Score Time: ', dt_result1.mean_score_time)
            st.write('Standard Score Time: ', dt_result1.std_score_time)
            st.write('Parameters: ', dt_result1.params)
            st.write('Split 0 Test Score: ', dt_result1.split0_test_score)
            st.write('Split 1 Test Score: ',dt_result1.split1_test_score )
            st.write('Split 2 Test Score: ',dt_result1.split2_test_score )
            st.write('Mean Test Score: ',dt_result1.mean_test_score )
            st.write('Standard Test Score: ',dt_result1.std_test_score )
            st.write('Rank Test Score: ',dt_result1.rank_test_score )  
            
        else:
            st.title('Logistic Regression')
            lr_C = ["0.05","0.10","1.00","10.00"]
            lr_penalty = ["l2"]
            lr_solver = ["saga","lbfgs"]
            lr_C_choice = st.sidebar.selectbox('Select C', lr_C)
            lr_C_df = lr_data[lr_data['param_C']== float(lr_C_choice)] 
            lr_penalty_choice = st.sidebar.selectbox('Select Penalty', lr_penalty)
            lr_penalty_df = lr_C_df[lr_C_df['param_penalty']==(lr_penalty_choice)] 
            lr_solver_choice = st.sidebar.selectbox('Select Solver', lr_solver)
            lr_result1 = lr_penalty_df[lr_penalty_df['param_solver']==lr_solver_choice]  
            st.write('Mean Fit Time: ',lr_result1.mean_fit_time)
            st.write('Standard Fit Time: ', lr_result1.std_fit_time)
            st.write('Mean Score Time: ', lr_result1.mean_score_time)
            st.write('Standard Score Time: ', lr_result1.std_score_time)
            st.write('Parameters: ', lr_result1.params)
            st.write('Split 0 Test Score: ', lr_result1.split0_test_score)
            st.write('Split 1 Test Score: ',lr_result1.split1_test_score )
            st.write('Split 2 Test Score: ',lr_result1.split2_test_score )
            st.write('Mean Test Score: ',lr_result1.mean_test_score )
            st.write('Standard Test Score: ',lr_result1.std_test_score )
            st.write('Rank Test Score: ',lr_result1.rank_test_score )
    
    elif page == 'Deep Learning':
        st.title('Artificial Neural Network - 2 Hidden Layers')
        dnn2_activationin = ["tanh","softmax"]
        dnn2_batchsize = ["2","5","10"]
        dnn2_epochs = ["25","50"]
        dnn2_nodes = ["6","10","20"]
        dnn2_activationin_choice = st.sidebar.selectbox('Select Activation Function', dnn2_activationin)
        dnn2_activationin_df = dnn2_data[dnn2_data['param_activation_in']== (dnn2_activationin_choice)]   
        dnn2_batchsize_choice = st.sidebar.selectbox('Select Batch Size', dnn2_batchsize)
        dnn2_batchsize_df = dnn2_activationin_df[dnn2_activationin_df['param_batch_size']==int(dnn2_batchsize_choice)]
        dnn2_epochs_choice = st.sidebar.selectbox('Select no. of Epochs', dnn2_epochs)
        dnn2_epochs_df = dnn2_batchsize_df[dnn2_batchsize_df['param_epochs']==int(dnn2_epochs_choice)]
        dnn2_nodes_choice = st.sidebar.selectbox('Select no. of Nodes', dnn2_nodes)
        dnn2_result1 = dnn2_epochs_df[dnn2_epochs_df['param_nodes']==int(dnn2_nodes_choice)]
        st.write('Mean Fit Time: ',dnn2_result1.mean_fit_time)
        st.write('Standard Fit Time: ', dnn2_result1.std_fit_time)
        st.write('Mean Score Time: ', dnn2_result1.mean_score_time)
        st.write('Standard Score Time: ', dnn2_result1.std_score_time)
        st.write('Parameters: ', dnn2_result1.params)
        st.write('Split 0 Test Score: ', dnn2_result1.split0_test_score)
        st.write('Split 1 Test Score: ',dnn2_result1.split1_test_score )
        st.write('Split 2 Test Score: ',dnn2_result1.split2_test_score )
        st.write('Mean Test Score: ',dnn2_result1.mean_test_score )
        st.write('Standard Test Score: ',dnn2_result1.std_test_score )
        st.write('Rank Test Score: ',dnn2_result1.rank_test_score )     
    
    elif page == 'Association Rule Mining':
        add_selectbox = st.sidebar.selectbox("Readmission ?",("Yes", "No"))
        support = st.sidebar.slider("Support:", min_value = 0.0, value = 0.5, max_value = 1.00, step = 0.01)
        confidence = st.sidebar.slider("Confidence:", min_value = 0.0, value = 0.5, max_value = 1.00, step = 0.01)
        lift = st.sidebar.slider("Lift:", min_value = 1.00, value = 1.00, max_value = 2.00, step = 0.05)
        max_rule_length = st.sidebar.slider("Maximum Rule Length:", 0, 10, 0, 1)
        #num_top_rules = st.sidebar.slider("Number of top rules selected:", 0, 100, 25, 5)
        st.title("Association Rule Mining - Diabetic Patient Readmission ")
        filter = st.sidebar.selectbox("Sort Rules in descending order by:",("lift", "confidence","support"))
        if st.sidebar.button('Run Algorithm'):
            #Attrition status
            if add_selectbox == 'Yes':
                #Call ARM algo file
                #rules = arm.run('YES',config)
                filtered_data = getARMdata('YES',arm_data)
                print(filtered_data.head())
                print('FILTERED DATA-------------')
                rules = arm(filtered_data,support,confidence,lift,max_rule_length,filter)
                print(rules.head())
                print('RULES--------------------------')
                #Plot rule DF
                st.table(rules)
                #Plot scatter plot
                st.title('Scatter Plot')
                scatter_chart = st.altair_chart(alt.Chart(rules)
                .mark_circle(size=60)
                .encode(x='support', y='confidence', color='lift')
                .interactive())
            if add_selectbox == 'No':
                filtered_data = getARMdata('NO',arm_data)
                rules = arm(filtered_data,support,confidence,lift,max_rule_length,filter)
                st.table(rules)
                st.title('Scatter Plot')
                scatter_chart = st.altair_chart(alt.Chart(rules)
                .mark_circle(size=60)
                .encode(x='support', y='confidence', color='lift')
                .interactive())       
    else:
        st.title('Best Model Performances')
        bestmodel_df

#Data preprocessing for ARM
@st.cache
def getARMdata(status,arm_data):    
    arm_data1 = arm_data
    return arm_data1

#ARM function
@st.cache
def arm(arm_data1,support,confidence,lift,max_rule_length,filter):
    if filter == 'lift':
        threshold = lift
    elif filter == 'confidence':
        threshold = confidence
    else:
        threshold = support
    #Prepare Dataset for Association Rule Mining
    arm_data2 = pd.DataFrame({col: str(col)+'=' for col in arm_data1}, index=arm_data1.index) + arm_data1.astype(str)
    #Run APriori with Apyori Library
    records = []
    for i in range(0,len(arm_data2)):
        records.append([str(arm_data2.values[i,j]) for j in range(0, len(arm_data2.columns))])
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    #Use Association Rules from Mlxtend Library
    frequent_itemsets = apriori(df, use_colnames=True, min_support=support, max_len=max_rule_length)
    rules = association_rules(frequent_itemsets, metric=filter, min_threshold=threshold)
    #target = '{\'Attrition=Yes\'}'
    arm_results = rules.sort_values(by=filter, ascending=False).head(50)
    return arm_results

if __name__ == '__main__':
    main()