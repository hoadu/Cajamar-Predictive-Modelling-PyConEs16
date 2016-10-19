#!/usr/bin/env python
# -*- coding: utf-8 -*-

#---------------------------------------------------------------------------#
#                                                                           #   
#                           CAJAMAR DATATHON                                #
#                   EQUIPO: RECOMPILE BEFORE FLIGHT                         #
#                PyConES 2016 - Octubre 2016 - Almería                      #
#                                                                           #
#---------------------------------------------------------------------------#


#---------------------------------------------------------------------------#
#                           1- IMPORT LIBRARIES                             #
#---------------------------------------------------------------------------#

# Made in Python3.5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Used scikit-learn version 0.18

# Avoid the DeprecationWarning appearance
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier

# https://github.com/scikit-learn-contrib/imbalanced-learn
from imblearn.over_sampling import SMOTE

#---------------------------------------------------------------------------#
#                           2- MODEL PARAMETERS                             #
#---------------------------------------------------------------------------#

test_size_value = 0.2       # !!! Set to 0 for real training and prediction of test data + Comment block #6
random_state_value = 1729   


def main():

    # Input datasets as pandas dataframes
    df_original = pd.read_csv('Predictive Modelling Train.txt', decimal=",", sep="|")
    df_original_predict = pd.read_csv('Predictive Modelling Test.txt', decimal=",", sep="|")

    #---------------------------------------------------------------------------#
    #                           3- DATA MANIPULATION                            #
    #---------------------------------------------------------------------------#

    # Train - Drop ID and remove duplicates in train set
    df_original = df_original.drop(['ID'], 1)
    df_original = df_original.drop_duplicates()

    # Predict - Save ID of dataframe to predict
    test_id = df_original_predict.ID

    # Predict - Drop ID and convert to numpy array 
    test_preprocessed = df_original_predict.drop(["ID"], 1)
    X_predict = np.array(test_preprocessed)

    # Train - Separate class and features
    X = np.array(df_original.drop(['TARGET'],axis=1))
    y = np.array(df_original.TARGET.values)

    # Split train dataset
    X_to_balance, X_real_test, y_to_balance, y_real_test = train_test_split(X, y, test_size=test_size_value, random_state=random_state_value)

    # Oversample data (TARGET=1) to balance
    sm = SMOTE(kind='regular')
    X_balanced, y_balanced = sm.fit_sample(X_to_balance, y_to_balance)

    # Create new features
    df_real     = feature_engineering_df(X_real_test, df_original)
    df_balanced = feature_engineering_df(X_balanced, df_original)
    df_test_processed = feature_engineering_df(X_predict, df_original)

    # Convert pandas dataframes to numpy arrays
    X_balanced = np.array(df_balanced)
    X_real_test = np.array(df_real)
    X_predict = np.array(df_test_processed)


    #---------------------------------------------------------------------------#
    #                           4- FEATURE SELECTION                            #
    #---------------------------------------------------------------------------#

    # Define classifier for feature importance and selection
    clf1 = ExtraTreesClassifier(n_jobs=-1, random_state=random_state_value)

    selector = clf1.fit(X_balanced, y_balanced)

    # Choose best features
    fs = SelectFromModel(selector, prefit=True)

    # Discard non selected features
    X_real_test = fs.transform(X_real_test)
    X_balanced = fs.transform(X_balanced)
    X_predict_final = fs.transform(X_predict)


    #---------------------------------------------------------------------------#
    #                           5- MODEL TRAIN + FIT                            #
    #---------------------------------------------------------------------------#

    # Define prediction classifier and fit
    clf2 = KNeighborsClassifier(n_jobs=-1, n_neighbors=9)

    clf2.fit(X_balanced, y_balanced)


    #---------------------------------------------------------------------------#
    #                           6- MODEL EVALUATION                             #
    #---------------------------------------------------------------------------#

    # !!! IMPORTANT: COMMENT WHOLE BLOCK WHEN DOING REAL TRAINING AND PREDCTION (test_size_value = 0)

    # Print used classifiers and their parameters
    print("Feature selection classifier: ",  clf1, "\n")
    print("Model classifier: ", clf2, "\n")

    # Calculate predictions
    y_pred = clf2.predict_proba(X_real_test)[:,1]
    y_pred_int = clf2.predict(X_real_test)

    # Evaluate model 
    print("Roc AUC: ", roc_auc_score(y_real_test, y_pred ,average='macro'))
    accuracy = clf2.score(X_real_test, y_real_test)
    print("Accuracy: ", accuracy)
    print("f1 Score: ", f1_score(y_real_test, y_pred_int, average='macro'))

    # Hardcoded benchmark of filling prediction with most common class (0)
    zeros_benchmark = 1-7477/459992 
    print("Filling with 0's benchmark:    ", zeros_benchmark)

    # Fixed accuracy with zeros_benchmark
    print("Fixed accuracy with benchmark: ", (accuracy-zeros_benchmark)/(1-zeros_benchmark))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_real_test, y_pred_int)
    print("\n[Confusion Matrix]: \n", conf_matrix)

    print("\n------------------------------------------------------------------\n\n")

    #---------------------------------------------------------------------------#
    #                       7- PREDICTION AND SUBMISSION                        #
    #---------------------------------------------------------------------------#

    # Make prediction
    predict_submission = clf2.predict(X_predict_final)

    # Save in csv
    submission = pd.DataFrame({"ID":test_id, "TARGET": predict_submission})
    submission.to_csv("submission.csv", index=False, sep="|")


#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------#
#                       8- FEATURE ENGINEERING FUNCTION                     #
#---------------------------------------------------------------------------#


def feature_engineering_df(array_to_modify, original_dataframe):

    # Convert numpy array to pandas dataframe
    saved_index = list(range(array_to_modify.shape[0]))
    saved_columns = original_dataframe.columns
    saved_columns = saved_columns.drop(['TARGET'],1)

    df = pd.DataFrame(array_to_modify, index=saved_index, columns=saved_columns)

    # Absolute quantity of money moved
    df['ABS_MOVEMENTS']=df['IMP_CONS_01'].abs()+df['IMP_CONS_02'].abs()+df['IMP_CONS_03'].abs()+df['IMP_CONS_04'].abs()+df['IMP_CONS_05'].abs()+\
                        df['IMP_CONS_06'].abs()+df['IMP_CONS_07'].abs()+df['IMP_CONS_08'].abs()+df['IMP_CONS_09'].abs()+df['IMP_CONS_10'].abs()+\
                        df['IMP_CONS_12'].abs()+df['IMP_CONS_13'].abs()+df['IMP_CONS_14'].abs()+df['IMP_CONS_15'].abs()+\
                        df['IMP_CONS_16'].abs()+df['IMP_CONS_17'].abs()

    # Sum of money movements
    df['SUM_MOVEMENTS']=df['IMP_CONS_01']+df['IMP_CONS_02']+df['IMP_CONS_03']+df['IMP_CONS_04']+df['IMP_CONS_05']+\
                        df['IMP_CONS_06']+df['IMP_CONS_07']+df['IMP_CONS_08']+df['IMP_CONS_09']+df['IMP_CONS_10']+\
                        df['IMP_CONS_12']+df['IMP_CONS_13']+df['IMP_CONS_14']+df['IMP_CONS_15']+\
                        df['IMP_CONS_16']+df['IMP_CONS_17']

    # Quantity of zeros in row
    df['ZEROS_IN_ROW']=(df == 0).astype(int).sum(axis=1)

    # Standard deviation
    df['STD'] = df.apply(lambda x: np.std(x), axis=1)

    # Sum of balances
    df['SUMA_SALDOS']=df['IMP_SAL_01']+df['IMP_SAL_02']+df['IMP_SAL_03']+df['IMP_SAL_04']+df['IMP_SAL_05']+df['IMP_SAL_06']+df['IMP_SAL_07']+\
                      df['IMP_SAL_08']+df['IMP_SAL_09']+df['IMP_SAL_10']+df['IMP_SAL_11']+df['IMP_SAL_12']+df['IMP_SAL_13']+df['IMP_SAL_14']+\
                      df['IMP_SAL_15']+df['IMP_SAL_16']+df['IMP_SAL_17']+df['IMP_SAL_18']+df['IMP_SAL_19']+df['IMP_SAL_20']+df['IMP_SAL_21']                  

    # Sum of financial products
    df['CANTIDAD_PRODUCTOS']=df['IND_PROD_01']+df['IND_PROD_02']+df['IND_PROD_03']+df['IND_PROD_04']+df['IND_PROD_05']+df['IND_PROD_06']+df['IND_PROD_07']+\
                             df['IND_PROD_08']+df['IND_PROD_09']+df['IND_PROD_10']+df['IND_PROD_11']+df['IND_PROD_12']+df['IND_PROD_13']+df['IND_PROD_14']+\
                             df['IND_PROD_15']+df['IND_PROD_16']+df['IND_PROD_17']+df['IND_PROD_18']+df['IND_PROD_19']+df['IND_PROD_20']+df['IND_PROD_21']+\
                             df['IND_PROD_22']+df['IND_PROD_23']

    # Number of operations with financial products
    df['NUMERO_OPERACIONES']=df['NUM_OPER_24']+df['NUM_OPER_02']+df['NUM_OPER_03']+df['NUM_OPER_04']+df['NUM_OPER_05']+df['NUM_OPER_06']+df['NUM_OPER_07']+\
                             df['NUM_OPER_08']+df['NUM_OPER_09']+df['NUM_OPER_10']+df['NUM_OPER_11']+df['NUM_OPER_12']+df['NUM_OPER_13']+df['NUM_OPER_14']+\
                             df['NUM_OPER_15']+df['NUM_OPER_16']+df['NUM_OPER_17']+df['NUM_OPER_18']+df['NUM_OPER_19']+df['NUM_OPER_20']+df['NUM_OPER_21']+\
                             df['NUM_OPER_22']+df['NUM_OPER_23']

    return df


#---------------------------------------------------------------------------------

main()



