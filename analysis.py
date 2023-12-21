#P3 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

y_test = pd.read_csv("predictions/y_test.csv")
y_pred_DT4 = pd.read_csv("predictions/y_pred_DT4.csv")
y_pred_DT5 = pd.read_csv("predictions/y_pred_DT5.csv")
y_pred_DT6 = pd.read_csv("predictions/y_pred_DT6.csv")

y_pred_NN_relu_6_4 = pd.read_csv("predictions/y_pred_NN_relu_6-4.csv")
y_pred_NN_relu_10_8_4 = pd.read_csv("predictions/y_pred_NN_relu_10-8-4.csv")
y_pred_NN_relu_10_8_6 = pd.read_csv("predictions/y_pred_NN_relu_10-8-6.csv")
y_pred_NN_tanh_6_4 = pd.read_csv("predictions/y_pred_NN_tanh_6-4.csv")
y_pred_NN_tanh_10_8_4 = pd.read_csv("predictions/y_pred_NN_tanh_10-8-4.csv")
y_pred_NN_tanh_10_8_6 = pd.read_csv("predictions/y_pred_NN_tanh_10-8-6.csv")
print("Test")
total_columns = y_test.size

 
def classification(matrix_confusion,total):
    nb_true_positiv = np.array([0,0,0,0]) # i = 0, i = 1, i = 2, i = 3, i = class
    nb_true_negativ = np.array([0,0,0,0])
    nb_false_positiv = np.array([0,0,0,0])
    nb_false_negativ =  np.array([0,0,0,0])
    for i in range(0,4):
        for j in range(0,4):
            if(i == j):
                nb_true_positiv[i] = matrix_confusion[i][j]
            if(i != j):
                nb_false_negativ[i] += matrix_confusion[j][i]
                nb_false_positiv[i] += matrix_confusion[i][j]
        nb_true_negativ[i] = total - (nb_false_negativ[i] + nb_false_positiv[i] + nb_true_positiv[i])

    return nb_true_positiv,nb_true_negativ,nb_false_positiv,nb_false_negativ

def normalize_relu_tanh_for_confusion_matrix(pred):
    df_true_pred  = pd.DataFrame({'Class': []})
    df_true_pred['Class'] = df_true_pred['Class'].astype('int')
    len_pred = len(pred)
    max_row = 0.0
    idx_max_row = 0
    for i in range(0,len_pred):
        for j in range(0,4):
            if(max_row < pred.loc[i][j]):
                max_row = pred.loc[i][j]
                idx_max_row = j
        df_true_pred = df_true_pred.append({'Class': idx_max_row },ignore_index=True)
        max_row = 0.0
    return df_true_pred


def matrice_confusion(df,test):
    nb_lines = len(df)
    nb_test_lines = nb_lines
    #a droite predicted, en haut valeurs test
    matrice = np.array([[0,0,0,0], [0,0,0,0],
                        [0,0,0,0],[0,0,0,0]])
    for i in range(0,nb_lines):
        save_loc = df.loc[i][0]
        if(df.loc[i][0] == test.loc[i][0]):
            matrice[save_loc][save_loc] += 1
        else:
            test_loc = test.loc[i][0]
            matrice[save_loc][test_loc] += 1

    print(matrice)
    return matrice


def f_precision(nb_true_positiv,nb_false_positiv):
    if(nb_false_positiv == 0.0):
        return 0.0
    return float(nb_true_positiv)/(float(nb_true_positiv)+(float(nb_false_positiv)))

def f_recall(nb_true_positiv,nb_false_negativ):
    if(nb_false_negativ == 0.0):
        return 0.0
    return float(nb_true_positiv)/(float(nb_true_positiv)+float(nb_false_negativ))

def f_accuracy(nb_true_positiv,nb_true_negativ,total):
   return (float(nb_true_positiv)+float(nb_true_negativ))/float(total)

def f1_score(precision,recall):
    if((precision+recall) == 0.0):
        return 0.0
    return 2*(precision*recall)/(precision+recall)


def print_classification(t_positivs,t_negativs,f_positivs,f_negativs,modele,total):
    Accuracys = np.array([0.0,0.0,0.0,0.0])
    Precisions = np.array([0.0,0.0,0.0,0.0])
    Recalls = np.array([0.0,0.0,0.0,0.0])
    F1_Scores = np.array([0.0,0.0,0.0,0.0])
    for i in range(0,4):
        Precisions[i] = f_precision(t_positivs[i],f_positivs[i])
        Recalls[i] = f_recall(t_positivs[i],f_negativs[i])
        Accuracys[i] = f_accuracy(t_positivs[i],t_negativs[i],total)
        F1_Scores[i] = f1_score(Precisions[i],Recalls[i])

    print("                                                 Modele {}".format(modele))
    print("Classes/Accuracy  | c1: {} | c2: {} | c3: {} | c4: {} |".format(Accuracys[0],Accuracys[1],Accuracys[2],Accuracys[3]))
    print("Classes/Precision | c1: {} | c2: {} | c3: {} | c4: {} |".format(Precisions[0],Precisions[1],Precisions[2],Precisions[3]))
    print("Classes/Recall    | c1: {} | c2: {} | c3: {} | c4: {} |".format(Recalls[0],Recalls[1],Recalls[2],Recalls[3]))
    print("Classes/F1-Score  | c1: {} | c2: {} | c3: {} | c4: {} |".format(F1_Scores[0],F1_Scores[1],F1_Scores[2],F1_Scores[3]))



####MATRICE DE CONFUSION 
print("Confusion matrix DT")
print("Confusion matrix DT4 ")
confusion_matrix_DT4 = matrice_confusion(y_pred_DT4,y_test)
print("Confusion matrix DT5 ")
confusion_matrix_DT5 = matrice_confusion(y_pred_DT5,y_test)
print("Confusion matrix DT6 ")
confusion_matrix_DT6 = matrice_confusion(y_pred_DT6,y_test)

print("Confusion matric Relu")
print("Confusion matrix 6_4 ")
confusion_matrix_relu_6_4 = matrice_confusion(normalize_relu_tanh_for_confusion_matrix(y_pred_NN_relu_6_4),y_test)
print("Confusion matrix 10_8_4 ")
confusion_matrix_relu_10_8_4 = matrice_confusion(normalize_relu_tanh_for_confusion_matrix(y_pred_NN_relu_10_8_4),y_test)
print("Confusion matrix 10_8_6 ")
confusion_matrix_relu_10_8_6 = matrice_confusion(normalize_relu_tanh_for_confusion_matrix(y_pred_NN_relu_10_8_6),y_test)

print("Confusion matrix Tanh")
print("Confusion matrix 6_4 ")
confusion_matrix_tanh_6_4 = matrice_confusion(normalize_relu_tanh_for_confusion_matrix(y_pred_NN_tanh_6_4),y_test)
print("Confusion matrix 10_8_4 ")
confusion_matrix_tanh_10_8_4 = matrice_confusion(normalize_relu_tanh_for_confusion_matrix(y_pred_NN_tanh_10_8_4),y_test)
print("Confusion matrix 10_8_6 ")
confusion_matrix_tanh_10_8_6 = matrice_confusion(normalize_relu_tanh_for_confusion_matrix(y_pred_NN_tanh_10_8_6),y_test)


####CLASSIFICATION
print("Classification DT4")
t_positiv_dt4,t_negativ_dt4,f_positiv_dt4,f_negativ_dt4 = classification(confusion_matrix_DT4,total_columns)
print_classification(t_positiv_dt4,t_negativ_dt4,f_positiv_dt4,f_negativ_dt4,"DT4",total_columns)
print("Classification DT5")
t_positiv_dt5,t_negativ_dt5,f_positiv_dt5,f_negativ_dt5 = classification(confusion_matrix_DT5,total_columns)
print_classification(t_positiv_dt5,t_negativ_dt5,f_positiv_dt5,f_negativ_dt5,"DT5",total_columns)
print("Classification DT6")
t_positiv_dt6,t_negativ_dt6,f_positiv_dt6,f_negativ_dt6 = classification(confusion_matrix_DT6,total_columns)
print_classification(t_positiv_dt4,t_negativ_dt4,f_positiv_dt4,f_negativ_dt4,"DT6",total_columns)

print("Classification Relu 6-4")
t_positiv_relu6_4,t_negativ_relu6_4,f_positiv_relu6_4,f_negativ_relu6_4 = classification(confusion_matrix_relu_6_4,total_columns)
print_classification(t_positiv_relu6_4,t_negativ_relu6_4,f_positiv_relu6_4,f_negativ_relu6_4,"Relu 6-4",total_columns)
print("Classification Relu 10-8-4")
t_positiv_relu10_8_4,t_negativ_relu10_8_4,f_positiv_relu10_8_4,f_negativ_relu10_8_4 = classification(confusion_matrix_relu_10_8_4,total_columns)
print_classification(t_positiv_relu10_8_4,t_negativ_relu10_8_4,f_positiv_relu10_8_4,f_negativ_relu10_8_4,"Relu 10-8-4",total_columns)
print("Classification Relu 10-8-6")
t_positiv_relu10_8_6,t_negativ_relu10_8_6,f_positiv_relu10_8_6,f_negativ_relu10_8_6 = classification(confusion_matrix_relu_10_8_6,total_columns)
print_classification(t_positiv_relu10_8_6,t_negativ_relu10_8_6,f_positiv_relu10_8_6,f_negativ_relu10_8_6,"Relu 10-8-6",total_columns)

print("Classification Tanh 6-4")
t_positiv_tanh6_4,t_negativ_tanh6_4,f_positiv_tanh6_4,f_negativ_tanh6_4 = classification(confusion_matrix_tanh_6_4,total_columns)
print_classification(t_positiv_tanh6_4,t_negativ_tanh6_4,f_positiv_tanh6_4,f_negativ_tanh6_4,"Tanh 6-4",total_columns)
print("Classification Tanh 10-8-4")
t_positiv_tanh10_8_4,t_negativ_tanh10_8_4,f_positiv_tanh10_8_4,f_negativ_tanh10_8_4 = classification(confusion_matrix_tanh_10_8_4,total_columns)
print_classification(t_positiv_tanh10_8_4,t_negativ_tanh10_8_4,f_positiv_tanh10_8_4,f_negativ_tanh10_8_4,"Tanh 10-8-4",total_columns)
print("Classification Tanh 10-8-6")
t_positiv_tanh10_8_6,t_negativ_tanh10_8_6,f_positiv_tanh10_8_6,f_negativ_tanh10_8_6 = classification(confusion_matrix_tanh_10_8_6,total_columns)
print_classification(t_positiv_tanh10_8_6,t_negativ_tanh10_8_6,f_positiv_tanh10_8_6,f_negativ_tanh10_8_6,"Tanh 10-8-6",total_columns)

