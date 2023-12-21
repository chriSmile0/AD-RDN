#code rna
import inc_rna as ir
############

ir.my_prinft("lol") #ok 

#1) neuranet = supervisé
df = ir.pd.read_csv("../synthetic.csv")

df_columns = df.columns.values.tolist()

features = df_columns[0:14] #4 premiere colonnes
label = df_columns[14:] #4eme colonne , etiquette


X = df[features]
y = df[label]
y = ir.pd.get_dummies(y)

#2_2


#....
X_train,X_test,y_train,y_test = ir.train_test_split(X,y,test_size=0.15,random_state=42)

#3
#Voir cours pour Q1-2-3
#Q1 : tanh : renvoie une matrice d'activation et approximation de l'erreur par la dérivée
#      sigmoid : renvoie une matrice d'activation et approximation de l'erreur par la dérivée
#      softmax : renvoie le softmax de Z
#      relu  : renvoie une matrice d'activation et approximation de l'erreur par la dérivée

# Cout : cross_entropy_cost :
#       MSE_cost :  

#3_1
print("X train : ")
print(X_train)
print("X_test  : ")
print(X_test)
print("Y_train : ")
print(y_train)
print("Y test : ")
print(y_test)

"""
my_neuralnet = ir.NeuralNet(X_train,y_train,X_test,y_test)
print(my_neuralnet._activation)
err_train,err_test = my_neuralnet.train_epoch()

print("Erreur d'entrainement : {}".format(err_train))
print("Erreur de validation {}".format(err_test))"""