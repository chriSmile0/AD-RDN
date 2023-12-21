#code question 
#P1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("synthetic.csv")
result_class0 = df.loc[(df['Class'] == 0)]
print(len(result_class0.index)) # nb class 0

result_class1 = df.loc[(df['Class'] == 1)]
print(len(result_class1.index)) # nb class 1

result_class2 = df.loc[(df['Class'] == 2)]
print(len(result_class2.index)) # nb class 2

result_class3 = df.loc[(df['Class'] == 3)]
print(len(result_class3.index)) # nb class 3


Attr_A = np.array(df['Attr_B'])
attr_b = np.array(df['Attr_G'])
#plt.scatter(Attr_A,attr_b, color = 'blue')
#plt.scatter(attr_b,Attr_A, color = 'red')
#plt.show()

#Q4
# Attr_A et attr_b separables NON
# Attr_A et Attr_A separables OUI
# Attr_A et attr_d separables OUI
# Attr_A et attr_e separables OUI
# Attr_A et attr_f separables OUI
# Attr_A et attr_g separables NON
# Attr_A et attr_h separables NON

#Tout ceux qui sont oui avec a le seront ensemble : 
    # Exemple c et d 
#Tout ceux qui sont non avec a ne le seront pas entre eux : 
    # Exemple b et g 

# Comme certaines oui et d'autres non , on va plus pencher pour le non 

##Autre situation avec les classes 
"""
Attr_Alass0 = np.array(df.loc[(df['Class'] == 0)])
Attr_Alass0.resize(1,100)
Attr_Alass1 = np.array(df.loc[(df['Class'] == 1)])
Attr_Alass1.resize(1,100)
Attr_Alass2 = np.array(df.loc[(df['Class'] == 2)])
Attr_Alass2.resize(1,100)
Attr_Alass3 = np.array(df.loc[(df['Class'] == 3)])
Attr_Alass3.resize(1,100)
plt.scatter(Attr_Alass0,Attr_Alass1, color = 'blue')
plt.scatter(Attr_Alass1,Attr_Alass0, color = 'red')
plt.scatter(Attr_Alass2,Attr_Alass0, color = 'green')
plt.scatter(Attr_Alass3,Attr_Alass0, color = 'yellow')
plt.show()
"""
##Encore une autre facon de faire 
"""
df_Attr_a_class = np.array(df['Attr_A'])
df_Attr_a_class.resize(1,100)
df_attr_b_class = np.array(df['Attr_N'])
df_attr_b_class.resize(1,100)

df_Attr_a_b_class = df[["Attr_A","Attr_N","Class"]]
df_Attr_a_class = df_Attr_a_b_class[['Attr_A','Class']]
df_attr_b_class = df_Attr_a_b_class[['Attr_N','Class']]

Attr_A_class0 = np.array(df_Attr_a_class.loc[(df_Attr_a_class['Class'] == 0)])
attr_b_class0 = np.array(df_attr_b_class.loc[(df_attr_b_class['Class'] == 0)])
Attr_A_class0.resize(1,100)
attr_b_class0.resize(1,100)
Attr_A_class1 = np.array(df_Attr_a_class.loc[(df_Attr_a_class['Class'] == 1)])
attr_b_class1 = np.array(df_attr_b_class.loc[(df_attr_b_class['Class'] == 1)])
Attr_A_class1.resize(1,50)
attr_b_class1.resize(1,50)

Attr_A_class2 = np.array(df_Attr_a_class.loc[(df_Attr_a_class['Class'] == 2)])
attr_b_class2 = np.array(df_attr_b_class.loc[(df_attr_b_class['Class'] == 2)])
Attr_A_class2.resize(1,50)
attr_b_class2.resize(1,50)

Attr_A_class3 = np.array(df_Attr_a_class.loc[(df_Attr_a_class['Class'] == 3)])
attr_b_class3 = np.array(df_attr_b_class.loc[(df_attr_b_class['Class'] == 3)])
Attr_A_class3.resize(1,50)
attr_b_class3.resize(1,50)

####to continue !!!!!!!!!!!!!
plt.scatter(Attr_A_class0,attr_b_class0, color = 'blue')
plt.scatter(Attr_A_class1,attr_b_class1, color = 'orange')
plt.scatter(Attr_A_class2,attr_b_class2, color = 'green')
plt.scatter(Attr_A_class3,attr_b_class3, color = 'red')
#same for a class 2 and class 3
plt.title("Attr_A and Attr_D by class")
plt.xlabel("Attr_A")
plt.ylabel("Attr_N")
plt.grid()
plt.show()"""




#utile pour la Q4 , cela nous montre que nos données sont inséparables 
## Equivalent a notre tableau juste plus haut 
import seaborn as sns

dfa = pd.read_csv('synthetic.csv')
print ( dfa.head())
print ( dfa.info() )

labels_codes ,labels = pd.factorize(dfa['Class'])


dfa['Class'] = labels_codes

print(dfa)

print(dfa.info())

g = sns.pairplot(dfa, hue='Class')
plt.show()


