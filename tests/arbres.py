#code arbres
import inc_arbres as ia #on accede a nos fonctions et aussi aux differentes libs
#########

df = ia.pd.read_csv('../synthetic.csv')
print ( df.head())
print ( df.info() )


labels_codes ,labels = ia.pd.factorize(df['Class'])


df['Class'] = labels_codes

print(df)

print(df.info())

#g = ia.sns.pairplot(df, hue='Class')
#ia.plt.show()
#3 


# Nombre d instances dans le dataframe :
nb_lignes = df.shape[0]

# Decompte du nombre d instance de chaque classe dans le dataframe
series = df['Class'].value_counts()

attribute_list =  ['Attr_A' ,'Attr_B' ,'Attr_C' , 'Attr_D','Attr_E',
                        'Attr_F' ,'Attr_G' ,'Attr_H' ,'Attr_I' ,'Attr_J',
                        'Attr_K' ,'Attr_L' ,'Attr_M' ,'Attr_N']

print(attribute_list)
print("Version trivial du split")
best_atribute,best_gain,best_split,best_partitions = ia.splitting_attribute(df,'Class',attribute_list)
print("best attribute :{}".format(best_atribute))
print("best gain :{}".format(best_gain))
print("best split :{}".format(best_split))
print("best partitions :{}".format(best_atribute))

#entropie de df.['Class']
print(ia.entropie(df['Class']))

#test sur le gain avec les quartiles
print("Version avec le split par quartile")
best_atribute,best_gain,best_quartile,best_split,best_partitions = ia.splitting_attribute_quartile_version(df,'Class',attribute_list)
print("best attribute :{}".format(best_atribute))
print("best gain :{}".format(best_gain))
print("best quartile :{}".format(best_quartile))
print("best split :{}".format(best_split))
print("best partitions :{}".format(best_atribute))