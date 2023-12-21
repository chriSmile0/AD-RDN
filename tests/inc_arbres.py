##include arbres 
## contient les fonctions pour l'utilisations des arbres de decisions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def entropie(labels):
    classes, nb_occ = np.unique(labels, return_counts=True)
    entropy = 0
    for i in range(len(classes)):
        p_i = nb_occ[i]/np.sum(nb_occ)
        log_2 = math.log2(p_i)

        entropy += p_i * log_2

    return -entropy



def gain_calculation(data,target,a):
    series = data[target].value_counts()
    h_total = entropie(series)
    gain = 0
    split_value = 0
    partitions = []
    sorted_data = data.sort_values(by=a)
    classe = sorted_data[target][0]
    for i in sorted_data[target]: 
        if sorted_data[target][i] != classe:
            split_value = sorted_data[a][i]
            partitions[0] = sorted_data[sorted_data[a] < split_value]
            partitions[1] = sorted_data[sorted_data[a] >= split_value]
            acc = 0.0
            for j in range(len(partitions)):
                acc += partitions[j].shape[0]/sorted_data.shape[0]
            gain = h_total - acc
            break
    return gain,split_value,partitions




class Node:
    def __init__(self, *args, **kwargs):
        feuille = False
        if(len(args)==2):
            self._feuille = args[1]
            self._prediction = args[0]
        else : 
            self._prediction = args[4]
            self._feuille = False
            self._split = args[0]
            self._a = args[1]
            self._right = args[2]
            self._left = args[3]

    def feuille(self):
        return self._feuille

    def result(self,spacing):
        return '{}'.format(self._prediction)

def print_tree(node,spacing=''):
    if node is None:
        return 
    if node.feuille() == True:
        print(spacing + node.result(spacing))
        return 
    print('{}[Attribute: {} Split value : {}]'.format(spacing,node.a,node.split_value))

    print(spacing + '> True')
    print_tree(node.left,spacing + '-')

    print(spacing + '> False')
    print_tree(node.right,spacing + '-')
    return 

def splitting_attribute(data,target,attribute_list):
    h_total = entropie(data[target])
    best_atribute = 0
    best_gain = 0
    best_split = 0
    best_partitions = []
    for a in attribute_list:
        sorted_data = data.sort_values(by=a) # on tri par l'attribut en entrer
        classe_init = sorted_data[target][0]
        for i in sorted_data[target]:
            split = 0
            if i != classe_init:
                split = sorted_data[a][i]
                partitions   = [sorted_data[sorted_data[a] < split], sorted_data[sorted_data[a] >= split]]
                E_partitions = [entropie(partitions[j][target]) for j in range(len(partitions))]
                acc = 0.0
                for j in range(len(partitions)):
                    acc += partitions[j].shape[0] / sorted_data.shape[0] * E_partitions[j]
                
                gain = h_total - acc
                if gain > best_gain:
                    best_atribute = a
                    best_gain = gain
                    best_split = split
                    best_partitions = partitions
                    
                break
    return best_atribute,best_gain,best_split,best_partitions   

def construction_arbre(data,target,attribute_list,profondeur):
    attribute,gain,split,partitions = splitting_attribute(data,target,attribute_list)
    prediction = data[target].value_counts()
    attribute_list.remove(attribute)
   
        
    seuil = 2 #on defini notre propre seuil a 1 , sinon passage en param
    if (profondeur > seuil) or (attribute_list is None):
        return Node(prediction,True)
        
    left = construction_arbre(partitions[0],target,attribute_list,profondeur+1)
    right = construction_arbre(partitions[1],target,attribute_list,profondeur+1)

    return Node(split,attribute,right,left,prediction)


#ici on cherche l'algo 2 du tp2 section 3.2
def splitting_attribute_quartile_version(data,target,attribute_list):
    h_total = entropie(data[target])
    best_atribute = 0
    best_gain = 0
    best_split = 0
    best_partitions = []
    quartile_list = [0.25,0.50,0.75]
    for a in attribute_list:
        sorted_data = data.sort_values(by=a) # on tri par l'attribut en entrer
        classe_init = sorted_data[target][0]
        for i in sorted_data[target]:
            split = 0
            if i != classe_init:
                for n in quartile_list:
                    split = sorted_data[a].quantile(n)
                    compare = sorted_data[a]
                    quartile = n
                    partitions   = [sorted_data[compare < split], sorted_data[compare >= split]]
                    E_partitions = [entropie(partitions[j][target]) for j in range(len(partitions))]
                    acc = 0.0
                    for j in range(len(partitions)):
                        acc += partitions[j].shape[0] / sorted_data.shape[0] * E_partitions[j]
                    
                    gain = h_total - acc
                    if gain > best_gain:
                        best_atribute = a
                        best_gain = gain
                        best_quartile = quartile
                        best_split = split
                        best_partitions = partitions        
            break
    return best_atribute,best_gain,best_quartile,best_split,best_partitions   