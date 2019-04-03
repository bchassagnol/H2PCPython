# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:40:17 2019

@author: bchassagno
"""

import pyAgrum as gum
import pandas as pd
import os
from hpc import hpc


class useH2PC ():
    """H2PC is a new hybrid algorithm combining scoring and constraint-structured learning,
    which can be considered as an improvement of MMHC in many regards. Especially, it clearly enables
    to reduce the number of false negative edges.   
    
    
    """
    def __init__(self, filename,learner,seuil_p_value=0.05):
        #check if file is present, if instance of the parameter is correct and the file's extension
        if isinstance(filename, str):
            self.filename=filename
        else:
            raise TypeError("le format attendu pour le fichier d'entrée est de type string")
            
        
        
        if not os.path.isfile(filename):
            raise FileNotFoundError("fichier non trouve a l'emplacement attendu")
        
        
        _, extension=os.path.splitext(self.filename)
        
        if extension!=".csv":
            raise TypeError("le format attendu pour le fichier d'entrée est de type .csv")
        self.seuil_pvalue=seuil_p_value
        if isinstance(learner,gum.pyAgrum.BNLearner):
            self.learner=learner 
        else:
            raise TypeError("le format attendu pour l'apprentissage est pyAgrum.BNLearner")
            
        self.database=pd.read_csv(self.filename,header=0)
        self.variables=list(self.database)
        
    def learnBN(self):
        dico_couverture_markov={}
        for target in self.variables: 
            print("on calcule la couverture de la variable ",target)
            dico_couverture_markov[target]=hpc(target,self.database,self.variables,self.learner).couverture_markov()
        print(dico_couverture_markov)
        
            
        
        
    
    
    