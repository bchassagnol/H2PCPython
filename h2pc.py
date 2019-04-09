# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:40:17 2019

@author: bchassagno
"""

import pyAgrum as gum
import pandas as pd
import os
from hpc import hpc
import itertools 
import pickle
import pprint as pp


class H2PC ():
    """H2PC is a new hybrid algorithm combining scoring and constraint-structured learning,
    which can be considered as an improvement of MMHC in many regards. Especially, it clearly enables
    to reduce the number of false negative edges.   
    
    
    """
    def __init__(self, learner,seuil_p_value=0.05,verbosity=False,score_algorithm="MIIC"):
        #check if file is present, if instance of the parameter is correct and the file's extension
        """
        if isinstance(filename, str):
            self.filename=filename
        else:
            raise TypeError("le format attendu pour le fichier d'entrée est de type string")
            
        
        
        if not os.path.isfile(filename):
            raise FileNotFoundError("fichier non trouve a l'emplacement attendu")
        
        
        
        _, extension=os.path.splitext(self.filename)
        
        if extension!=".csv":
            raise TypeError("le format attendu pour le fichier d'entrée est de type .csv")
        """
        self.seuil_pvalue=seuil_p_value
        if isinstance(learner,gum.pyAgrum.BNLearner):
            self.learner=learner 
        else:
            raise TypeError("Format expected for learning is pyAgrum.BNLearner")
        self.variables=learner.names()
        self.verbosity=verbosity
        self.score_algorithm=score_algorithm
        
    def check_consistency(self,dictionnary_neighbourhood):
        #initialize dictionnary of empty sets
        consistent_dictionnary_neighbourhood={k: set() for k in self.variables}
    
        for couple in itertools.combinations(dictionnary_neighbourhood.keys(),2): 
            
            variable_1,variable_2=couple 
            neighbourhood_variable1=dictionnary_neighbourhood[variable_1].copy()
            neighbourhood_variable2=dictionnary_neighbourhood[variable_2].copy()
            if (variable_1 in neighbourhood_variable2) and (variable_2 in neighbourhood_variable1):                
                #under the assumption of exctness of tests, if variable 1 is in neighbourhood of variable 2
                #is equivalent that variable 2 is in neighbourhood of variable 1
                consistent_dictionnary_neighbourhood[variable_1]=consistent_dictionnary_neighbourhood[variable_1].union({variable_2})
                consistent_dictionnary_neighbourhood[variable_2]=consistent_dictionnary_neighbourhood[variable_2].union({variable_1})
             
        return consistent_dictionnary_neighbourhood
       
        
    def _unique_edges(self,consistent_dictionnary):
        set_unique_edges=set()
        for variable in consistent_dictionnary.keys():
            #to check if neighbourhood is not empty
            if consistent_dictionnary[variable]:               
                for neighbour in consistent_dictionnary[variable]:
                    edge=(variable,neighbour)
                    set_unique_edges.add(edge)
        return (set_unique_edges)
     
    def _add_set_unique_possible_edges(self,unique_possible_edges):
        for unique_edge in unique_possible_edges:
            self.learner.addPossibleEdge(*unique_edge)
           
     
    def _apply_score_algorithm(self):
        possible_algorithm = {'MIIC': self.learner.useMIIC(),'Greedy_climbing':self.learner.useGreedyHillClimbing(),'3off2':self.learner.use3off2()}
        return possible_algorithm.get(self.score_algorithm,'algorithm still not implemented')
    def learnBN(self):
        dico_couverture_markov={}
        #computation of local neighbourhood for each node   
        """        
        for target in self.variables:    
            dico_couverture_markov[target]=hpc(target,self.learner,verbosity=False).couverture_markov()            
            if self.verbosity:
                print("We compute with HPC the markov blanket of '{}' : '{}' \n\n".format(target,dico_couverture_markov[target]))
        
        with open('dictionnary', 'wb') as fichier:
             mon_pickler = pickle.Pickler(fichier)
             mon_pickler.dump(dico_couverture_markov)
        
        """
        with open('dictionnary', 'rb') as fichier:
            mon_depickler = pickle.Unpickler(fichier)
            dico_couverture_markov = mon_depickler.load()
        
        
        
        consistent_dictionnary=self.check_consistency(dico_couverture_markov)
        #print("le dico apres verification consistence est ",pp.pprint(consistent_dictionnary,width=1))
       
        
        unique_possible_edges=self._unique_edges(consistent_dictionnary)
        #add set of unique edges as unique possible addings for h2pc
        self._add_set_unique_possible_edges(unique_possible_edges)
        
        #score_based learning according to input_score
        self._apply_score_algorithm()
    
        
        
        
        
        
        
if __name__ == "__main__":    
    learner=gum.BNLearner("test.csv") 
    essai_graphe=H2PC(learner,verbosity=False)
    essai_graphe.learnBN()
    
    

    

    
 
    
    
    
        
            
        
        
    
    
    