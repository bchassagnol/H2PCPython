# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:00:52 2019

@author: bchassagno
"""
import pyAgrum as gum
import pandas as pd

from itertools import product
from collections import OrderedDict

class hpc():
    """
    Hpc enables to learn the markov blanket around a target node, 
    that's to say the set of spouses, children and parents around it. 
    
    """
    def __init__(self,target,database,variable_set,learner,seuil_pvalue=0.05,verbosity=False):
        self.learner=learner        
        self.target=target
        self.database=database
        self.variable_set=set(variable_set)
        self.seuil_pvalue=seuil_pvalue
        self.parent_set=set()
        self.spouses_set=set()
        self.verbosity=verbosity
        self.d_separation={}
        
    
    def couverture_markov(self):
         self._DE_PCS()
         self._DE_SPS()
         #voisinage=self._DE_SPS().union(self._DE_PCS()).copy()
         #PC=FDR_IAPC(voisinage)
         return self.parent_set.union(self.spouses_set)
            
            
    def _DE_PCS(self):
        
        parent_set=self.variable_set.copy()
        parent_set.remove(self.target)
        old_parent_set=parent_set.copy()
        
        #teste les dependances une a une de toutes les variables avec la variable cible
        #si independant, on l'enleve du set des parents et enfants
        for variable in old_parent_set:            
            stat,pvalue=self.learner.chi2(self.target,variable)
           
            if self.verbosity:
                self.testIndepFromChi2(self.target,variable,self._isIndep(pvalue))
            
            if self._isIndep(pvalue):
                parent_set.remove(variable)
                self.d_separation[variable]={}
                
        old_parent_set=parent_set.copy()
        
        #teste les independances conditionnelles, si indepandant de la cible conditionnellement
        #avec une des variables enfants ou parents, on l'enleve du voisinage de la cible
        for variable in old_parent_set:            
            for condition in old_parent_set.difference({variable}):  
                stat,pvalue=self.learner.chi2(self.target,variable,[condition])
               
                if self.verbosity:
                    self.testIndepFromChi2(self.target,variable,self._isIndep(pvalue),condition)
               
                if self._isIndep(pvalue):
                    parent_set.remove(variable)             
                    self.d_separation[variable]={condition}   
                    break
        
        self.parent_set=parent_set
        
    
    def _DE_SPS(self):
        
        for variable in self.parent_set:
            #partie 1:detection des conjoints de la cible (enfants communs)
            spouses_x=set()
            set_externe=self.variable_set.difference({self.target}.union(self.parent_set)).copy()            
            for variable_extern in set_externe:                
                condition={variable}.union(self.d_separation[variable_extern])
                condition=list(condition)
                            
                stat,pvalue=self.learner.chi2(self.target,variable_extern,condition)
                if not self._isIndep(pvalue):
                    spouses_x=spouses_x.union({variable_extern})
                
                if self.verbosity:
                    self.testIndepFromChi2(self.target,variable_extern,self._isIndep(pvalue),condition)
             
            #partie 2: suppression des conjoints eux-mêmes ancêtres ou descdendants d'autres conjoints 
            for spouse in spouses_x:                
                for spouse_intern in spouses_x.difference({spouse}):                       
                    condition={variable}.union({spouse_intern})
                    condition=list(condition)                    
                    stat,pvalue=self.learner.chi2(self.target,spouse,condition)
                    if self.verbosity:
                        self.testIndepFromChi2(self.target,spouse,self._isIndep(pvalue),condition)
                    if self._isIndep(pvalue):
                        spouses_x=spouses_x.difference({spouse})
                        break         
            self.spouses_set=self.spouses_set.union(spouses_x)
            
    def _IAMBFDR(self,voisinage,MB_cible,dico_p_value):
        
        for neighbour in voisinage:
            condition=voisinage.difference({neighbour})
            stat,p_value=self.learner.chi2(self.target,neighbour,list(condition))
            dico_p_value[neighbour]=p_value
            #sort dictionnary by increasing p_value
            dico_p_value = OrderedDict(sorted(dico_p_value.items(), key=lambda t: t[1]))
        """
        somme_pvalue=0.0
        for sorted_neighbour in dico_p_value.keys():
            if sorted_neighbour in MB_cible:
        """        
            
            
    def _FDR_IAPC(self,voisinage):
        MB_cible=self._IAMBFDR(voisinage,MB_cible=set(),dico_p_value={})
            
             
                    
                
                
    
            
    def _isIndep(self,pvalue):
        return pvalue>=self.seuil_pvalue

    def testIndepFromChi2(self,var1,var2,result_test,kno=[]):
        """
        Just prints the resultat of the chi2
        """        
        
        if len(kno)==0:
            print("From Chi2 tests, is '{}' indep from '{}' ==> {}".format(var1,var2,result_test))
        else:
            print("From Chi2 tests, is '{}' indep from '{}' given {} : {}".format(var1,var2,kno,result_test))       
  
if __name__ == "__main__":    
    learner=gum.BNLearner("test.csv")    
    database=pd.read_csv("test.csv" ,header=0)   
    variable_set=list(database)
    couverture_markov_variable=hpc('VENTLUNG',database,variable_set,learner,verbosity=False)
    print(couverture_markov_variable.couverture_markov())
   
    
    
    
    
   
    
 