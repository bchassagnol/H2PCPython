# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:00:52 2019

@author: bchassagno
"""
import pyAgrum as gum
import pandas as pd

import itertools 
from collections import OrderedDict
from fractions import Fraction
import os

class hpc():
    """
    Hpc enables to learn the markov blanket around a target node, 
    that's to say the set of spouses, children and parents around it. 
    
    """
    def __init__(self,target,learner,seuil_pvalue=0.05,verbosity=False,whitelisted=set(),blacklisted=set(),known_good=set(),known_bad=set()):
        if isinstance(learner,gum.pyAgrum.BNLearner):
            self.learner=learner 
        else:
            raise TypeError("le format attendu pour l'apprentissage est pyAgrum.BNLearner")            
        self.target=target        
        self.variable_set=set(learner.names())
        self.seuil_pvalue=seuil_pvalue        
        self.verbosity=verbosity
        
        #whitelisted is the set of nodes linked by an edge to our target
        #bkacklisted is the set of nodes which shouldn't be linked by an edge to our target
        self.whitelisted={self._is_node_linked(arc) for arc in whitelisted if self._is_node_linked(arc) is not None}   
        self.blacklisted={self._is_node_linked(arc) for arc in blacklisted if self._is_node_linked(arc) is not None}
        
        self.known_good=known_good
        self.known_bad=known_bad
        
       
        
    
    def _is_node_linked(self,arc):        
        if (arc[0]==self.target):            
            return arc[1]
        elif (arc[1]==self.target):
            return arc[0]
        
 
    
    def couverture_markov(self):
         #dictionnary here is used to store clearly both neighbours and balnket markov, 
         #according to the volunty's user
         markov_dictionnary={"neighbours":set(),"blanket_markov":set()}
         PCS,d_separation,p_values=self._DE_PCS()
         print("le set maximal des paretns est ", PCS)
          #optimisation : 0 or 1 node in PCS --> PC == PCS
         if(len(PCS) < 2):
             markov_dictionnary["neighbours"],markov_dictionnary["blanket_markov"]=PCS,PCS                             
             return(markov_dictionnary)
          # 2. [RSPS] Search remaining spouses superset, those not already in PCS
          
   
         SPS=self._DE_SPS(PCS,d_separation)
         """
         super_voisinage=PCS.union(SPS).copy() 
          #optimisation : 2 nodes in PC and no SP --> PC == PCS
         if len(super_voisinage)<3:
             #in that case, impossible to have found a spouse, as it is required to have at least 2 parents to keep on in the previous phase
             markov_dictionnary["neighbours"],markov_dictionnary["blanket_markov"]=PCS,PCS                             
             return(markov_dictionnary)
           
          # 3. [PC] Get the Parents and Children from nodes within PCS and RSPS
         PC=self._FDR_IAPC(super_voisinage,self.target)
         
         
          # 4. [Neighbourhood OR] Detect and add false-negatives to PC, by checking if
          #     the target is present in potential neighbours' neighbourhood         
         for par_child in PCS.difference(PC):
             #determine set of potential candidates
             
             voisinage_par_child=super_voisinage.union({self.target}).difference({par_child})  
             if self.target in self._FDR_IAPC(voisinage_par_child,par_child):
                 PC=PC.union({par_child}).copy()
                 
         markov_dictionnary["neighbours"],markov_dictionnary["blanket_markov"]=PC,PCS.union(SPS)                             
         return(markov_dictionnary)    
         """
         
    
    
       
             
            
            
    def _DE_PCS(self):
        
        parent_set=self.variable_set.copy()
        #we remove from the neighbours the blacklisted nodes and the target
        parent_set=parent_set.difference({self.target},self.blacklisted)
        known_good=self.known_good.copy()  
        d_separation={}
        dict_p_values={}
        
        nodes_to_check=parent_set.difference(known_good.union({self.target},self.whitelisted,self.blacklisted))
        
     
        
        # Phase (I): remove X if Ind(Target,variable) (0-degree d-separated nodes)
        for variable in nodes_to_check:  
            stat,pvalue=self.learner.chi2(self.target,variable)
            """
            if self.verbosity:
                self.testIndepFromChi2(self.target,variable,self._isIndep(pvalue))
            """
            if self._isIndep(pvalue):
                parent_set.remove(variable)
                d_separation[variable]=set()
                """
                if self.verbosity:
                    #self.testIndepFromChi2(self.target,variable,self._isIndep(pvalue))
                    print("node '{}' is removed of the markov blanket".format(variable))
                """
               
            else:
                dict_p_values[variable]=pvalue
                
        #update d-separation for blacklisted nodes (known to be not neighbours, but potentially spouses)
        for bad_node in self.blacklisted:
            d_separation[bad_node]=set()
                
        
       # Phase (II): remove X if Ind(T,X|Y) (1-degree d-separated nodes)         
              
        # heuristic 1 : sort the PC candidates to potentially remove in decreasing p-value order
        # this way we are more prone to remove less correlated nodes first, and thus leave the loop quicklier
        reversed_order_dictionnary = OrderedDict(sorted(dict_p_values.items(), key=lambda t: t[1],reverse=True))
        
        
        new_nodes_to_check_against=self.whitelisted.union(known_good)        
    
        for variable in reversed_order_dictionnary.keys():  
             # heuristic 2 : sort the d-separating canditates in increasing p-value order
             # this way we are more prone to remove with highly correlated nodes first, which brought the most information
            ordered_dictionnary_new_nodes={new_node:0 for new_node in new_nodes_to_check_against}
            ordered_dictionnary = OrderedDict(sorted(dict_p_values.items(), key=lambda t: t[1]))
            ordered_dictionnary.update(ordered_dictionnary_new_nodes)            
            del(ordered_dictionnary[variable])
            
         
            for condition in ordered_dictionnary.keys():  
                stat,pvalue=self.learner.chi2(self.target,variable,[condition])
                """
                if self.verbosity:
                    self.testIndepFromChi2(self.target,variable,self._isIndep(pvalue),[condition])
                """
               
                if self._isIndep(pvalue):
                    #if conditionnaly independant, we remove the node from the blanket markov
                    #secondly, we update the pvalues database (remvoing the one corresponding to the deleted node)
                    """
                    if self.verbosity:
                        print("node '{}' is removed from the MB, as conditionnaly independant by '{}' ".format(variable,condition))
                    """
                    parent_set.remove(variable)  
                    
                    d_separation[variable]={condition}
                    del(dict_p_values[variable])
                    break
                else:
                    if pvalue>dict_p_values[variable]:
                        dict_p_values[variable]=pvalue
                    
        if self.verbosity:
            print("superset of parents is '{}' ".format(parent_set))
        
        return parent_set,d_separation,dict_p_values
        
    
    def _DE_SPS(self,parent_set,d_separation):
        
        spouses_set=set()
        #check if parent set is empty, if not, we enter the loop
        #in python, an empty sequence will always be a false variable
        
                 
        for variable in parent_set:
            # Phase (I): search spouses Y in the form T->X<-Y from the
            # remaining nodes (not in pcs)            
            spouses_x=set()
            pval_x={}
            set_externe=self.variable_set.difference({self.target}.union(parent_set))          
            for variable_extern in set_externe:  
                # optimisation : avoid irrelevant tests
                if variable in d_separation[variable_extern]:
                    #no need to go further, as this means that variable_extern was already d-separated fromt he set by x
                    #and so the result would be independant
                    next
                condition=list({variable}.union(d_separation[variable_extern]))
               
                            
                stat,pvalue=self.learner.chi2(self.target,variable_extern,condition)
                if self.verbosity:
                    self.testIndepFromChi2(self.target,variable_extern,self._isIndep(pvalue),condition)
                if not self._isIndep(pvalue):
                    spouses_x=spouses_x.union({variable_extern})
                    pval_x[variable_extern]=pvalue
                    if self.verbosity:
                        print("node '{}' is added to the set of spouses by '{}' ".format(variable_extern,variable))
               
                
                
            # # heuristic : sort the candidates in decreasing p-value order
            # this way we are more prone to remove less correlated nodes first
            ordered_pvalx=OrderedDict(sorted(pval_x.items(), key=lambda t: t[1],reverse=True))
             
            
            # Phase (II): remove false positive spouses Y in the form T->X<-Z<-...<-Y
            # (descendants or ancestors of spouses Z)
            for spouse in ordered_pvalx.keys():
                temp_ordered_pvalx=OrderedDict(ordered_pvalx)
                del(temp_ordered_pvalx[spouse])
                for spouse_intern in temp_ordered_pvalx:                       
                    condition=list({variable}.union({spouse_intern}))
                    stat,pvalue=self.learner.chi2(self.target,spouse,condition)
                    
                    if self.verbosity:
                        self.testIndepFromChi2(self.target,spouse,self._isIndep(pvalue),condition)
                    
                    if self._isIndep(pvalue):
                        if self.verbosity:
                            print("node '{}' is removed from the set of spouses by '{}' ".format(spouse,spouse_intern))
                        spouses_x=spouses_x.difference({spouse})
                        break            
            spouses_set=spouses_set.union(spouses_x)      
        print("set of spouses is ",spouses_set)
        return(spouses_set)
            
    def _IAMBFDR(self,target,voisinage):
        # whitelisted nodes are included by default (if there's a direct arc
        # between them of course they are in each other's markov blanket).
        
        # known good nodes are included by default 
        MB_cible=self.whitelisted.union(self.known_good)
        
        #stock several neighbourhoods computed
        mb_storage=[]
        
        #nodes responsible for already determined markov blankets
        culprit=set()
        current_included_node=None
        current_excluded_node=None
        
        #we restrain heuristically the set of combinations by this condition
        limit_iterations=len(voisinage)*100
        number_iterations=0
        
        #fdr rate
        somme_indice=self._somme_indice(len(voisinage)) 
        dico_p_value={}
        
        
        
        
        while (number_iterations<limit_iterations):
            #re_intialize data
            #previsous_exclusion enables to know if there was or not a node excluded
            #if yes, we skip the phase of inclusion
            previous_exclusion=False
            #check if markov found has been modified, if not, we stop looping
            markov_change=False
            
           
            #1) p-value calculation
            for neighbour in voisinage:
                condition=MB_cible.difference({neighbour})
                stat,p_value=self.learner.chi2(target,neighbour,list(condition))
                dico_p_value[neighbour]=p_value
                #sort dictionnary by increasing p_value
            dico_p_value = OrderedDict(sorted(dico_p_value.items(), key=lambda t: t[1]))
            """
            if self.verbosity:
                 print("P-values ordered are '{}' ".format(dico_p_value))
            """
               
            
            #2) neighbour exclusion
            for (index,sorted_neighbour) in enumerate(dico_p_value.keys()):
                #here we want to avoid the case where we exclude a node, that was included just before
                #+whitelisted and known.good nodes cannot be removed
                candidates_to_remove=MB_cible.difference({current_included_node}.union(self.known_good,self.whitelisted))
                if sorted_neighbour in candidates_to_remove:
                    
                    controle_pvalue=dico_p_value[sorted_neighbour]*(len(sorted_neighbour)/(index+1))*somme_indice
                    if controle_pvalue>self.seuil_pvalue:
                        current_excluded_node=sorted_neighbour
                        current_included_node=None
                        MB_cible=MB_cible.difference({sorted_neighbour})
                        previous_exclusion=True
                        markov_change=True                     
                        """
                        if self.verbosity:
                            print("current excluded node is '{}' ".format(current_excluded_node))
                        """
                        break
                        
            #3) neighbour inclusion
            if not previous_exclusion:                
                for (index,sorted_neighbour) in enumerate(dico_p_value.keys()):
                    #here we want to avoid the case where we include a node that was excluded just before
                    #+ remove nodes which are supposed not to belong to the markov blanket               
                    
                    
                    possible_candidates_to_add=voisinage.difference(MB_cible.union(culprit,self.known_bad))                 
                    if sorted_neighbour in possible_candidates_to_add:                
                        controle_pvalue=dico_p_value[sorted_neighbour]*(len(sorted_neighbour)/(index+1))*somme_indice
                        if controle_pvalue<=self.seuil_pvalue:
                            #print("on ajoute la variable ",sorted_neighbour)
                            markov_change=True
                            current_excluded_node=None
                            current_included_node=sorted_neighbour
                            MB_cible=MB_cible.union({sorted_neighbour})  
                            """
                            if self.verbosity:
                                print("current included node is '{}' ".format(current_included_node))
                            """
                            break
                            
                
            #4) control of potentially repeated blankets
            #if markov_change is still false, then this means that there was neither inclusion nor exclusion of new nodes
            #we can then stop the while loop
            if not markov_change:                
                break
            
            #print("les couvertures trouvees actuellement sont les suivantes ",mb_storage)
            if MB_cible in mb_storage:
                #we also remove the markov blanket found, as it was already computed
                #lists in Python are sorted by order of inclusion, so they can be used as stacks
                mb_storage.pop()
                number_iterations-=1
                culprit.add(current_excluded_node)
                culprit.add(current_included_node)                   
                if self.verbosity:
                    print("current repeated blanket alreay found is '{}' ".format(MB_cible))
            else:
                """
                if self.verbosity:
                    print("current blanket found is '{}' ".format(MB_cible))
                """
                mb_storage.append(MB_cible)
            #end of a complete inclusion or exclusion of a node, we compute then another set of blankets
            #we re-initialise all the values
            number_iterations+=1
            
        
        return MB_cible
        
                
      
    def _somme_indice(self,nb_variables):
        somme=Fraction()
        for indice in range (1,nb_variables):
            somme+=Fraction(1/indice)
        return float(somme)
    
    def _filter_hybrid(self,MB_cible,target):
       #filter the markov boundary got with iambfdr
            
        MB_filtered=MB_cible.copy()
        for potential_spouse in MB_cible:
            #generate all combinations of conditions
            #sort by decreasing set length to potentially stop quicklier?
            condition_set=self._powerset(MB_cible.difference({potential_spouse}))
            for condition in condition_set:
                stat,p_value=self.learner.chi2(target,potential_spouse,condition)                  
                if p_value>=self.seuil_pvalue:                    
                    if self.verbosity:
                        print(" node", potential_spouse, "is not anymore a neighbour of", target, " based on condition ",condition)                      
                    MB_filtered=MB_cible.difference(potential_spouse).copy()
                    #if yes, we stop here, as node considered won't respect anymore the basic definition of markov boundary 
                    break 
        return MB_filtered
            
    def _FDR_IAPC(self,voisinage,target): 
        """# Build the parents and children (PC) set of a node from it's parents and
        # children superset (PCS) and it's remaining spouses superset (RSPS).
        """
        #learn markov boundary of target     
        MB_cible=self._IAMBFDR(target,voisinage)
        MB_filtered=self._filter_hybrid(MB_cible,target)
                       
        return MB_filtered
            
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
  

    def _powerset(self,iterable):    
        s = list(iterable)
        all_combi=list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))
        liste_all_combinations=[list(i) for i in all_combi]
        return liste_all_combinations
    



if __name__ == "__main__":  
    true_bn=gum.loadBN(os.path.join("true_graphes_structures","alarm.bif"))
    
    gum.generateCSV(true_bn,"sample_alarm.csv",20000,False)
    learner=gum.BNLearner("sample_alarm.csv")     
    object_hpc=hpc('VENTTUBE',learner,verbosity=True) 
    object_hpc.couverture_markov()
    
    object_hpc.learner.chi2('VENTTUBE','CATECHOL',['VENTMACH','VENTALV'])

    
    
    
   
    

    
   
    print(hpc('VENTTUBE',learner,verbosity=True).variable_set)
  
    
    
    
    
   
    

   
    
 