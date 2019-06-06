# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:18:08 2019

@author: Bastien
"""
import pandas as pd
import pyAgrum as gum
from sklearn import preprocessing
import numpy as np
from scipy import stats 
import numbers
import math
from sklearn.utils import shuffle, resample


class indepandances ():
    
    def __init__(self,ind_x,ind_y,df,conditions=None,type_test="x2",method_pyagrum=True,learner=None,debug=False,power_rule=None,verbosity=False):
        if not isinstance(df,pd.core.frame.DataFram):
           raise TypeError ("expected format is a dataframe")
        else:
            if df.isnull().values.any():
                raise ValueError ("we can't perform tests on databases with missng values")
            else:
                self.df=df
                self.nrows=len(df.index)
        
        if isinstance(ind_x,int) and ind_x<len(df.columns) and ind_x>=0:
                self.ind_x=ind_x
        else:
                raise TypeError("Format expected for indexing is integer included between 0 and number of features")
                
        if isinstance(ind_y,int) and ind_y<len(df.columns) and ind_y>=0:
                self.ind_y=ind_y
        else:
                raise TypeError("Format expected for indexing is integer included between 0 and number of features")
         
        #in Python, this condition only works if conditions is implemented
        if conditions:
            for indice in conditions:
                if not isinstance(indice,int):
                    raise TypeError ("expected format is a list of integer indexes")
                elif (indice==ind_x or indice==ind_y):
                    raise ValueError("duplication of either x or y in the condition")
            self.ind_z=conditions
        else:
            self.ind_z=[]
            
        liste_tests=["x2","x2_adjusted","x2_perm","x2_perm_improved","x2_sp","G2","G2_adjusted","G2_perm","G2_perm_adjust","G2_sp"]
        if type_test in liste_tests:
            self.type_test=type_test
        else:
            raise ValueError("Possible tests are included in these tests: {}".format(type_test))
            
        if isinstance (method_pyagrum,bool):
            if type_test in ["x2","G2"]:
                self.method_pyagrum=method_pyagrum
            else:
                raise ValueError("Only valuable tests are G2 and x2 with Pyagrum libraries")
        else:
            raise TypeError("Format expected for method pyagrum must be boolean.")
            
        if method_pyagrum:
            if isinstance(learner,gum.pyAgrum.BNLearner):
                learner=self.learner
            else:       
                raise TypeError("Format expected for method pyagrum is pyAgrum.BNLearner, when Pyagrum method is activated.")
        
        if isinstance(debug,bool):
            self.debug=debug
        else:
            raise TypeError("Expected format is boolean for debug.")
            
        if power_rule:
            if isinstance (power_rule,numbers.Number):
                self.power_rule=power_rule
                
        if isinstance (verbosity, bool):
            self.verbosity=verbosity
        else:
            raise TypeError("Expected format for verbosity is boolean")
        
                
        self.levels_x=0
        self.levels_y=0
        if conditions:
            self.levels_z=0
            
    def column_treatment(self):
        column_x=self.df.iloc[:,self.x].values
        column_y=self.df.iloc[:,self.y].values
        
        #convert each factor as a class of integers
        #role of compression
        le = preprocessing.LabelEncoder()
        column_x=le.fit_transform(column_x)
        column_y=le.fit_transform(column_y)
        
        #create a dataframe combining the 3 columns
        if self.ind_z:
            column_z=np.array(self.df.iloc[:,self.ind_z[0]].values)
            for indice in self.ind_z[1:]:
                column_z=np.array([x1 + x2  for x1,x2 in zip(column_z,self.df.iloc[:,indice].values)])
            condition_df = {'X':column_x,'Y':column_y,'Z':le.fit_transform(column_z)}
            condition_df=pd.DataFrame.from_dict(condition_df)
            #compute nlevels of x, y and z
            self.levels_x,self.levels_y,self.levels_z=len(condition_df['X'].unique()),len(condition_df['Y'].unique()),len(condition_df['X'].unique()),len(condition_df['Z'].unique())
        else:
            condition_df = {'X':column_x,'Y':column_y}
            condition_df=pd.DataFrame.from_dict(condition_df)
            self.levels_x,self.levels_y=len(condition_df['X'].unique()),len(condition_df['Y'].unique())
        return condition_df
        
    def apply_heuristic_one(self):
        if self.z:
            return self.nrows < (self.power_rule * self.levels_x * self.levels_y * self.levels_z)             
        else:
            return self.nrows < (self.power_rule * self.levels_x * self.levels_y)                
     
        
    def compute_chi2_statistic(self,group,adjust_degree=False,total_df=None):    
        effectif_z=len(group)   
        effectif_observe_by_z=np.array(pd.crosstab(group['X'],group['Y']))    
        effectif_xz=np.sum(joint_table_by_z,axis=1,keepdims =True)
        effectif_yz=np.sum(joint_table_by_z,axis=0,keepdims=True)  
        if self.verbosity:
            print("contingency table for variable Z is {}, so subtotals for XZ are {} anf for YZ are {} : ".format(effectif_observe_by_z,effectif_xz,effectif_yz))
        
        effectif_theorique_by_z=np.dot(effectif_xz,effectif_yz)/effectif_z         
        chi2_stat_by_z=np.sum((effectif_theorique_by_z-effectif_observee_by_z)**2/effectif_theorique_by_z)    
        if self.adjust_degree:
            #adjust with null XZ frequencies
            total_df-=((self.levels_x-effectif_observe_by_z.shape[0])*(self.levels_y-1))
            #adjust with null YZ frequencies
            total_df-=((self.levels_y-effectif_observe_by_z.shape[1])*(self.levels_x-1))
            return chi2_stat_by_z,total_df
            if self.verbosity:
                print("Current degree of freedom is {} ",total_df)
        
        return chi2_stat_by_z
    
    
        
    
    def chi2_classic_test(self,condition_df):
        if self.ind_z:
            grouped_by_z=condition_df.groupby(['Z'])
            statistic=grouped.agg(self.compute_chi2_statistic).sum()['X']
            df=(self.levels_x-1)*(self.levels_y-1)*self.levels_z
            p_value=1-stats.chi2.ppf(statistic,df)
            return (statistic,df,p_value)
        else:
            contingency_table=pd.crosstab(condition_df['X'],condition_df['Y'])
            chi2, p, dof, expected_frequency=stats.chi2_contingency(contingency_table,correction=False)
            return (chi2, dof, p)
     
        
    def chi2_adjusted_test(self,condition_df):
          if self.ind_z:
            grouped_by_z=condition_df.groupby(['Z'])
            adjusted_df=(self.levels_x-1)*(self.levels_y-1)*self.levels_z
            chi2=0
            for  name,group in condition_df.groupby(['Z']):
                if self.verbosity:
                    print("Studied Z group is ", name)
                chi_temp,adjusted_df=self.compute_chi2_statistic(group,adjust_degree=True,total_df=adjusted_df)
                chi2+=chip_temp            
            
            p_value=1-stats.chi2.ppf(statistic,adjusted_df)
            return (chi2,dadjusted_df,p_value)
          else:               
            contingency_table=pd.crosstab(condition_df['X'],condition_df['Y'])
            #here adjustement is done with yate's suggestion
            chi2, p, dof, expected_frequency=stats.chi2_contingency(contingency_table,correction=True)
            return (chi2, dof, p)
        
    def chi2_permutation(self,condition_df,permutation_number=5000,semi_parametric_option=False):
        #chi2 permutation with condition
        if self.ind_z:
            #first compute chi2 statistic in the original database
            chi2_original=self.chi2_classic_test(condition_df)[0]
            #store stats for each permuation
            chi2_permut_vector=np.empty(permutation_number)
            for permutation in range (permutation_number):
                if self.verbosity:
                    print("we are in {} ".format(permutation))
                chi2_permut=0            
                for  name,group in condition_df.groupby(['Z']):                     
                   temp_group=group.copy()
                    #we shuffle X column, by groups of similar Z, to assert that subtotals remain the same                   
                   temp_group['X']=np.random.permutation(temp_group['X'].values)
                   chi2_permut+=self.compute_chi2_statistic(temp_group)
                chi2_permut_vector[permutation]=chi2_permut
            
        
        #testing non conditionnal independance chi2
        else:               
            #first compute chi2 statistic in the original database
            chi2_original=self.chi2_classic_test(condition_df)[0]
            #store stats for each permuation
            chi2_permut_vector=np.empty(permutation_number)
            for permutation in range (permutation_number):
                if self.verbosity:
                    print("we are in {} ".format(permutation))
                temp_df=condition_df.copy()
                temp_df['X']=np.random.permutation(temp_df['X'].values)
                chi2_permut=self.chi2_classic_test(condition_df)[0]
                chi2_permut_vector[permutation]=chi2_permut
        if semi_parametric_option:
            df_adjusted=np.mean(chi2_permut_vector)
            return (chi2_original,df_adjusted, 1-stats.chi2.ppf(chi2_original,df_adjusted))
        else:
            return (chi2_original,None,np.count_nonzero(chi2_original<chi2_permut_vector)/permutation_number)
        
    def chi2_adjustable_permutation(self,condition_df):
        
    def chi2_semi_parametric(self,condition_df):
        self.chi2_permutation(condition_df, semi_parametric_option=True)
        
    
    def  chi2_test(self):        
        if self.method_pyagrum:
            return self.learner.chi2(self.x,self.y,self.z)
        else:
            #get the corresponding column for each variable
            condition_df=self.column_treatment()
            #check validation of heuristic one, by default, if not enough values are present, we suppose nodes are inependant
            if self.apply_heuristic_one():
                return (math.inf, None, 1)
            else:                                 
                method_chi2={"x2":chi2_classic_test,"x2_adjusted":chi2_adjusted_test,"x2_perm":chi2_adjustable_permutation,"x2_perm_improved":chi2_adjustable_permutation,"x2_sp":chi2_semi_parametric}
                return method_chi2[self.type_test](condition_df)
            

                    
            
    def G2_test(self):
        if self.method_pyagrum:
            return self.learner.chi2(self.x,self.y,self.z)
        else:
            #get the corresponding column for each variable
            condition_df=self.column_treatment()     
 
    
    def realize_test(self):        
        
        if "x2" in self.type_test:
            return self.chi2_test()
        else:
            return self.G2_test()        
        
            
        
            
        
"""                
a = np.array(["foo", "foo", "foo", "foo", "bar", "bar","bar", "bar", "foo", "foo","foo","foo","foo"], dtype=object)
b = np.array(["one", "one", "one", "two", "one", "one", "one", "one", "two", "two","two","two","three"], dtype=object)
c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny",  "shiny", "dull", "shiny", "shiny","mince","shiny","mince"], dtype=object)  
condition_df = pd.DataFrame(columns=['X','Y','Z'])   
condition_df['X'],condition_df['Y'],condition_df['Z']=a,b,c

def compute_effectif_theorique(group):
    
    effectif_z=len(group)   
    joint_table_by_z=np.array(pd.crosstab(group['X'],group['Y']))
    print(pd.crosstab(group['X'],group['Y']))
    calculation_number=joint_table_by_z.shape[0]*joint_table_by_z.shape[1]    
    effectif_xz=np.sum(joint_table_by_z,axis=1,keepdims =True)
    effectif_yz=np.sum(joint_table_by_z,axis=0,keepdims=True)    
    print("les effectis sont {} et {} ".format( effectif_xz,effectif_yz))
    #replace with levels of x and y
    print(effectif_xz.shape,effectif_yz.shape)
    effectif_theorique_by_z=np.dot(effectif_xz,effectif_yz)/effectif_z
    print(effectif_theorique_by_z)
    effectif_observee_by_z=joint_table_by_z
    #effectif_theorique_by_z=np.reshape(np.dot(effectif_xz,effectif_yz)/effectif_z,calculation_number)    
    #effectif_observee_by_z=np.reshape(joint_table_by_z,calculation_number)       
    chi2_stat_by_z=np.sum((effectif_theorique_by_z-effectif_observee_by_z)**2/effectif_theorique_by_z)    
    print(chi2_stat_by_z)
    return chi2_stat_by_z


grouped=condition_df.groupby(['Z'])
groupe_dull=grouped.get_group('dull')
print(compute_effectif_theorique(groupe_dull))



for  name,group in condition_df.groupby(['Z']):    
    print("le nom du groupe est ", name, "et sa taille est ", len(group))
    compute_effectif_theorique(group)
    

    







forme_pandas=pd.crosstab(condition_df['X'],[condition_df['Z'],condition_df['Y']],dropna=False)
matrice_scores=np.array(forme_pandas)

print(forme_pandas)
mat_YZ=np.reshape(np.sum(matrice_scores,axis=0),(1,6))
print(mat_YZ)



good_format=np.reshape(matrice_scores,(3,2,2))
print(good_format)
np.sum(good_format,axis=2)
mat_XZ=np.reshape(np.sum(good_format,axis=1),(6,1))
print(mat_XZ)


forme_pandas=pd.crosstab(condition_df['X'],[condition_df['Y'],condition_df['Z']],dropna=False)
print(forme_pandas)



for  name,group in condition_df.groupby(['Z']):  
   print("group est {} et sa forme est {} ".format(name,group))
   print(pd.crosstab(group['X'],group['Y']))
   
np.random.RandomState()
for  name,group in condition_df.groupby(['Z']):  
   temp_group=group.copy()
   print("group est {} et sa forme avant tansformation est {} ".format(name,temp_group))
   temp_group['X']=np.random.permutation(temp_group['X'].values)
   print("group est {} et sa forme est {} ".format(name,temp_group))
   print(pd.crosstab(temp_group['X'],temp_group['Y']))
   
stats.norm.ppf(0.95)
"""    
