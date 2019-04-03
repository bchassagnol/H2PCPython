# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:20:18 2019

@author: bchassagno
"""
import os
os.chdir(r'C:\Users\bchassagno\Documents\stage_projet_bayesian_network\test_pyagrum')

import pyAgrum as gum
import pyAgrum.lib.bn_vs_bn as bvb
from h2pc import useH2PC

bn=gum.loadBN("alarm.bif")
gum.generateCSV(bn, "test.csv",10000)
learner=gum.BNLearner("test.csv")
learner.useMIIC()
bn2=learner.learnBN()
g=bvb.GraphicalBNComparator(bn,bn2)
g.scores()
# we create a quite large database
gum.generateCSV(bn,"sample.csv",100000,False)

aprentissage_h2pc=useH2PC("test.csv",learner)
aprentissage_h2pc.learnBN()







arc_1=gum.Arc(10,11)
arc_2=gum.Arc(20,21)
liste_arc=[arc_1,arc_2]
print(liste_arc[1].first())

#test independance
def isIndep(pvalue):
    return pvalue>=0.05

def testIndepFromChi2(learner,var1,var2,kno=[]):
    """
    Just prints the resultat of the chi2
    """
    stat,pvalue=learner.chi2(var1,var2,kno)
    print(stat,pvalue)
    if len(kno)==0:
        print("From Chi2 tests, is '{}' indep from '{}' ==> {}".format(var1,var2,isIndep(pvalue)))
    else:
        print("From Chi2 tests, is '{}' indep from '{}' given {} : {}".format(var1,var2,kno,isIndep(pvalue)))
    
