# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:20:18 2019

@author: bchassagno
"""


import pyAgrum as gum
from h2pc import H2PC


learner=gum.BNLearner("test.csv")
aprentissage_h2pc=H2PC(learner,verbosity=True)
aprentissage_h2pc.test()







