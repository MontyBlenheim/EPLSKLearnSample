'''
Created on 4 Sep 2018

@author: ANTM
'''

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.datasets import load_boston

from apama.eplplugin import EPLAction, EPLPluginBase

class TestPlugin(EPLPluginBase):
    def __init__(self,init):
        super(TestPlugin,self).__init__(init)
        # Define "classifiers" to be used
        self.classifiers = {
            "Empirical Covariance": EllipticEnvelope(support_fraction=1., contamination=0.261),
            "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(contamination=0.261),
            "OCSVM": OneClassSVM(nu=0.261, gamma=0.05)
        }
        
        # Get data
        self.TrainingData = load_boston()['data'][:, [8, 10]]
        
    
    @EPLAction("action<>")
    def Train(self):
        # Learn a frontier for outlier detection with several classifiers
        for clf in self.classifiers.values():
            clf.fit(self.TrainingData)
    
    @EPLAction("action<apamax.ml.HousingData> returns dictionary<string, boolean>")
    def CheckIfOutlier(self, d):
        asData = [d.fields["RAD"], d.fields["PTRATIO"]]
        res = {}
        for (clf_name, clf) in self.classifiers.items():
            # Test this data point against our classifier
            predictions = clf.predict([asData])
            # This is numpy.bool_ by default, since predictions[0] is a numpy.int32 or int64. Cast so EPL understands it
            entry = {clf_name : bool(predictions[0] == -1)}   # -1 means outlier, 1 means inlier
            res.update(entry)
        
        return res