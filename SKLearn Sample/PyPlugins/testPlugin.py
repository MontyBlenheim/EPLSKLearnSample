'''
Created on 4 Sep 2018

@author: ANTM
'''

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.datasets import load_boston

from apama.eplplugin import EPLAction, EPLPluginBase

class TestPlugin(EPLPluginBase):
    def __init__(self,init):
        super(TestPlugin,self).__init__(init)
        # Define "classifiers" to be used
        self.classifiers = {
            "Empirical Covariance": EllipticEnvelope(support_fraction=1.,
                                                     contamination=0.261),
            "Robust Covariance (Minimum Covariance Determinant)":
            EllipticEnvelope(contamination=0.261),
            "OCSVM": OneClassSVM(nu=0.261, gamma=0.05)}
        
        # Get data
        self.TrainingData = load_boston()['data'][:, [8, 10]]  # two clusters
        # Keep this for drawing graphs with future data
        
        self.Data = []#self.TrainingData.copy()
        self.IsOutlier = []
        
    
    @EPLAction("action<>")
    def Train(self):
        # Learn a frontier for outlier detection with several classifiers
        for clf in self.classifiers.values():
            clf.fit(self.TrainingData)
    
    @EPLAction("action<apama.test.HousingData> returns dictionary<string, boolean>")
    def CheckIfOutlier(self, d):
        asData = [d.fields["RAD"], d.fields["PTRATIO"]]
        if len(self.Data) == 0:
            self.Data = [asData]
        else:
            self.Data = np.append(self.Data, [asData], axis=0)

        res = {}
        for (clf_name, clf) in self.classifiers.items():
            predictions = clf.predict([asData])
            # This is numpy.bool by default
            entry = {clf_name : bool(predictions[0] == -1)}   # -1 means outlier, 1 means inlier
            res.update(entry)
        
        self.IsOutlier.append(res)
        
        return res
    
    @EPLAction("action<string>")
    def GraphResults(self, classifier):
        plt.close()
        
        legend1 = {}

        xx1, yy1 = np.meshgrid(np.linspace(-10, 40, 500), np.linspace(-10, 40, 500))
        clf = self.classifiers[classifier]
        plt.figure(1)
        Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
        Z1 = Z1.reshape(xx1.shape)
        legend1[classifier] = plt.contour(xx1, yy1, Z1, levels=[0], linewidths=2, colors='b')
        
        legend1_values_list = list(legend1.values())
        legend1_keys_list = list(legend1.keys())
        
        # Plot the results (= shape of the data points cloud)
        plt.title("Outlier detection on a real data set (boston housing)")
        
        outliers = []
        inliers = []
        for i, data in enumerate(self.Data):
            if self.IsOutlier[i][classifier]:
                if len(outliers) == 0:
                    outliers = np.array([data])
                else:
                    outliers = np.append(outliers, [data], axis=0)
            else:
                if len(inliers) == 0:
                    inliers = np.array([data])
                else:
                    inliers = np.append(inliers, [data], axis=0)
                
        plt.scatter(outliers[:, 0], outliers[:, 1], color='red')
        plt.scatter(inliers[:, 0], inliers[:, 1], color='black')
        
        self.getLogger().info("Setting X limits")
        plt.xlim((xx1.min(), xx1.max()))
        self.getLogger().info("Setting Y limits")
        plt.ylim((yy1.min(), yy1.max()))
        self.getLogger().info("Setting Legend")
        plt.legend([legend1_values_list[0].collections[0]],
                   [legend1_keys_list[0]],
                   loc="upper center",
                   prop=matplotlib.font_manager.FontProperties(size=12))
        self.getLogger().info("Setting y Label")
        plt.ylabel("accessibility to radial highways")
        self.getLogger().info("Setting x Label")
        plt.xlabel("pupil-teacher ratio by town")
        
        plt.show()