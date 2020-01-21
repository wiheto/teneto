# ?from sklearn.base import BaseEstimator, ClusterMixin
# from sklearn.metrics import make_scorer
# from teneto.communitydetection import tctc 
# import numpy as np 



# def scorer_overall_difference(X, y): 
#     """
#     """
#     vals = np.unique(y)
#     X = np.array(X)
#     for v in vals:
#         v1 = X[y == int(vals[0])]
#         v2 = X[y == int(vals[1])]
#     return -np.abs(np.mean(v1) - np.mean(v2))

# #def scorer_timelocked_difference(X, y): 



# class TCTC_Estimator(BaseEstimator):  
#     """
#     """
#     def __init__(self, epsilon=None, tau=None, sigma=None, kappa=None, scoring='overalldifference'):
#         self.epsilon = epsilon
#         self.tau = tau
#         self.sigma = sigma
#         self.kappa = kappa
#         self.scoring = scoring

#     def fit(self, X, y=None):
#         """
#         This should fit classifier. All the "work" should be done here.

#         Note: assert is not a good choice here and you should rather
#         use try/except blog with exceptions. This is just for short syntax.
#         """
#         X = np.array(X, ndmin=3)
#         self.X_ = X
#         self.y_ = y
#         print(y)
#         coms = []
#         print(X.shape)
#         for n in np.arange(X.shape[0]):
#             coms.append(tctc(X[n,:,:], epsilon=self.epsilon, tau=self.tau, sigma=self.sigma, kappa=self.kappa))
#         self.tctc = np.array(coms)

#         return self

#     def score(self, X, y):
#         coms = []
#         for n in np.arange(X.shape[0]):
#             coms.append(tctc(X[n,:,:], epsilon=self.epsilon, tau=self.tau, sigma=self.sigma, kappa=self.kappa))

#         if self.scoring == 'overalldifference':
#             score = scorer_overall_difference(coms, y)
#         return score


#     # def _meaning(self, x):
#     #     # returns True/False according to fitted classifier
#     #     # notice underscore on the beginning
#     #     return( True if x >= self.treshold_ else False )







# # from sklearn.model_selection import GridSearchCV
# # #sample rate 
# # Fs = 100
# # #frequency
# # f = 1
# # #number of samples
# # sample = 200
# # ts = 50
# # x = np.arange(sample)
# # y1 = np.sin(2 * np.pi * f * x / Fs)
# # y2 = np.sin(2 * np.pi * f*2 * x / Fs)

# # # Simulate data
# # np.random.seed(2018)
# # s1 = np.array(y1) + np.random.normal(0,0.2,(ts,len(y1)))
# # s2 = (np.array(y2) * -1) + np.random.normal(0,0.2, (ts, len(y1)))
# # s3a = np.array(y1) + np.random.normal(0,0.2,(int(ts/2),len(y1)))
# # s3b = (np.array(y2) * -1) + np.random.normal(0,0.2, (int(ts/2), len(y1)))
# # s3 = np.vstack([s3a, s3b])
# # x = np.stack([s1,s2,s3]).transpose([1,2,0])
# # y = np.zeros(ts)
# # y[:int(ts/2)] = 1

# # parameters = {
# #     'epsilon': np.arange(0.1,1,0.1), 
# #     'tau': np.arange(2,11),
# #     'sigma': np.arange(2,4), 
# #     'kappa': np.arange(0,3)
# # } 

# # from sklearn.model_selection import StratifiedKFold

# # # Reason for adding StratifiedKFold is because cv seems to do a non-shuffled split otherwise
# # gs = GridSearchCV(TCTC_Estimator(), parameters, iid=False, cv=StratifiedKFold())

# # gs.fit(x,y)


