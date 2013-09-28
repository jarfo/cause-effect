import features as f
import numpy as np
from sklearn import pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier

gbc_params_cc = {
    'loss':'deviance',
    'learning_rate': 0.1,
    'n_estimators': 400,
    'subsample': 1.0,
    'min_samples_split': 8,
    'min_samples_leaf': 1,
    'max_depth': 6,
    'init': None,
    'random_state': 1,
    'max_features': None,
    'verbose': 0
    }

gbc_params_cn = {
    'loss':'deviance',
    'learning_rate': 0.1,
    'n_estimators': 390,
    'subsample': 1.0,
    'min_samples_split': 8,
    'min_samples_leaf': 1,
    'max_depth': 7,
    'init': None,
    'random_state': 1,
    'max_features': None,
    'verbose': 0
    }

gbc_params_nn = {
    'loss':'deviance',
    'learning_rate': 0.1,
    'n_estimators': 390,
    'subsample': 1.0,
    'min_samples_split': 8,
    'min_samples_leaf': 1,
    'max_depth': 9,
    'init': None,
    'random_state': 1,
    'max_features': None,
    'verbose': 0
    }

gbc_params_union = {
    'loss':'deviance',
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 1.0,
    'min_samples_split': 8,
    'min_samples_leaf': 1,
    'max_depth': 9,
    'init': None,
    'random_state': 1,
    'max_features': None,
    'verbose': 0
    }

selected_symmetric_categorical_features = [
    'Moment31[A,A type,B,B type]',
    'Moment31[B,B type,A,A type]',
    'Sub[Conditional Distribution Entropy Variance[A,A type,B,B type],Conditional Distribution Entropy Variance[B,B type,A,A type]]',
    'Sub[Conditional Distribution Skewness Variance[A,A type,B,B type],Conditional Distribution Skewness Variance[B,B type,A,A type]]',
    'Discrete Mutual Information[A,A type,B,B type]',
    'Log[Number of Samples[A]]',
    'Normalized Discrete Entropy[A,A type]',
    'Normalized Discrete Entropy[B,B type]',
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]]',
    'Number of Unique Samples[A]',
    'Number of Unique Samples[B]',
    'Normalized Error Probability[A,A type,B,B type]',
    'Normalized Error Probability[B,B type,A,A type]',
]

selected_onestep_categorical_features = [
    'Moment31[A,A type,B,B type]',
    'Moment31[B,B type,A,A type]',
    'Sub[Conditional Distribution Entropy Variance[A,A type,B,B type],Conditional Distribution Entropy Variance[B,B type,A,A type]]',
    'Sub[Conditional Distribution Skewness Variance[A,A type,B,B type],Conditional Distribution Skewness Variance[B,B type,A,A type]]',
    'Discrete Mutual Information[A,A type,B,B type]',
    'Log[Number of Samples[A]]',
    'Normalized Discrete Entropy[A,A type]',
    'Normalized Discrete Entropy[B,B type]',
    'Normalized Entropy Baseline[A,A type]',
    'Normalized Entropy Baseline[B,B type]',
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]]',
    'Number of Unique Samples[A]',
    'Number of Unique Samples[B]',
    'Normalized Error Probability[A,A type,B,B type]',
    'Normalized Error Probability[B,B type,A,A type]',
    ]

selected_direction_categorical_features = [
    'Conditional Distribution Entropy Variance[A,A type,B,B type]',
    'Conditional Distribution Entropy Variance[B,B type,A,A type]',
    'Conditional Distribution Kurtosis Variance[A,A type,B,B type]',
    'Conditional Distribution Kurtosis Variance[B,B type,A,A type]',
    'Conditional Distribution Skewness Variance[A,A type,B,B type]',
    'Conditional Distribution Skewness Variance[B,B type,A,A type]',
    'Moment31[A,A type,B,B type]',
    'Moment31[B,B type,A,A type]',
    'Normalized Discrete Entropy[A,A type]',
    'Normalized Discrete Entropy[B,B type]',
    'Normalized Entropy Baseline[A,A type]',
    'Normalized Entropy Baseline[B,B type]',
    'Number of Unique Samples[A]',
    'Number of Unique Samples[B]',
    'Skewness[A,A type]',
    'Skewness[B,B type]',
    ]

selected_independence_categorical_features = [
    'Abs[Sub[Abs[Moment21[A,A type,B,B type]],Abs[Moment21[B,B type,A,A type]]]]',
    'Abs[Sub[Conditional Distribution Entropy Variance[A,A type,B,B type],Conditional Distribution Entropy Variance[B,B type,A,A type]]]',
    'Abs[Sub[Conditional Distribution Kurtosis Variance[A,A type,B,B type],Conditional Distribution Kurtosis Variance[B,B type,A,A type]]]',
    'Abs[Sub[Normalized Error Probability[A,A type,B,B type],Normalized Error Probability[B,B type,A,A type]]]',
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]]',
    ]

selected_direction_numerical_features = [
    'Moment21[A,A type,B,B type]',
    'Moment21[B,B type,A,A type]',
    'Conditional Distribution Entropy Variance[A,A type,B,B type]',
    'Conditional Distribution Entropy Variance[B,B type,A,A type]',
    'Discrete Entropy[A,A type]',
    'Discrete Entropy[B,B type]',
    'Normalized Entropy[A,A type]',
    'Normalized Entropy[B,B type]',
    'Number of Unique Samples[A]',
    'Number of Unique Samples[B]',
    'Skewness[A,A type]',
    'Skewness[B,B type]',
    'Uniform Divergence[A,A type]',
    'Uniform Divergence[B,B type]',
    ]

selected_independence_numerical_features = [
    'Abs[Pearson R[A,A type,B,B type]]',
    'Abs[Sub[Abs[Moment31[A,A type,B,B type]],Abs[Moment31[B,B type,A,A type]]]]',
    'Abs[Sub[Moment31[A,A type,B,B type],Moment31[B,B type,A,A type]]]',
    'Abs[Sub[Conditional Distribution Kurtosis Variance[A,A type,B,B type],Conditional Distribution Kurtosis Variance[B,B type,A,A type]]]',
    'Abs[Sub[Normalized Discrete Entropy[A,A type],Normalized Discrete Entropy[B,B type]]]',
    'Polyfit Error[A,A type,B,B type]',
    'Polyfit Error[B,B type,A,A type]',
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Discrete Joint Entropy[A,A type,B,B type]]',
    'Normalized Entropy Baseline[B,B type]',
    'Normalized Entropy Baseline[A,A type]',
    'Normalized Entropy[A,A type]',
    'Normalized Entropy[B,B type]',
    'Number of Unique Samples[A]',
    'Number of Unique Samples[B]',
    ]
    
selected_onestep_numerical_features = [
    'Abs[Moment21[A,A type,B,B type]]',
    'Abs[Moment21[B,B type,A,A type]]',
    'Abs[Moment31[A,A type,B,B type]]', 
    'Abs[Moment31[B,B type,A,A type]]', 
    'Abs[Pearson R[A,A type,B,B type]]',
    'Adjusted Mutual Information[A,A type,B,B type]',
    'Conditional Distribution Entropy Variance[A,A type,B,B type]',
    'Conditional Distribution Entropy Variance[B,B type,A,A type]',
    'Conditional Distribution Kurtosis Variance[A,A type,B,B type]', 
    'Conditional Distribution Kurtosis Variance[B,B type,A,A type]', 
    'Conditional Distribution Similarity[A,A type,B,B type]',
    'Conditional Distribution Similarity[B,B type,A,A type]', 
    'Conditional Distribution Skewness Variance[A,A type,B,B type]', 
    'Conditional Distribution Skewness Variance[B,B type,A,A type]', 
    'Discrete Entropy[A,A type]',
    'Discrete Entropy[B,B type]',
    'Gaussian Divergence[A,A type]',
    'Gaussian Divergence[B,B type]',
    'Log[Number of Unique Samples[A]]',
    'Log[Number of Unique Samples[B]]',
    'Normalized Discrete Entropy[A,A type]',
    'Normalized Discrete Entropy[B,B type]',
    'Normalized Entropy Baseline[A,A type]',
    'Normalized Entropy Baseline[B,B type]',
    'Skewness[A,A type]',
    'Skewness[B,B type]',
    'Uniform Divergence[A,A type]',
    'Uniform Divergence[B,B type]'
    ]

selected_symmetric_numerical_features = [
    'Abs[Moment21[A,A type,B,B type]]', 
    'Abs[Moment21[B,B type,A,A type]]', 
    'Abs[Moment31[A,A type,B,B type]]', 
    'Abs[Moment31[B,B type,A,A type]]', 
    'Abs[Pearson R[A,A type,B,B type]]', 
    'Adjusted Mutual Information[A,A type,B,B type]', 
    'Conditional Distribution Entropy Variance[A,A type,B,B type]', 
    'Conditional Distribution Entropy Variance[B,B type,A,A type]', 
    'Conditional Distribution Kurtosis Variance[A,A type,B,B type]', 
    'Conditional Distribution Kurtosis Variance[B,B type,A,A type]', 
    'Conditional Distribution Similarity[A,A type,B,B type]', 
    'Conditional Distribution Similarity[B,B type,A,A type]', 
    'Conditional Distribution Skewness Variance[A,A type,B,B type]', 
    'Conditional Distribution Skewness Variance[B,B type,A,A type]', 
    'Discrete Conditional Entropy[A,A type,B,B type]', 
    'Discrete Conditional Entropy[B,B type,A,A type]', 
    'Discrete Entropy[A,A type]', 
    'Discrete Entropy[B,B type]', 
    'Gaussian Divergence[A,A type]', 
    'Gaussian Divergence[B,B type]', 
    'HSIC[A,A type,B,B type]', 
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]]', 
    'Normalized Entropy[A,A type]',
    'Normalized Entropy[B,B type]', 
    'Number of Unique Samples[A]',
    'Number of Unique Samples[B]', 
    'Polyfit[A,A type,B,B type]', 
    'Polyfit[B,B type,A,A type]', 
    'Skewness[A,A type]', 
    'Skewness[B,B type]', 
    ]

selected_direction_cn_features = [
    'Abs[Moment21[A,A type,B,B type]]',
    'Abs[Moment21[B,B type,A,A type]]',
    'Abs[Moment31[A,A type,B,B type]]',
    'Abs[Moment31[B,B type,A,A type]]',
    'Conditional Distribution Entropy Variance[A,A type,B,B type]', 
    'Conditional Distribution Entropy Variance[B,B type,A,A type]', 
    'Conditional Distribution Similarity[A,A type,B,B type]',
    'Conditional Distribution Similarity[B,B type,A,A type]', 
    'Conditional Distribution Skewness Variance[A,A type,B,B type]', 
    'Conditional Distribution Skewness Variance[B,B type,A,A type]', 
    'Discrete Entropy[A,A type]',
    'Discrete Entropy[B,B type]', 
    'IGCI[A,A type,B,B type]', 
    'IGCI[B,B type,A,A type]', 
    'Normalized Discrete Entropy[A,A type]',
    'Normalized Discrete Entropy[B,B type]', 
    'Normalized Entropy Baseline[A,A type]', 
    'Normalized Entropy Baseline[B,B type]', 
    'Normalized Entropy[A,A type]',
    'Normalized Entropy[B,B type]', 
    'Number of Unique Samples[A]',
    'Number of Unique Samples[B]', 
     ]

selected_independence_cn_features = [
    'Abs[Pearson R[A,A type,B,B type]]',
    'Abs[Sub[Abs[Moment21[A,A type,B,B type]],Abs[Moment21[B,B type,A,A type]]]]', 
    'Abs[Sub[Abs[Moment31[A,A type,B,B type]],Abs[Moment31[B,B type,A,A type]]]]', 
    'Abs[Sub[Conditional Distribution Entropy Variance[A,A type,B,B type],Conditional Distribution Entropy Variance[B,B type,A,A type]]]',
    'Abs[Sub[Conditional Distribution Similarity[A,A type,B,B type],Conditional Distribution Similarity[B,B type,A,A type]]]',
    'Abs[Sub[Conditional Distribution Skewness Variance[A,A type,B,B type],Conditional Distribution Skewness Variance[B,B type,A,A type]]]',
    'Abs[Sub[Normalized Discrete Entropy[A,A type],Normalized Discrete Entropy[B,B type]]]', 
    'Abs[Sub[Skewness[A,A type],Skewness[B,B type]]]', 
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Discrete Joint Entropy[A,A type,B,B type]]', 
    'Number of Unique Samples[A]',
    'Number of Unique Samples[B]', 
    ]

selected_onestep_cn_features = [
    'Abs[Moment21[A,A type,B,B type]]',
    'Abs[Moment21[B,B type,A,A type]]',
    'Abs[Moment31[A,A type,B,B type]]',
    'Abs[Moment31[B,B type,A,A type]]',
    'Conditional Distribution Entropy Variance[A,A type,B,B type]', 
    'Conditional Distribution Entropy Variance[B,B type,A,A type]', 
    'Conditional Distribution Similarity[A,A type,B,B type]',
    'Conditional Distribution Similarity[B,B type,A,A type]', 
    'Conditional Distribution Skewness Variance[A,A type,B,B type]', 
    'Conditional Distribution Skewness Variance[B,B type,A,A type]', 
    'Discrete Entropy[A,A type]', 
    'Discrete Entropy[B,B type]', 
    'IGCI[A,A type,B,B type]', 
    'IGCI[B,B type,A,A type]', 
    'Kurtosis[A,A type]', 
    'Kurtosis[B,B type]', 
    'Log[Number of Unique Samples[A]]',
    'Normalized Discrete Entropy[A,A type]', 
    'Normalized Discrete Entropy[B,B type]', 
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Discrete Joint Entropy[A,A type,B,B type]]', 
    'Normalized Entropy Baseline[A,A type]', 
    'Normalized Entropy Baseline[B,B type]', 
    'Normalized Entropy[A,A type]', 
    'Normalized Entropy[B,B type]', 
    'Normalized Error Probability[A,A type,B,B type]',
    'Normalized Error Probability[B,B type,A,A type]',
    'Skewness[A,A type]', 
    'Skewness[B,B type]', 
    ]

selected_symmetric_cn_features = [
    'Conditional Distribution Entropy Variance[A,A type,B,B type]',
    'Conditional Distribution Entropy Variance[B,B type,A,A type]',
    'Conditional Distribution Kurtosis Variance[A,A type,B,B type]', 
    'Conditional Distribution Kurtosis Variance[B,B type,A,A type]', 
    'Conditional Distribution Skewness Variance[A,A type,B,B type]', 
    'Conditional Distribution Skewness Variance[B,B type,A,A type]', 
    'Discrete Conditional Entropy[A,A type,B,B type]', 
    'Discrete Entropy[A,A type]',
    'Discrete Entropy[B,B type]', 
    'IGCI[A,A type,B,B type]', 
    'IGCI[B,B type,A,A type]', 
    'Kurtosis[A,A type]', 
    'Kurtosis[B,B type]', 
    'Log[Number of Unique Samples[A]]', 
    'Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]', 
    'Min[Normalized Entropy Baseline[A,A type],Normalized Entropy Baseline[B,B type]]', 
    'Min[Normalized Entropy[A,A type],Normalized Entropy[B,B type]]', 
    'Normalized Discrete Entropy[A,A type]', 
    'Normalized Discrete Entropy[B,B type]', 
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Discrete Joint Entropy[A,A type,B,B type]]', 
    'Normalized Entropy Baseline[A,A type]',
    'Normalized Entropy[B,B type]', 
]

features_all = [
#    'Max[A]',
#    'Max[B]',
#    'Min[A]',
#    'Min[B]',
    'Numerical[A type]',
    'Numerical[B type]',
    'Sub[Numerical[A type],Numerical[B type]]',
    'Abs[Sub[Numerical[A type],Numerical[B type]]]',
    
    'Number of Samples[A]',
    'Log[Number of Samples[A]]',
    
    'Number of Unique Samples[A]',
    'Number of Unique Samples[B]',
    'Max[Number of Unique Samples[A],Number of Unique Samples[B]]',
    'Min[Number of Unique Samples[A],Number of Unique Samples[B]]',
    'Sub[Number of Unique Samples[A],Number of Unique Samples[B]]',
    'Abs[Sub[Number of Unique Samples[A],Number of Unique Samples[B]]]',
    
    'Log[Number of Unique Samples[A]]',
    'Log[Number of Unique Samples[B]]',
    'Max[Log[Number of Unique Samples[A]],Log[Number of Unique Samples[B]]]',
    'Min[Log[Number of Unique Samples[A]],Log[Number of Unique Samples[B]]]',
    'Sub[Log[Number of Unique Samples[A]],Log[Number of Unique Samples[B]]]',
    'Abs[Sub[Log[Number of Unique Samples[A]],Log[Number of Unique Samples[B]]]]',
    
    'Ratio of Unique Samples[A]',
    'Ratio of Unique Samples[B]',
    'Max[Ratio of Unique Samples[A],Ratio of Unique Samples[B]]',
    'Min[Ratio of Unique Samples[A],Ratio of Unique Samples[B]]',
    'Sub[Ratio of Unique Samples[A],Ratio of Unique Samples[B]]',
    'Abs[Sub[Ratio of Unique Samples[A],Ratio of Unique Samples[B]]]',
    
    'Normalized Entropy Baseline[A,A type]',
    'Normalized Entropy Baseline[B,B type]',
    'Max[Normalized Entropy Baseline[A,A type],Normalized Entropy Baseline[B,B type]]',
    'Min[Normalized Entropy Baseline[A,A type],Normalized Entropy Baseline[B,B type]]',
    'Sub[Normalized Entropy Baseline[A,A type],Normalized Entropy Baseline[B,B type]]',
    'Abs[Sub[Normalized Entropy Baseline[A,A type],Normalized Entropy Baseline[B,B type]]]',
    
    'Normalized Entropy[A,A type]',
    'Normalized Entropy[B,B type]',
    'Max[Normalized Entropy[A,A type],Normalized Entropy[B,B type]]',
    'Min[Normalized Entropy[A,A type],Normalized Entropy[B,B type]]',
    'Sub[Normalized Entropy[A,A type],Normalized Entropy[B,B type]]',
    'Abs[Sub[Normalized Entropy[A,A type],Normalized Entropy[B,B type]]]',
    
    'IGCI[A,A type,B,B type]',
    'IGCI[B,B type,A,A type]',
    'Sub[IGCI[A,A type,B,B type],IGCI[B,B type,A,A type]]',
    'Abs[Sub[IGCI[A,A type,B,B type],IGCI[B,B type,A,A type]]]',

    'Gaussian Divergence[A,A type]',
    'Gaussian Divergence[B,B type]',
    'Max[Gaussian Divergence[A,A type],Gaussian Divergence[B,B type]]',
    'Min[Gaussian Divergence[A,A type],Gaussian Divergence[B,B type]]',
    'Sub[Gaussian Divergence[A,A type],Gaussian Divergence[B,B type]]',
    'Abs[Sub[Gaussian Divergence[A,A type],Gaussian Divergence[B,B type]]]',
    
    'Uniform Divergence[A,A type]',
    'Uniform Divergence[B,B type]',
    'Max[Uniform Divergence[A,A type],Uniform Divergence[B,B type]]',
    'Min[Uniform Divergence[A,A type],Uniform Divergence[B,B type]]',
    'Sub[Uniform Divergence[A,A type],Uniform Divergence[B,B type]]',
    'Abs[Sub[Uniform Divergence[A,A type],Uniform Divergence[B,B type]]]',
    
    'Discrete Entropy[A,A type]',
    'Discrete Entropy[B,B type]',
    'Max[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]',
    'Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]',
    'Sub[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]',
    'Abs[Sub[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]]',
    
    'Normalized Discrete Entropy[A,A type]',
    'Normalized Discrete Entropy[B,B type]',
    'Max[Normalized Discrete Entropy[A,A type],Normalized Discrete Entropy[B,B type]]',
    'Min[Normalized Discrete Entropy[A,A type],Normalized Discrete Entropy[B,B type]]',
    'Sub[Normalized Discrete Entropy[A,A type],Normalized Discrete Entropy[B,B type]]',
    'Abs[Sub[Normalized Discrete Entropy[A,A type],Normalized Discrete Entropy[B,B type]]]',
    
    'Discrete Joint Entropy[A,A type,B,B type]',
    'Normalized Discrete Joint Entropy[A,A type,B,B type]',
    'Discrete Conditional Entropy[A,A type,B,B type]',
    'Discrete Conditional Entropy[B,B type,A,A type]',
    'Discrete Mutual Information[A,A type,B,B type]',
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]]',
    'Normalized Discrete Mutual Information[Discrete Mutual Information[A,A type,B,B type],Discrete Joint Entropy[A,A type,B,B type]]',
    'Adjusted Mutual Information[A,A type,B,B type]',
    
    'Polyfit[A,A type,B,B type]',
    'Polyfit[B,B type,A,A type]',
    'Sub[Polyfit[A,A type,B,B type],Polyfit[B,B type,A,A type]]',
    'Abs[Sub[Polyfit[A,A type,B,B type],Polyfit[B,B type,A,A type]]]',

    'Polyfit Error[A,A type,B,B type]',
    'Polyfit Error[B,B type,A,A type]',
    'Sub[Polyfit Error[A,A type,B,B type],Polyfit Error[B,B type,A,A type]]',
    'Abs[Sub[Polyfit Error[A,A type,B,B type],Polyfit Error[B,B type,A,A type]]]',

    'Normalized Error Probability[A,A type,B,B type]',
    'Normalized Error Probability[B,B type,A,A type]',
    'Sub[Normalized Error Probability[A,A type,B,B type],Normalized Error Probability[B,B type,A,A type]]',
    'Abs[Sub[Normalized Error Probability[A,A type,B,B type],Normalized Error Probability[B,B type,A,A type]]]',

    'Conditional Distribution Entropy Variance[A,A type,B,B type]',
    'Conditional Distribution Entropy Variance[B,B type,A,A type]',
    'Sub[Conditional Distribution Entropy Variance[A,A type,B,B type],Conditional Distribution Entropy Variance[B,B type,A,A type]]',
    'Abs[Sub[Conditional Distribution Entropy Variance[A,A type,B,B type],Conditional Distribution Entropy Variance[B,B type,A,A type]]]',

    'Conditional Distribution Skewness Variance[A,A type,B,B type]',
    'Conditional Distribution Skewness Variance[B,B type,A,A type]',
    'Sub[Conditional Distribution Skewness Variance[A,A type,B,B type],Conditional Distribution Skewness Variance[B,B type,A,A type]]',
    'Abs[Sub[Conditional Distribution Skewness Variance[A,A type,B,B type],Conditional Distribution Skewness Variance[B,B type,A,A type]]]',

    'Conditional Distribution Kurtosis Variance[A,A type,B,B type]',
    'Conditional Distribution Kurtosis Variance[B,B type,A,A type]',
    'Sub[Conditional Distribution Kurtosis Variance[A,A type,B,B type],Conditional Distribution Kurtosis Variance[B,B type,A,A type]]',
    'Abs[Sub[Conditional Distribution Kurtosis Variance[A,A type,B,B type],Conditional Distribution Kurtosis Variance[B,B type,A,A type]]]',

    'Conditional Distribution Similarity[A,A type,B,B type]',
    'Conditional Distribution Similarity[B,B type,A,A type]',
    'Sub[Conditional Distribution Similarity[A,A type,B,B type],Conditional Distribution Similarity[B,B type,A,A type]]',
    'Abs[Sub[Conditional Distribution Similarity[A,A type,B,B type],Conditional Distribution Similarity[B,B type,A,A type]]]',

    'Moment21[A,A type,B,B type]',
    'Moment21[B,B type,A,A type]',
    'Sub[Moment21[A,A type,B,B type],Moment21[B,B type,A,A type]]',
    'Abs[Sub[Moment21[A,A type,B,B type],Moment21[B,B type,A,A type]]]',
    
    'Abs[Moment21[A,A type,B,B type]]',
    'Abs[Moment21[B,B type,A,A type]]',
    'Sub[Abs[Moment21[A,A type,B,B type]],Abs[Moment21[B,B type,A,A type]]]',
    'Abs[Sub[Abs[Moment21[A,A type,B,B type]],Abs[Moment21[B,B type,A,A type]]]]',

    'Moment31[A,A type,B,B type]',
    'Moment31[B,B type,A,A type]',
    'Sub[Moment31[A,A type,B,B type],Moment31[B,B type,A,A type]]',
    'Abs[Sub[Moment31[A,A type,B,B type],Moment31[B,B type,A,A type]]]',
            
    'Abs[Moment31[A,A type,B,B type]]',
    'Abs[Moment31[B,B type,A,A type]]',
    'Sub[Abs[Moment31[A,A type,B,B type]],Abs[Moment31[B,B type,A,A type]]]',
    'Abs[Sub[Abs[Moment31[A,A type,B,B type]],Abs[Moment31[B,B type,A,A type]]]]',

    'Skewness[A,A type]',
    'Skewness[B,B type]',
    'Sub[Skewness[A,A type],Skewness[B,B type]]',
    'Abs[Sub[Skewness[A,A type],Skewness[B,B type]]]',        
    
    'Abs[Skewness[A,A type]]',
    'Abs[Skewness[B,B type]]',
    'Max[Abs[Skewness[A,A type]],Abs[Skewness[B,B type]]]',
    'Min[Abs[Skewness[A,A type]],Abs[Skewness[B,B type]]]',
    'Sub[Abs[Skewness[A,A type]],Abs[Skewness[B,B type]]]',
    'Abs[Sub[Abs[Skewness[A,A type]],Abs[Skewness[B,B type]]]]',
    
    'Kurtosis[A,A type]',
    'Kurtosis[B,B type]',
    'Max[Kurtosis[A,A type],Kurtosis[B,B type]]',
    'Min[Kurtosis[A,A type],Kurtosis[B,B type]]',
    'Sub[Kurtosis[A,A type],Kurtosis[B,B type]]',
    'Abs[Sub[Kurtosis[A,A type],Kurtosis[B,B type]]]',

    'HSIC[A,A type,B,B type]',
    'Pearson R[A,A type,B,B type]',
    'Abs[Pearson R[A,A type,B,B type]]'
    ]

class Pipeline(pipeline.Pipeline):
    def predict(self, X):
        try:
            p = pipeline.Pipeline.predict_proba(self, X)
            if p.shape[1] == 2:
                p = p[:,1]
            elif p.shape[1] == 3:
                p = p[:,2] - p[:,0]
        except AttributeError:
            p = pipeline.Pipeline.predict(self, X)
        return p


def get_pipeline(features, regressor=None, params=None):
    steps = [
        ("extract_features", f.FeatureMapper(features)),
        ("regressor", regressor(**params)),
        ]
    return Pipeline(steps)

class CauseEffectEstimatorOneStep(BaseEstimator):
    def __init__(self, features=None, regressor=None, params=None, symmetrize=True):
        self.extractor = f.extract_features
        self.classifier = get_pipeline(features, regressor, params)
        self.symmetrize = symmetrize
    
    def extract(self, features):
        return self.extractor(features)

    def fit(self, X, y=None):
        self.classifier.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self.classifier.fit_transform(X, y)

    def transform(self, X):
        return self.classifier.transform(X)

    def predict(self, X):
        predictions = self.classifier.predict(X)
        if self.symmetrize:
            predictions[0::2] = (predictions[0::2] - predictions[1::2])/2
            predictions[1::2] = -predictions[0::2]
        return predictions

class CauseEffectEstimatorSymmetric(BaseEstimator):
    def __init__(self, features=None, regressor=None, params=None, symmetrize=True):
        self.extractor = f.extract_features
        self.classifier_left = get_pipeline(features, regressor, params)
        self.classifier_right = get_pipeline(features, regressor, params)
        self.symmetrize = symmetrize
    
    def extract(self, features):
        return self.extractor(features)

    def fit(self, X, y=None):
        target_left = np.array(y)
        target_left[target_left != 1] = 0
        weight_left = np.ones(len(target_left))
        weight_left[target_left==0] = sum(target_left==1)/float(sum(target_left==0))    
        try:
            self.classifier_left.fit(X, target_left, regressor__sample_weight=weight_left)
        except TypeError:
            self.classifier_left.fit(X, target_left)
        target_right = np.array(y)
        target_right[target_right != -1] = 0
        target_right[target_right == -1] = 1
        weight_right = np.ones(len(target_right))
        weight_right[target_right==0] = sum(target_right==1)/float(sum(target_right==0))        
        try:
            self.classifier_right.fit(X, target_right, regressor__sample_weight=weight_right)
        except TypeError:
            self.classifier_right.fit(X, target_right)
       
        return self

    def fit_transform(self, X, y=None):
        target_left = np.array(y)
        target_left[target_left != 1] = 0
        X_left = self.classifier_left.fit_transform(X, target_left)
        target_right = np.array(y)
        target_right[target_right != -1] = 0
        target_right[target_right == -1] = 1
        X_right = self.classifier_right.fit_transform(X, target_right)
        return X_left, X_right

    def transform(self, X):
        return self.classifier_left.transform(X), self.classifier_right.transform(X)

    def predict(self, X):
        predictions_left = self.classifier_left.predict(X)
        predictions_right = self.classifier_right.predict(X)
        predictions = predictions_left - predictions_right
        if self.symmetrize:
            predictions[0::2] = (predictions[0::2] - predictions[1::2])/2
            predictions[1::2] = -predictions[0::2]
        return predictions

class CauseEffectEstimatorID(BaseEstimator):
    def __init__(self, features_independence=None, features_direction=None, regressor=None, params=None, symmetrize=True):
        self.extractor = f.extract_features
        self.classifier_independence = get_pipeline(features_independence, regressor, params)
        self.classifier_direction = get_pipeline(features_direction, regressor, params)
        self.symmetrize = symmetrize
    
    def extract(self, features):
        return self.extractor(features)

    def fit(self, X, y=None):
        #independence training pairs
        train_independence = X
        target_independence = np.array(y)
        target_independence[target_independence != 0] = 1
        weight_independence = np.ones(len(target_independence))
        weight_independence[target_independence==0] = sum(target_independence==1)/float(sum(target_independence==0))        
        try:
            self.classifier_independence.fit(train_independence, target_independence, regressor__sample_weight=weight_independence)
        except TypeError:
            self.classifier_independence.fit(train_independence, target_independence)
        #direction training pairs
        direction_filter = y != 0
        train_direction = X[direction_filter]
        target_direction = y[direction_filter]
        weight_direction = np.ones(len(target_direction))
        weight_direction[target_direction==0] = sum(target_direction==1)/float(sum(target_direction==0))        
        try:
            self.classifier_direction.fit(train_direction, target_direction, regressor__sample_weight=weight_direction)
        except TypeError:
            self.classifier_direction.fit(train_direction, target_direction)
        return self

    def fit_transform(self, X, y=None):
        #independence training pairs
        train_independence = X
        target_independence = np.array(y)
        target_independence[target_independence != 0] = 1
        X_ind = self.classifier_independence.fit_transform(train_independence, target_independence)
        #direction training pairs
        direction_filter = y != 0
        train_direction = X[direction_filter]
        target_direction = y[direction_filter]
        self.classifier_direction.fit(train_direction, target_direction)
        X_dir = self.classifier_direction.transform(X)
        return X_ind, X_dir

    def transform(self, X):
        X_ind = self.classifier_independence.transform(X)
        X_dir = self.classifier_direction.transform(X)
        return X_ind, X_dir

    def predict(self, X):
        predictions_independence = self.classifier_independence.predict(X)
        if self.symmetrize:
            predictions_independence[0::2] = (predictions_independence[0::2] + predictions_independence[1::2])/2
            predictions_independence[1::2] = predictions_independence[0::2]
        assert predictions_independence.min() >= 0
        predictions_direction = self.classifier_direction.predict(X)
        if self.symmetrize:
            predictions_direction[0::2] = (predictions_direction[0::2] - predictions_direction[1::2])/2
            predictions_direction[1::2] = -predictions_direction[0::2]
        return predictions_independence * predictions_direction

class CauseEffectSystemCombination(BaseEstimator):  
    def extract(self, features):
        return self.extractor(features)

    def fit(self, X, y=None):
        for m in self.systems:
            m.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return [m.fit_transform(X, y) for m in self.systems]

    def transform(self, X):
        return [m.transform(X) for m in self.systems]

    def predict(self, X):
        a = np.array([m.predict(X) for m in self.systems])
        if self.weights is not None:
            return np.dot(self.weights, a)
        else:
            return a

class CauseEffectSystemCombinationCC(CauseEffectSystemCombination):
    def __init__(self, extractor=f.extract_features, weights=None, symmetrize=True):
        self.extractor = extractor
        self.systems = [
            CauseEffectEstimatorID(
                features_direction=selected_direction_categorical_features, 
                features_independence=selected_independence_categorical_features,
                regressor=GradientBoostingClassifier,
                params=gbc_params_cc,
                symmetrize=symmetrize), 
            CauseEffectEstimatorSymmetric(
                features=selected_symmetric_categorical_features, 
                regressor=GradientBoostingClassifier,
                params=gbc_params_cc,
                symmetrize=symmetrize),
            CauseEffectEstimatorOneStep(
                features=selected_onestep_categorical_features,
                regressor=GradientBoostingClassifier,
                params=gbc_params_cc,
                symmetrize=symmetrize),
        ]
        self.weights = weights

class CauseEffectSystemCombinationCN(CauseEffectSystemCombination):
    def __init__(self, extractor=f.extract_features, weights=None, symmetrize=True):
        self.extractor = extractor
        self.systems = [
            CauseEffectEstimatorID(
                features_direction=selected_direction_cn_features, 
                features_independence=selected_independence_cn_features,
                regressor=GradientBoostingClassifier,
                params=gbc_params_cn,
                symmetrize=symmetrize), 
            CauseEffectEstimatorSymmetric(
                features=selected_symmetric_cn_features, 
                regressor=GradientBoostingClassifier,
                params=gbc_params_cn,
                symmetrize=symmetrize),
            CauseEffectEstimatorOneStep(
                features=selected_onestep_cn_features,
                regressor=GradientBoostingClassifier,
                params=gbc_params_cn,
                symmetrize=symmetrize),
        ]
        self.weights = weights

class CauseEffectSystemCombinationNN(CauseEffectSystemCombination):
    def __init__(self, extractor=f.extract_features, weights=None, symmetrize=True):
        self.extractor = extractor
        self.systems = [
            CauseEffectEstimatorID(
                features_direction=selected_direction_numerical_features, 
                features_independence=selected_independence_numerical_features,
                regressor=GradientBoostingClassifier,
                params=gbc_params_nn,
                symmetrize=symmetrize), 
            CauseEffectEstimatorSymmetric(
                features=selected_symmetric_numerical_features, 
                regressor=GradientBoostingClassifier,
                params=gbc_params_nn,
                symmetrize=symmetrize),
            CauseEffectEstimatorOneStep(
                features=selected_onestep_numerical_features,
                regressor=GradientBoostingClassifier,
                params=gbc_params_nn,
                symmetrize=symmetrize),
        ]
        self.weights = weights

class CauseEffectSystemCombinationUnion(CauseEffectSystemCombination):
    def __init__(self, extractor=f.extract_features, weights=None, symmetrize=True):
        self.extractor = extractor
        self.systems = [
            CauseEffectEstimatorID(
                features_direction=sorted(list(set(selected_direction_categorical_features + selected_direction_cn_features + selected_direction_numerical_features))), 
                features_independence=sorted(list(set(selected_independence_categorical_features + selected_independence_cn_features + selected_independence_numerical_features))),
                regressor=GradientBoostingClassifier,
                params=gbc_params_union,
                symmetrize=symmetrize), 
            CauseEffectEstimatorSymmetric(
                features=sorted(list(set(selected_symmetric_categorical_features + selected_symmetric_cn_features + selected_symmetric_numerical_features))),
                regressor=GradientBoostingClassifier,
                params=gbc_params_union,
                symmetrize=symmetrize),
            CauseEffectEstimatorOneStep(
                features=sorted(list(set(selected_onestep_categorical_features + selected_onestep_cn_features + selected_onestep_numerical_features))),
                regressor=GradientBoostingClassifier,
                params=gbc_params_union,
                symmetrize=symmetrize),
        ]
        self.weights = weights
