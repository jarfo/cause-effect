import sys
import data_io
import numpy as np
import estimator as ce
import features as f
import pandas as pd
import cPickle as pickle
import util

MODEL = ce.CauseEffectSystemCombinationCN
MODEL_PARAMS = {
    'weights':[0.16658564, 0.21129943, 0.15732704, 0.15749277, 0.22415686, 0.08313826],
    'symmetrize':False
    }

def main():
    
    set1 = 'train' if len(sys.argv) < 2 else sys.argv[1]
    set2 = [] if len(sys.argv) < 3 else sys.argv[2:]
    train_filter = None
    train_filter2 = None
    
    model = MODEL(**MODEL_PARAMS)
    
    print("Reading in training data " + set1)
    train = data_io.read_data(set1)
    print("Extracting features")
    train = model.extract(train)
    print("Saving train features")
    data_io.write_data(set1, train)
    target = data_io.read_target(set1)
    
    train2 = None
    target2 = None
    for s in set2:
        print "Reading in training data", s
        tr = data_io.read_data(s)
        print "Extracting features"
        tr = model.extract(tr)
        print "Saving train features"
        data_io.write_data(s, tr)
        tg = data_io.read_target(s)
        train2 = tr if train2 is None else pd.concat((train2, tr), ignore_index=True)
        target2 = tg if target2 is None else pd.concat((target2, tg), ignore_index=True)
        train2, target2 = util.random_permutation(train2, target2)
        train_filter2  = ((train2['A type'] != 'Numerical') & (train2['B type'] == 'Numerical'))
        #train_filter2 |= ((train2['A type'] == 'Numerical') & (train2['B type'] != 'Numerical'))

    # Data selection
    train, target = util.random_permutation(train, target)
    train_filter  = ((train['A type'] != 'Numerical') & (train['B type'] == 'Numerical')) 
    #train_filter |= ((train['A type'] == 'Numerical') & (train['B type'] != 'Numerical'))

    if train_filter is not None:
        train = train[train_filter]
        target = target[train_filter]
    if train_filter2 is not None:
        train2 = train2[train_filter2]
        target2 = target2[train_filter2]

    print("Training model with optimal weights")
    X = pd.concat([train, train2]) if train2 is not None else train
    y = np.concatenate((target.Target.values, target2.Target.values)) if target2 is not None else target.Target.values  
    model.fit(X, y) 
    model_path = "cnmodel.pkl"
    print "Saving model", model_path
    data_io.save_model(model, model_path)

if __name__=="__main__":
    main()
