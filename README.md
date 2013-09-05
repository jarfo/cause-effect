contact name: José A. R. Fonollosa
email: jose.fonollosa@upc.edu
code version: 1.0
code date: 19-AUG-2013
installation instructions: python 2.7 code. No installation required
required python modules: numpy, pandas, sklearn, scipy
Tested on a Linux machine (Fedora 17) with python 2.7.3 and the following versions of the python libraries
numpy==1.6.2
pandas==0.10.0
scikit-learn==0.13.1
scipy==0.10.1
usage instructions: python predict.py CEdata_test_pairs.csv CEdata_test_publicinfo.csv CEdata_test_predictions.csv
compute power: tested on server with 96GB RAM, 2 processors, 16 cores, but most of the code use a single core and I guess 4GB RAM is sufficient
Time to process the complete validation data: 35 minutes (most on feature extraction)
Performance on validation data: 0.81464

FAST TEST (first 9 entries of the validation data)
python predict.py CEfinal_valid_pairs_head.csv CEfinal_valid_publicinfo_head.csv CEfinal_valid_predictions_head.csv
Time to process: 50 seconds
Result: CEfinal_valid_predictions_head.csv
SampleID,Target
valid1,0.86626938658374497
valid2,-0.14576526662965034
valid3,-0.0021485422866871912
valid4,0.031906993654532693
valid5,-0.029357137397800256
valid6,-1.4842313319059746
valid7,0.019240706010512032
valid8,-0.00057276910468823594
valid9,-0.0018462447369619708

TRAINING
python cctrain.py train train1 train2
python cntrain.py train train1 train2
python nntrain.py train train1 train2
python train.py train train1 train2
