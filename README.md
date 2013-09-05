   Copyright José A. R. Fonollosa <jarfo@yahoo.com>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

code version: 1.02
code date: 05-SEP-2013
installation instructions: python 2.7 code. No installation required
required python modules: numpy, pandas, sklearn, scipy
Tested on a Linux machine (Fedora 17) with python 2.7.3 and the following versions of the python libraries
numpy==1.6.2
pandas==0.10.0
scikit-learn==0.13.1
scipy==0.10.1

TRAINING
- Download data from Kaggle (cause-effect competition)
- Edit SETTINGS.json to indicate your data folders
- Train the 4 models: ccmodel, cnmodel, nnmodel and model
python cctrain.py train train1 train2
python cntrain.py train train1 train2
python nntrain.py train train1 train2
python train.py train train1 train2

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
