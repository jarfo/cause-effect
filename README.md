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

TRAINING (Aprox. 4 hours)
- Download train, SUP1 and SUP2 data from Kaggle (cause-effect competition)
- Edit SETTINGS.json to indicate your data folders
- Train the models
    ./train.sh

FAST TEST (first 9 entries of the validation data)
python predict.py CEfinal_valid_pairs_head.csv CEfinal_valid_publicinfo_head.csv CEfinal_valid_predictions_head.csv
Time to process: 50 seconds
Result: CEfinal_valid_predictions_head.csv
SampleID,Target
valid1, 0.900748410
valid2,-0.138517826
valid3,-0.001153197
valid4, 0.031484862
valid5,-0.025577883
valid6,-1.585543675
valid7, 0.019442410
valid8,-0.000503116
valid9,-0.001260493

See the data page of the Kaggle cause-effect competition for information about the data
