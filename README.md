## Optimizing LoGoFunc
### Comparative analysis of classifiers and feature selection techniques.

This work expands upon the oringal LoGoFunc tool, which is [described here](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-023-01261-9) and is available to [download here.](https://gitlab.com/itan-lab/logofunc)

The full dissertation [can be found here. (PDF)](T_Nichols_Dissertation.pdf)

**Note:** Before running any of the workspaces contained in this repo, you first need to copy the contents of the `Datasets` folder into the respective data folder within the workspace. These are moderately large datasets so this was done to save space.

### Conda environments

1. I used two conda environments for this project. `logofunc3` has the versions of Scikit-learn and other packages that are specified in the LoGoFunc documentation. This environment should be used for all of the training and testing procedures.
2. The other environment is `tabulate` which I used for all of the data analysis and graphing operations. 
3. The table that describes all of the notebooks contains a column that specifies which environment should be used to run those blocks of code.
4. I exported the packages I explicitly installed for each environment using the `conda env export --from-history > *.yml` command, so it should install the required packages regardless if platform when you run `conda env create -f {env name}.yml`. The YAML files are in the `Conda env` directory.
5. If for some reason this does not work these are the packages required by LoGoFunc:


```shell
1. conda create -n logofunc python=3 pandas=1.5.0 joblib lightgbm=3.2.1 scikit-learn=1.1.2 imbalanced-learn=0.8.0
2. conda activate logofunc
```

### Directory structure

|WORKSPACE  |FOLDER     |NOTEBOOK        |DESCRIPTION                                   |ENVIRONMENT|
|-----------|-----------|----------------|----------------------------------------------|-----------|
|Classifiers|scripts    |graphing        |Graphs the classifier metrics.                |tabulate   |
|Classifiers|scripts    |train           |Train the LGBM and LGBM-P classifiers.        |logofunc3  |
|Classifiers|scripts    |encode          |Encodes feature names for XGBoost.            |tabulate   |
|Classifiers|classifiers|LinearSVM       |Assess the LinearSVM classifier.              |logofunc3  |
|Classifiers|classifiers|SGDClassifier   |Assess the SGD classifier.                    |logofunc3  |
|Classifiers|classifiers|AdaBoost        |Assess the AdaBoost classifier.               |logofunc3  |
|Classifiers|classifiers|CatBoost        |Assess the CatBoost classifier.               |logofunc3  |
|Classifiers|classifiers|ExtraTrees      |Assess the ExtraTrees classifier.             |logofunc3  |
|Classifiers|classifiers|XGBoost         |Assess the XGBoost classifier.                |logofunc3  |
|Classifiers|classifiers|PyTorch         |Assess the Pytorch MLP classifier.            |logofunc3  |
|Classifiers|classifiers|Stacking        |Assess the Stacking classifier.               |logofunc3  |
|Classifiers|classifiers|LGBM-P test     |Assess the LGBM-P classifier.                 |logofunc3  |
|Classifiers|classifiers|LGBM test       |Assess the LGBM classifier.                   |logofunc3  |
|LOFO Ranker|scripts    |graphing        |Graphs the Top 25 feature accuracies.         |tabulate   |
|LOFO Ranker|scripts    |numbering       |Creates a numbered JSON of features.          |tabulate   |
|LOFO Ranker|scripts    |train           |Train all 499 LOFO models.                    |logofunc3  |
|LOFO Ranker|scripts    |test            |Test all 499 LOFO models.                     |logofunc3  |
|LOFO Ranker|scripts    |encode_all      |Encodes feature names for XGBoost.            |tabulate   |
|LOFO Ranker|scripts    |generate_subsets|Generates the LOFO subsets.                   |tabulate   |
|LOFO Ranker|scripts    |decode          |Decodes the ranked feature names.             |tabulate   |
|LOFO Ranker|scripts    |ranked_subsets  |Creates subsets based on LOFO rankings.       |tabulate   |
|ENS Ranker |scripts    |generate_subsets|Creates subsets based on ENS rankings.        |tabulate   |
|ENS Ranker |scripts    |ensemble_ranking|Runs the ensemble ranking algorithms.         |tabulate   |
|ENS Ranker |scripts    |decoding        |Decodes the features after ranking.           |tabulate   |
|ENS Ranker |scripts    |variance        |Plots the mean rankings against the variances.|tabulate   |
|LOFO-LGBM  |LOFO-LGBM  |roc curves      |Compute AP, find best model, plot ROC curves. |tabulate   |
|LOFO-LGBM  |LOFO-LGBM  |train           |Train LGBM ensemble with LOFO ranked subsets. |logofunc3  |
|LOFO-LGBM  |LOFO-LGBM  |test            |Test all 499 LGBM models.                     |logofunc3  |
|LOFO-LGBM  |LOFO-LGBM  |graphs          |Plots both overall and class-wise metrics.    |tabulate   |
|LOFO-LGBM  |LOFO-LGBM  |sort-metrics    |Merges metrics collected by test script.      |tabulate   |
|LOFO-XGB   |LOFO-XGB   |roc curves      |Compute AP, find best model, plot ROC curves. |tabulate   |
|LOFO-XGB   |LOFO-XGB   |train           |Train XGB ensemble with LOFO ranked subsets.  |logofunc3  |
|LOFO-XGB   |LOFO-XGB   |test            |Test all 499 XGB models.                      |logofunc3  |
|LOFO-XGB   |LOFO-XGB   |graphs          |Plots both overall and class-wise metrics.    |tabulate   |
|LOFO-XGB   |LOFO-XGB   |sort-metrics    |Merges metrics collected by test script.      |tabulate   |
|ENS-LGBM   |ENS-LGBM   |roc curves      |Compute AP, find best model, plot ROC curves. |tabulate   |
|ENS-LGBM   |ENS-LGBM   |train           |Train LGBM ensemble with ENS ranked subsets.  |logofunc3  |
|ENS-LGBM   |ENS-LGBM   |test            |Test all 499 LGBM models.                     |logofunc3  |
|ENS-LGBM   |ENS-LGBM   |graphs          |Plots both overall and class-wise metrics.    |tabulate   |
|ENS-LGBM   |ENS-LGBM   |sort-metrics    |Merges metrics collected by test script.      |tabulate   |
|ENS-LGBM   |ENS-LGBM   |generate_subsets|Generates the ENS subsets.                    |tabulate   |
|ENS-XGB    |ENS-XGB    |roc curves      |Compute AP, find best model, plot ROC curves. |tabulate   |
|ENS-XGB    |ENS-XGB    |train           |Train XGB ensemble with ENS ranked subsets.   |logofunc3  |
|ENS-XGB    |ENS-XGB    |test            |Test all 499 XGB models.                      |logofunc3  |
|ENS-XGB    |ENS-XGB    |graphs          |Plots both overall and class-wise metrics.    |tabulate   |
|ENS-XGB    |ENS-XGB    |sort-metrics    |Merges metrics collected by test script.      |tabulate   |
|ENS-XGB    |ENS-XGB    |generate_subsets|Generates the ENS subsets.                    |tabulate   |

### Classifiers directory
1. The `LGBM` and `LGBM-P` models were trained using the original LoGoFunc `train.py` script. The only difference is `LGBM` was trained using the dataset without the `Protein_dom` feature, while `LGBM-P` used the full dataset. `LGBM` was intended to serve as the benchmark to compare other classifiers to, and `LGBM-P` was included to show the impact of removing this feature. The final results showed that `LGBM` performed slightly better during testing than `LGBM-P` did, indicating that removing this feature should have negative consequences.
2. `utils.py` comes straight from the LoGoFunc GitLab repository and contains pre-processing functions that are accessed by the classifier notebooks. `utilsencoded.py` is identical aside from having feature names encoded so that they will be valid for use with XGBoost when `XGBoost.ipynb` is run.
3. The `classifiers` folder contains one notebook for each of the classifiers I assessed. These handle hyperparameter tuning, training, and testing in one notebook. The model ensembles are stored in the `models` directory, test results go to `results`, and the various performance metrics end up in the `metrics` directory.
4. The `scripts` directory contains `encode.ipynb`, which is only used to encode feature names for use with XGBoost (because it considers many of the original names invalid!). It also contains `graphing.ipynb`, which was used to generate the classifier comparison plot for the report. The `train.ipynb` in this folder is just the original LoGoFunc training script ported to a notebook for convenience.
### Ensemble ranking directory
1. `scripts/ensemble_ranking.ipynb` is the core set of code that first loads the LoGoFunc datasets with 500 features, then drops the `Protein_dom` column and exports datasets with 499 features. The new reduced datasets are used for the rest of the process.
2. Next, the pre-processing snippet encodes, imputes, and scales the dataset. This increases feature count from 499 to 539.
3. Now the `ranks` dictionary is initialized and the 12 ranking algorithms rank feature importances in series. This will take several hours (or longer).
4. Next, the mean rankings are computed and exported as `feature_ranks.csv`.
5. `decode_rankings.ipynb` runs through a series of steps to decode the ranked features using `decoding_template.json` as a guide. The final rankings are exported as `decoded_ranks.json`.
6. The `generate_subsets.ipynb` notebook uses the decoded rankings to generate train and test subsets, each containing the top "x" most important features, ranging from 1 to 499. These will be used by ENS-XGB and ENS-LGBM for training/testing.
7. The `variance.ipynb` just plotted mean ranking scores against variances for the report.
### LOFO ranking directory

1. Because XGBoost considers some of the characters in the feature names invalid, they must first be encoded by running `scripts/encode_all.ipynb`. This will generate `data/X_train_encoded.csv` and `data/X_test_encoded.csv`, which will be used as the base datasets for the remainder of the ranking process. This only encodes the feature names and leaves the actual data intact. This script also encodes two JSON files needed for the preprocessing pipeline called `data/negone_median.json` and `data/patterns.json`. These preprocessing artifacts were inherited from the original LoGoFunc repository.
2. Next, running `scripts/generate_subsets.ipynb` will create the train and test subsets, each missing a single feature. These subsets are stored in `data/train_subsets` and `data/test_subsets`, respectively.
3. Running `scripts/train.ipynb` will iterate over every subset in `data/train_subsets` and create a model for each one, forming the basis for the ranking analysis.
4. After training all of the models, `scripts/test.ipynb` will iterate over every model stored in the `models` directory. This populates both the `results` and `metrics` directories and ranks the models in reverse order according to their overall accuracy, which is stored in `metrics/ranked_by_accuracy.csv`.
5. Running the `scripts/decode.ipynb` notebook will take `metrics/ranked_by_accuracy.csv` and decode the feature names before storing them as `metrics/accuracy_rankings.json`. This makes the ranked features human readable.
6. Finally, `scripts/ranked_subsets.ipynb` reads `metrics/accuracy_rankings.json`, `data/X_train_encoded.csv` and `data/X_test_encode.csv` to generate ranked subsets with the Top "X" features for both training and test datasets. These are stored in `data/ranked_subsets/train_features_csv` and `data/ranked_subsets/test_features_csv`, respectively. These are the subsets that will be used for training and testing LOFO-LGBM and LOFO-XGB.
7. `scripts/graphing.ipynb` is used for generating the graph of the Top 25 features based on the LOFO ranking. The table in my report was generated manually using the data found in `metrics/ranked_by_accuracy.csv`.
### ENS-XGB and ENS-LGBM directory

1. The `generate_subsets.ipynb` notebook uses the decoded rankings to generate train and test subsets, each containing the top "x" most important features, ranging from 1 to 499. These will be used by ENS-XGB and ENS-LGBM for training/testing. For ENS-XGB, this notebook subsequently encodes the feature names because otherwise XGBoost will terminate with an error.
2. `train.ipynb` iterates over all 499 of the subsets and trains each model ensemble. This takes over a day to run, so I used the `tqdm` package to provide a progress bar.
3. `test.ipynb` iterates over all 499 of the trained model ensembles and exports results to the `results` folder and the various performance metrics to the `metrics` folder.
4. `sort-metrics.ipynb` is just used for combining the individual metrics and reports into a new CSV. 
5. `graphs.ipynb` first computes macro-recall from the confusion matrices (I had originally used weighted recall, which gave the same values as ACC for every model). Then it plots overall performance metrics followed by class-wise performance metrics. I used PrettyTable to display data in a tabular format for ease of reading. 
6. `roc curves.ipynb` calculates AP score (I had originally just used PREC), performs the weighting/sensitivity analysis, arrives at optimal model (feature count), and then allows you to plot the ROC curves for that model. I. used PrettyTable again to display data in a tabular format.
### LOFO-XGB and LOFO-LGBM directory

1. This workspace is pre-populated with the ranked subsets generated by the `LOFO Ranker` workspace. These are found in `data/ranked_subsets/train_features_csv` and `data/ranked_subsets/test_features_csv`.
2. `LOFO-LGBM/train.ipynb` will iterate over all of the training subsets contained in the `data/ranked_subsets/train_features_csv` directory and generate an ensemble of 27 models, which are stored in the `LOFO-LGBM/models` directory.
3. Running `LOFO-LGBM/test.ipynb` will evaluate all of the models and collect metrics.
4. Next, using `LOFO-LGBM/sort_metrics.ipynb` will run analysis on the collected metrics and collate them into merged and sorted files stored in `LOFO-LGBM/metrics`.
5. `scripts/ranking_script.ipynb` performs sensitivity analysis on the class-wise metrics to aid in selecting the overall highest-performing model (feature count).
9. `scripts/roc curve.ipynb` computes AP and plots the ROC curve for the best model. It also displays the data for the best model in a table using the `prettytable` package.
6. `LOFO-LGBM/graphs.ipynb` creates graphs both for overall and class-wise metrics.
