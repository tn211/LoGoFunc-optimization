"""This Python file contains pre-processing functions with special encoding for the XGBoost notebook."""
import time
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import  RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin

NEGONE_FEATURES = ['feature487', 'feature489', 'feature491', 'feature492', 'feature493', 'feature494', 'feature495', 'feature496', 'feature497', 'feature498', 'feature499', 'feature458', 'feature4', 'feature6', 'feature10', 'feature13', 'feature31', 'feature35', 'feature36', 'feature46', 'feature47', 'feature48', 'feature49', 'feature45', 'feature50', 'feature51', 'feature52', 'feature53', 'feature54', 'feature55', 'feature56', 'feature57', 'feature58', 'feature59', 'feature60', 'feature61', 'feature62', 'feature63', 'feature64', 'feature65', 'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature71', 'feature72', 'feature73', 'feature74', 'feature75', 'feature76', 'feature77', 'feature84', 'feature85', 'feature86', 'feature87', 'feature88', 'feature89', 'feature90', 'feature91', 'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97', 'feature98', 'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature105', 'feature107', 'feature108', 'feature109', 'feature110', 'feature111', 'feature112', 'feature113', 'feature114', 'feature115', 'feature116', 'feature117', 'feature118', 'feature119', 'feature120', 'feature121', 'feature122', 'feature123', 'feature124', 'feature125', 'feature126', 'feature127', 'feature128', 'feature129', 'feature130', 'feature131', 'feature132', 'feature133', 'feature134', 'feature135', 'feature136', 'feature137', 'feature138', 'feature139', 'feature140', 'feature141', 'feature142', 'feature143', 'feature144', 'feature145', 'feature146', 'feature147', 'feature148', 'feature149', 'feature150', 'feature151', 'feature152', 'feature153', 'feature154', 'feature155', 'feature156', 'feature157', 'feature158', 'feature159', 'feature160', 'feature161', 'feature162', 'feature163', 'feature164', 'feature165', 'feature166', 'feature167', 'feature168', 'feature169', 'feature170', 'feature171', 'feature172', 'feature173', 'feature174', 'feature175', 'feature176', 'feature177', 'feature178', 'feature179', 'feature180', 'feature181', 'feature182', 'feature183', 'feature184', 'feature185', 'feature186', 'feature187', 'feature188', 'feature189', 'feature190', 'feature191', 'feature192', 'feature193', 'feature194', 'feature195', 'feature196', 'feature197', 'feature198', 'feature199', 'feature200', 'feature201', 'feature202', 'feature203', 'feature204', 'feature205', 'feature206', 'feature207', 'feature208', 'feature209', 'feature210', 'feature211', 'feature212', 'feature213', 'feature214', 'feature215', 'feature216', 'feature217', 'feature218', 'feature219', 'feature220', 'feature221', 'feature222', 'feature223', 'feature224', 'feature225', 'feature226', 'feature227', 'feature228', 'feature229', 'feature230', 'feature231', 'feature232', 'feature233', 'feature234', 'feature235', 'feature236', 'feature237', 'feature238', 'feature239', 'feature240', 'feature241', 'feature242', 'feature243', 'feature244', 'feature245', 'feature246', 'feature247', 'feature248', 'feature249', 'feature250', 'feature251', 'feature252', 'feature253', 'feature254', 'feature255', 'feature256', 'feature257', 'feature258', 'feature259', 'feature260', 'feature261', 'feature262', 'feature263', 'feature264', 'feature265', 'feature266', 'feature267', 'feature268', 'feature269', 'feature270', 'feature271', 'feature272', 'feature273', 'feature274', 'feature275', 'feature276', 'feature277', 'feature278', 'feature279', 'feature280', 'feature281', 'feature282', 'feature283', 'feature284', 'feature285', 'feature286', 'feature287', 'feature288', 'feature289', 'feature290', 'feature291', 'feature292', 'feature293', 'feature294', 'feature295', 'feature296', 'feature297', 'feature298', 'feature299', 'feature300', 'feature301', 'feature302', 'feature303', 'feature304', 'feature305', 'feature306', 'feature307', 'feature308', 'feature309', 'feature310', 'feature311', 'feature312', 'feature313', 'feature314', 'feature315', 'feature316', 'feature317', 'feature318', 'feature319', 'feature320', 'feature321', 'feature322', 'feature323', 'feature324', 'feature325', 'feature326', 'feature327', 'feature328', 'feature329', 'feature330', 'feature331', 'feature332', 'feature333', 'feature334', 'feature335', 'feature336', 'feature337', 'feature338', 'feature339', 'feature340', 'feature341', 'feature342', 'feature343', 'feature344', 'feature345', 'feature346', 'feature347', 'feature348', 'feature349', 'feature355', 'feature356', 'feature357', 'feature358', 'feature359', 'feature360', 'feature361', 'feature362', 'feature363', 'feature364', 'feature365', 'feature366', 'feature367', 'feature368', 'feature369', 'feature373', 'feature374', 'feature375', 'feature376', 'feature421', 'feature422', 'feature423', 'feature424', 'feature425', 'feature426', 'feature427', 'feature428', 'feature429', 'feature430', 'feature431', 'feature432', 'feature433', 'feature434', 'feature435', 'feature436', 'feature437', 'feature438', 'feature439', 'feature440', 'feature441', 'feature442', 'feature443', 'feature444', 'feature445', 'feature446', 'feature447', 'feature448', 'feature459', 'feature460', 'feature461', 'feature462', 'feature463', 'feature464', 'feature465', 'feature466', 'feature467', 'feature471', 'feature472', 'feature473', 'feature474', 'feature475', 'feature476', 'feature477', 'feature478', 'feature479', 'feature480', 'feature481', 'feature482', 'feature483', 'feature484', 'feature485', 'feature486']

MEDIAN_FEATURES = ['feature3', 'feature5', 'feature7', 'feature8', 'feature9', 'feature11', 'feature12', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 'feature32', 'feature33', 'feature34', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42', 'feature43', 'feature79', 'feature80', 'feature81', 'feature82', 'feature350', 'feature351', 'feature352', 'feature353', 'feature354', 'feature370', 'feature371', 'feature372', 'feature378', 'feature379', 'feature380', 'feature381', 'feature382', 'feature383', 'feature384', 'feature385', 'feature386', 'feature387', 'feature388', 'feature389', 'feature390', 'feature391', 'feature392', 'feature393', 'feature394', 'feature395', 'feature396', 'feature397', 'feature398', 'feature399', 'feature400', 'feature401', 'feature402', 'feature403', 'feature404', 'feature405', 'feature406', 'feature407', 'feature408', 'feature409', 'feature410', 'feature411', 'feature412', 'feature413', 'feature414', 'feature415', 'feature416', 'feature417', 'feature418', 'feature419', 'feature420', 'feature449', 'feature450', 'feature451', 'feature452', 'feature453', 'feature454', 'feature455', 'feature456', 'feature457', 'feature469', 'feature470']

def generate_preprocessor(numeric_features, categorical_features, N_JOBS, cat_encode_type, 
                            do_specificimpute, do_featureselection, 
                            do_sampling, do_pca, var_thresh, oversample_technique, 
                            negone_features=NEGONE_FEATURES, median_features=MEDIAN_FEATURES,
                            prefix='', do_feature_subset=False, max_features=1, do_removeppi=False, do_removegtex=False):
    cat_encoders = [OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1), 
                    OneHotEncoder(sparse=False, handle_unknown='infrequent_if_exist', min_frequency=10)]
    categorical_transformer = cat_encoders[cat_encode_type]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler(feature_range =(0, 1), clip=True))])

    median_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler(feature_range =(0, 1), clip=True))])

    negone_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler(feature_range =(0, 1), clip=True)),
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
    ])

    preprocessor = None
    if do_specificimpute:
        preprocessor = ColumnTransformer(
            transformers=[
                ('median', median_transformer, median_features),
                ('negone', negone_transformer, negone_features),
                ('cat', categorical_transformer, categorical_features),
        ])
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ])

    vt = VarianceThreshold(threshold=var_thresh)
    steps = [('initial', preprocessor), ('removeba', RemoveBeforeAfterTransformer()), ('variance_threshold', vt)]
    if do_sampling == 1:
        steps.append(('undersampling', RandomUnderSampler(random_state=42)))
    if do_sampling == 2:
        oversamplers = [SMOTE(n_jobs=N_JOBS,random_state=42), RandomOverSampler(random_state=42)]
        steps.append(('oversampling', oversamplers[oversample_technique]))
    if do_pca:
        steps.append(('pca', PCA()))

    preprocessor = Pipeline(steps=steps)
    return preprocessor

class RemoveBeforeAfterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drop_cols = None

    def fit(self, X, y=None):
        print(X.shape)
        self.drop_cols = [f'feature{i}' for i in [35, 36, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]]
        return self

    def transform(self, X, y=None):
        X = X.drop(columns=self.drop_cols)
        print(X.shape)
        return X

    def get_feature_names_out(self, input_features=None):
        return [f for f in input_features if f not in self.drop_cols]
    
def preprocess(preprocessor, train_data, train_labels, quiet=False):
    for k, v in preprocessor.steps:
        if k == 'initial':
            start = time.time()
            v.fit(train_data)
            train_data = pd.DataFrame(v.transform(train_data), columns=v.get_feature_names_out())
            end = time.time()
            if not quiet:
                print(k + ' took ' + str(end - start) + ' to run.')
        elif k == 'oversampling' or k == 'undersampling':
            start = time.time()
            input_features = train_data.columns
            train_data, train_labels = v.fit_resample(train_data, train_labels)
            train_data = pd.DataFrame(train_data, columns=input_features)
            end = time.time()
            if not quiet:
                print(k + ' took ' + str(end - start) + ' to run.')
        else:
            start = time.time()
            v.fit(train_data)
            input_features = train_data.columns
            train_data = pd.DataFrame(v.transform(train_data), columns=v.get_feature_names_out(input_features))
            end = time.time()
            if not quiet:
                print(k + ' took ' + str(end - start) + ' to run.')

    for col in train_data.columns:
        try:
            train_data[col] = train_data[col].astype('float')
        except:
            train_data[col] = train_data[col].astype('category')

    return train_data, train_labels

def transform(test_data, preprocessor, quiet=False):
    for k, v in preprocessor.steps:
        if k == 'initial':
            test_data = pd.DataFrame(v.transform(test_data), columns=v.get_feature_names_out())
        elif k == 'oversampling' or k == 'undersampling':
            continue
        else:
            test_data = v.transform(test_data)
    test_data = pd.DataFrame(test_data)

    for col in test_data.columns:
        try:
            test_data[col] = test_data[col].astype('float')
        except:
            test_data[col] = test_data[col].astype('category')

    return test_data.to_numpy()
