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

NEGONE_FEATURES = ['MOD_RES','REGION','INTERACTION_REGION','REQUIRED_FOR_INTER','ATP_binding_gbind','Ca2+_binding_gbind','DNA_binding_gbind','HEME_binding_gbind','Mg2+_binding_gbind','Mn2+_binding_gbind','RNA_binding_gbind','Dist2Mutation','BLOSUM62','ProteinLengthChange','TSSDistance','1000Gp3_AF','UK10K_AF','gnomAD_exomes_AF','gnomAD_genomes_AF','MSC_95CI','rel_cDNA_pos','rel_CDS_pos','rel_prot_pos','GDI','Selective_pressure','Clarks_distance','CDS_len','Number_of_paralogs','denovo_Zscore','RVIS','Indispensability_score','RSA','ASA','RSA_Zfit','before_RSA_3','before_RSA_8','before_RSA_15','after_RSA_3','after_RSA_8','after_RSA_15','before_ASA_3','before_ASA_8','before_ASA_15','after_ASA_3','after_ASA_8','after_ASA_15','Phosphorylation','Acetylation','Methylation','Ubiquitination','Glycosylation','PTM','AF_Relative_ASA','IUPRED2','ANCHOR2','before_IUPRED_3','before_IUPRED_8','before_IUPRED_15','after_IUPRED_3','after_IUPRED_8','after_IUPRED_15','before_ANCHOR_3','before_ANCHOR_8','before_ANCHOR_15','after_ANCHOR_3','after_ANCHOR_8','after_ANCHOR_15','A3D_SCORE','n_contacts','distance_com','concavity_score','S_DDG[SEQ]','S_DDG[3D]','hgmd_mutcount','gnomsingle_mutcount','gnom_mutcount','AF_confidence','isHomomultimer','num_interactions','ppi_combined_0','ppi_combined_1','ppi_combined_2','ppi_combined_3','ppi_combined_4','ppi_combined_5','ppi_combined_6','ppi_combined_7','ppi_combined_8','ppi_combined_9','ppi_combined_10','ppi_combined_11','ppi_combined_12','ppi_combined_13','ppi_combined_14','ppi_combined_15','ppi_combined_16','ppi_combined_17','ppi_combined_18','ppi_combined_19','ppi_combined_20','ppi_combined_21','ppi_combined_22','ppi_combined_23','ppi_combined_24','ppi_combined_25','ppi_combined_26','ppi_combined_27','ppi_combined_28','ppi_combined_29','ppi_combined_30','ppi_combined_31','ppi_combined_32','ppi_combined_33','ppi_combined_34','ppi_combined_35','ppi_combined_36','ppi_combined_37','ppi_combined_38','ppi_combined_39','ppi_combined_40','ppi_combined_41','ppi_combined_42','ppi_combined_43','ppi_combined_44','ppi_combined_45','ppi_combined_46','ppi_combined_47','ppi_combined_48','ppi_combined_49','ppi_combined_50','ppi_combined_51','ppi_combined_52','ppi_combined_53','ppi_combined_54','ppi_combined_55','ppi_combined_56','ppi_combined_57','ppi_combined_58','ppi_combined_59','ppi_combined_60','ppi_combined_61','ppi_combined_62','ppi_combined_63','s_het','DRNApredDNAscore_aa_window_3_prev','DRNApredDNAscore_aa_window_8_prev','DRNApredDNAscore_aa_window_15_prev','DRNApredDNAscore_aa_window_3_next','DRNApredDNAscore_aa_window_8_next','DRNApredDNAscore_aa_window_15_next','DRNApredDNAscore_aa','ASAquick_normscore_aa_window_3_prev','ASAquick_normscore_aa_window_8_prev','ASAquick_normscore_aa_window_15_prev','ASAquick_normscore_aa_window_3_next','ASAquick_normscore_aa_window_8_next','ASAquick_normscore_aa_window_15_next','ASAquick_normscore_aa','ASAquick_rawscore_aa_window_3_prev','ASAquick_rawscore_aa_window_8_prev','ASAquick_rawscore_aa_window_15_prev','ASAquick_rawscore_aa_window_3_next','ASAquick_rawscore_aa_window_8_next','ASAquick_rawscore_aa_window_15_next','ASAquick_rawscore_aa','DFLpredScore_aa_window_3_prev','DFLpredScore_aa_window_8_prev','DFLpredScore_aa_window_15_prev','DFLpredScore_aa_window_3_next','DFLpredScore_aa_window_8_next','DFLpredScore_aa_window_15_next','DFLpredScore_aa','DRNApredRNAscore_aa_window_3_prev','DRNApredRNAscore_aa_window_8_prev','DRNApredRNAscore_aa_window_15_prev','DRNApredRNAscore_aa_window_3_next','DRNApredRNAscore_aa_window_8_next','DRNApredRNAscore_aa_window_15_next','DRNApredRNAscore_aa','DisoDNAscore_aa_window_3_prev','DisoDNAscore_aa_window_8_prev','DisoDNAscore_aa_window_15_prev','DisoDNAscore_aa_window_3_next','DisoDNAscore_aa_window_8_next','DisoDNAscore_aa_window_15_next','DisoDNAscore_aa','DisoPROscore_aa_window_3_prev','DisoPROscore_aa_window_8_prev','DisoPROscore_aa_window_15_prev','DisoPROscore_aa_window_3_next','DisoPROscore_aa_window_8_next','DisoPROscore_aa_window_15_next','DisoPROscore_aa','DisoRNAscore_aa_window_3_prev','DisoRNAscore_aa_window_8_prev','DisoRNAscore_aa_window_15_prev','DisoRNAscore_aa_window_3_next','DisoRNAscore_aa_window_8_next','DisoRNAscore_aa_window_15_next','DisoRNAscore_aa','MMseq2_conservation_level_aa_window_3_prev','MMseq2_conservation_level_aa_window_8_prev','MMseq2_conservation_level_aa_window_15_prev','MMseq2_conservation_level_aa_window_3_next','MMseq2_conservation_level_aa_window_8_next','MMseq2_conservation_level_aa_window_15_next','MMseq2_conservation_level_aa','MMseq2_conservation_score_aa_window_3_prev','MMseq2_conservation_score_aa_window_8_prev','MMseq2_conservation_score_aa_window_15_prev','MMseq2_conservation_score_aa_window_3_next','MMseq2_conservation_score_aa_window_8_next','MMseq2_conservation_score_aa_window_15_next','MMseq2_conservation_score_aa','MoRFchibiScore_aa_window_3_prev','MoRFchibiScore_aa_window_8_prev','MoRFchibiScore_aa_window_15_prev','MoRFchibiScore_aa_window_3_next','MoRFchibiScore_aa_window_8_next','MoRFchibiScore_aa_window_15_next','MoRFchibiScore_aa','PSIPRED_helix_aa_window_3_prev','PSIPRED_helix_aa_window_8_prev','PSIPRED_helix_aa_window_15_prev','PSIPRED_helix_aa_window_3_next','PSIPRED_helix_aa_window_8_next','PSIPRED_helix_aa_window_15_next','PSIPRED_helix_aa','PSIPRED_strand_aa_window_3_prev','PSIPRED_strand_aa_window_8_prev','PSIPRED_strand_aa_window_15_prev','PSIPRED_strand_aa_window_3_next','PSIPRED_strand_aa_window_8_next','PSIPRED_strand_aa_window_15_next','PSIPRED_strand_aa','SCRIBERscore_aa_window_3_prev','SCRIBERscore_aa_window_8_prev','SCRIBERscore_aa_window_15_prev','SCRIBERscore_aa_window_3_next','SCRIBERscore_aa_window_8_next','SCRIBERscore_aa_window_15_next','SCRIBERscore_aa','SignalP_score_aa_window_3_prev','SignalP_score_aa_window_8_prev','SignalP_score_aa_window_15_prev','SignalP_score_aa_window_3_next','SignalP_score_aa_window_8_next','SignalP_score_aa_window_15_next','SignalP_score_aa','gtex_Adipose_-_Subcutaneous','gtex_Adipose_-_Visceral_(Omentum)','gtex_Adrenal_Gland','gtex_Artery_-_Aorta','gtex_Artery_-_Coronary','gtex_Artery_-_Tibial','gtex_Bladder','gtex_Brain_-_Amygdala','gtex_Brain_-_Anterior_cingulate_cortex_(BA24)','gtex_Brain_-_Caudate_(basal_ganglia)','gtex_Brain_-_Cerebellar_Hemisphere','gtex_Brain_-_Cerebellum','gtex_Brain_-_Cortex','gtex_Brain_-_Frontal_Cortex_(BA9)','gtex_Brain_-_Hippocampus','gtex_Brain_-_Hypothalamus','gtex_Brain_-_Nucleus_accumbens_(basal_ganglia)','gtex_Brain_-_Putamen_(basal_ganglia)','gtex_Brain_-_Spinal_cord_(cervical_c-1)','gtex_Brain_-_Substantia_nigra','gtex_Breast_-_Mammary_Tissue','gtex_Cells_-_Cultured_fibroblasts','gtex_Cells_-_EBV-transformed_lymphocytes','gtex_Cervix_-_Ectocervix','gtex_Cervix_-_Endocervix','gtex_Colon_-_Sigmoid','gtex_Colon_-_Transverse','gtex_Esophagus_-_Gastroesophageal_Junction','gtex_Esophagus_-_Mucosa','gtex_Esophagus_-_Muscularis','gtex_Fallopian_Tube','gtex_Heart_-_Atrial_Appendage','gtex_Heart_-_Left_Ventricle','gtex_Kidney_-_Cortex','gtex_Kidney_-_Medulla','gtex_Liver','gtex_Lung','gtex_Minor_Salivary_Gland','gtex_Muscle_-_Skeletal','gtex_Nerve_-_Tibial','gtex_Ovary','gtex_Pancreas','gtex_Pituitary','gtex_Prostate','gtex_Skin_-_Not_Sun_Exposed_(Suprapubic)','gtex_Skin_-_Sun_Exposed_(Lower_leg)','gtex_Small_Intestine_-_Terminal_Ileum','gtex_Spleen','gtex_Stomach','gtex_Testis','gtex_Thyroid','gtex_Uterus','gtex_Vagina','gtex_Whole_Blood','haplo','haplo_imputed','PHOSPHORYLATION','ACETYLATION','UBIQUITINATION','S-NITROSYLATION','N-GLYCOSYLATION','METHYLATION','O-GLYCOSYLATION','MYRISTOYLATION','C-GLYCOSYLATION','SUMOYLATION','S-GLYCOSYLATION','polyphen_nobs','polyphen_normasa','polyphen_dvol','polyphen_dprop','polyphen_bfact','polyphen_hbonds','polyphen_avenhet','polyphen_mindhet','polyphen_avenint','polyphen_mindint','polyphen_avensit','polyphen_mindsit','polyphen_idpmax','polyphen_idpsnp','polyphen_idqmin','motifECount','motifEHIPos','motifEScoreChng','Dst2Splice','motifDist','EncodeH3K4me1-sum','EncodeH3K4me1-max','EncodeH3K4me2-sum','EncodeH3K4me2-max','EncodeH3K4me3-sum','EncodeH3K4me3-max','EncodeH3K9ac-sum','EncodeH3K9ac-max','EncodeH3K9me3-sum','EncodeH3K9me3-max','EncodeH3K27ac-sum','EncodeH3K27ac-max','EncodeH3K27me3-sum','EncodeH3K27me3-max','EncodeH3K36me3-sum','EncodeH3K36me3-max','EncodeH3K79me2-sum','EncodeH3K79me2-max','EncodeH4K20me1-sum','EncodeH4K20me1-max','EncodeH2AFZ-sum','EncodeH2AFZ-max','EncodeDNase-sum','EncodeDNase-max','EncodetotalRNA-sum','EncodetotalRNA-max','Grantham_x','Freq100bp','Rare100bp','Sngl100bp','Freq1000bp','Rare1000bp','Sngl1000bp','Freq10000bp','Rare10000bp','Sngl10000bp','RemapOverlapTF','RemapOverlapCL','Charge','Volume','Hydrophobicity','Polarity','Ex','PAM250','JM','HGMD2003','VB','Transition','COSMIC','COSMICvsSWISSPROT','HAPMAP','COSMICvsHAPMAP',]
MEDIAN_FEATURES = ['CADD_raw','Conservation','MaxEntScan_alt','MaxEntScan_diff','MaxEntScan_ref','ada_score','rf_score','FATHMM_score','GERPplus_plus_NR','GERPplus_plus_RS','GM12878_fitCons_score','GenoCanyon_score','H1_hESC_fitCons_score','HUVEC_fitCons_score','LINSIGHT','LIST_S2_score','LRT_score','M_CAP_score','MPC_score','MVP_score','MutationAssessor_score','MutationTaster_score','PROVEAN_score','SiPhy_29way_logOdds','VEST4_score','fathmm_MKL_coding_score','fathmm_XF_coding_score','integrated_fitCons_score','phastCons100way_vertebrate','phastCons17way_primate','phastCons30way_mammalian','phyloP100way_vertebrate','phyloP17way_primate','phyloP30way_mammalian','Condel_score','SIFT_score','NearestExonJB_distance','NearestExonJB_len','Dominant_probability','Recessive_probability','polyphen_dscore','polyphen_score1','polyphen_score2','ConsScore','GC','CpG','minDistTSS','minDistTSE','priPhCons','mamPhCons','verPhCons','priPhyloP','mamPhyloP','verPhyloP','bStatistic_y','targetScan','mirSVR-Score','mirSVR-E','mirSVR-Aln','cHmm_E1','cHmm_E2','cHmm_E3','cHmm_E4','cHmm_E5','cHmm_E6','cHmm_E7','cHmm_E8','cHmm_E9','cHmm_E10','cHmm_E11','cHmm_E12','cHmm_E13','cHmm_E14','cHmm_E15','cHmm_E16','cHmm_E17','cHmm_E18','cHmm_E19','cHmm_E20','cHmm_E21','cHmm_E22','cHmm_E23','cHmm_E24','cHmm_E25','GerpRS','GerpRSpval','GerpN','GerpS','tOverlapMotifs','SpliceAI-acc-gain','SpliceAI-acc-loss','SpliceAI-don-gain','SpliceAI-don-loss','MMSp_acceptorIntron','MMSp_acceptor','MMSp_exon','MMSp_donor','MMSp_donorIntron','dbscSNV-ada_score','dbscSNV-rf_score',]

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
        self.drop_cols = X.columns[X.columns.str.contains('after|before|gnomAD_exomes_AF|gnomAD_genomes_AF', regex=True)]
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
