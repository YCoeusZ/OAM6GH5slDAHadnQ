from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np 
import pandas as pd
from sklearn.feature_selection import f_classif
import itertools
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import matplotlib.pyplot as plt 
from scipy.stats import norm

# def acc_at_cutoff(y_true, y_porbs, cutoff):
#     """
#     A function expecting output of binary predict_proba output and return the classification according to the cutoff. 
#     Designed to be used in sklearn make_scorer. 
    
#     :param y_true: The actual values. 
#     :param y_probs: Binary preditct_proba output. 
#     :param cutoff: The cutoff threshold. 
#     :return: The classification. 
#     """
#     if y_porbs.ndim == 2: 
#         y_prob=y_porbs[:,-1]
#     else: 
#         y_prob=y_porbs
#     y_p=1*(y_prob>=cutoff)
#     return accuracy_score(y_pred=y_p,y_true=y_true)  

def LDA_LR_feat_imp(pipe: Pipeline, LDA_name: str = "LDA", LR_name: str = "LogReg", n_class: int=2): 
    """
    A function that provides the index of the least important feature of a trained LDA Log Regression sklearn pipeline. 
    
    :param pipe: The trained sklearn pipeline. 
    :param LDA_name: Defaulted to "LDA". The name of LDA step in pipe as a string. 
    :param LR_name: Defaulted to "LogReg". The name of Log Regression step in pipe as a string. 
    :param n_class: Defaulted to 2. The number of classes in the classification task as an integer. 
    :return: The index, in the ordered list of features, the least imporant feature, as an integer. 
    """
    W=pipe[LDA_name].scalings_[:,:(n_class-1)]
    if n_class==2: 
        b=pipe[LR_name].coef_.ravel()
    else: 
        b=pipe[LR_name].coef_.T 
    IMP=np.absolute(W @ b)
    return int(IMP.argsort()[0])

class data_creator(BaseEstimator, TransformerMixin):
    def __init__(self, counts=False):
        """    
        Creates manufactured data "mean", "F_w_mean", "above_3", "above_4", and "above_5". Can also create count of 1 to 5 as well if set counts to True. 
        
        :param counts: Defaulted to False. Set to true to count the number of 1 to 5s. 
        """
        self.counts=counts
        pass 
    
    def fit(self, X: pd.DataFrame, y): 
        if type(y)!= pd.DataFrame:
            y_c=pd.DataFrame({"Y":y})
        else: 
            y_c=y.copy(deep=True)
        X_c=X.copy(deep=True)
        self.f_arr_=f_classif(X=X_c,y=y_c["Y"])[0]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: 
        check_is_fitted(self, "f_arr_")
        X_copy=X.copy(deep=True)
        X_copy["mean"]=X_copy[X.columns].mean(axis=1)
        X_copy["F_w_mean"]=X_copy[X.columns].apply(lambda row_arr: (np.array(row_arr)* self.f_arr_).mean(), axis=1)
        # x_t_copy["mi_w_mean"]=x_t_copy[features].apply(lambda row_arr: (np.array(row_arr)*mutual_info_classif(X=df2[features],y=df2["Y"],discrete_features=True)).sum()/len(features), axis=1)
        X_copy["above_3"]=X_copy[X.columns].apply(lambda row_arr: (row_arr >= 3).mean(), axis=1)
        X_copy["above_4"]=X_copy[X.columns].apply(lambda row_arr: (row_arr >= 4).mean(), axis=1)
        X_copy["above_5"]=X_copy[X.columns].apply(lambda row_arr: (row_arr >= 5).mean(), axis=1)
        if self.counts: 
            X_copy["count_1"]=X_copy[X.columns].apply(lambda row_arr: (row_arr == 1).sum(), axis=1)
            X_copy["count_2"]=X_copy[X.columns].apply(lambda row_arr: (row_arr == 2).sum(), axis=1)
            X_copy["count_3"]=X_copy[X.columns].apply(lambda row_arr: (row_arr == 3).sum(), axis=1)
            X_copy["count_4"]=X_copy[X.columns].apply(lambda row_arr: (row_arr == 4).sum(), axis=1)
            X_copy["count_5"]=X_copy[X.columns].apply(lambda row_arr: (row_arr == 5).sum(), axis=1)
        return X_copy 

class data_selector(BaseEstimator, TransformerMixin):
    def __init__(self,f_lower:float=1.0,p_upper:float=0.1,how:str="and", force=None):
        # self.X=None
        # self.y=None
        # self.total=None
        # self.sel=None
        """
        Eliminates unwanted feature according to f socre and p value given by anova f test. 
        
        :param f_lower: Defaulted to 1. The lower bound for f score. 
        :param p_upper: Defaulted to 0.1. The upper bound for p value. 
        :param how: Defaulted to "and". If changed to "or" change the logic connecter between the p value and f score condition to or. 
        :param force: Defaulted to None. If set to a list of parameters, it forces that to be the output features, ignoring all other parameters in this function. 
        """
        
        self.f_lower=f_lower
        self.p_upper=p_upper
        self.how=how
        self.force=force
        
    def fit(self, X, y): 
        if self.force is None: 
            if type(y)!= pd.DataFrame:
                y_c=pd.DataFrame({"Y":y})
            else: 
                y_c=y.copy(deep=True)
            X_c=X.copy(deep=True)
            # print(type(self.X))
            f_score, p_value= f_classif(X=X_c,y=y_c["Y"])
            df_f=pd.DataFrame({
                "features": X_c.columns, 
                "f score": f_score, 
                "p value": p_value
            })
            X_c["Y"]=y_c["Y"]
            df_corr=X_c.corr()
            # print(df_corr)
            df_corr["features"]=df_corr.index
            self.total_=pd.merge(left=df_f,right=df_corr,how="left",on=["features"])
            # print(self.total)
            if self.how=="and": 
                self.sel_=self.total_[(self.total_["f score"]>=self.f_lower)&(self.total_["p value"]<=self.p_upper)]
            elif self.how=="or": 
                self.sel_=self.total_[(self.total_["f score"]>=self.f_lower)|(self.total_["p value"]<=self.p_upper)]
            else: 
                raise ValueError("parameter how is wrong.")
            self.feat_sel_=list(self.sel_["features"].values)
            return self 
        else: 
            self.feat_sel_=list(self.force)
            return self
            
    def transform(self,X) -> pd.DataFrame:
        check_is_fitted(self, "feat_sel_")
        return X[self.feat_sel_]
        
def all_combin(arr_in): 
    """
    Creates all none empty combinations of arr_in. 
    
    :param arr_in: The input array like. 
    :return: A list [(...), (...), ... ] of none empty combinations of arr_in. 
    """
    all_combin=[]
    for n_chosen in range(1, len(arr_in)+1): 
        all_combin.extend(list(itertools.combinations(arr_in, n_chosen)))
    return all_combin

def show_result(splits, feat, tar, pipe): 
    """
    A function that shows the accuracy distribution, the imporance by permutation importance method, and finally the "average confusion matrix". 
    
    :param splits: The splits to perform the repeated experiment on. 
    :param feat: The feature pd.DataFrame input. 
    :param tar: The target pd.DataFrame input. 
    :param pipe: The model pipe to be experimented on. 
    """
    list_fold_acc=[]
    list_fold_f1=[]
    imp_record=None
    cmatrix_record=None
    # n_split = 5
    # n_repeats = 20
    # RSKF = RepeatedStratifiedKFold(n_splits=n_split, random_state=420, n_repeats=n_repeats)
    # splits = list(RSKF.split(X=feat, y=tar))
    for train_index, test_index in splits: 
        x_tr,x_te=feat.iloc[train_index], feat.iloc[test_index]
        y_tr, y_te=tar.iloc[train_index], tar.iloc[test_index]
            
        y_tr=np.ravel(y_tr.values)
        y_te=np.ravel((y_te.values))
            
        pipe = clone(pipe)
        pipe.fit(X=x_tr,y=y_tr)
        y_p=pipe.predict(X=x_te)
        acc=accuracy_score(y_pred=y_p,y_true=y_te)
        f1=f1_score(y_pred=y_p,y_true=y_te)
        list_fold_acc.append(acc)
        list_fold_f1.append(f1)
        # with parallel_backend("threading", n_jobs=-1):
        #     with threadpool_limits(limits=1):
        imp=permutation_importance(pipe,X=x_te.copy(deep=True),y=y_te,scoring="accuracy",n_repeats=30,n_jobs=-1,random_state=420)
        if imp_record is None: 
            imp_record=imp.importances_mean
        else: 
            imp_record=imp_record+imp.importances_mean
        cmatrix=confusion_matrix(y_pred=y_p,y_true=y_te)
        cmatrix=cmatrix/np.sum(cmatrix)
        if cmatrix_record is None: 
            cmatrix_record=cmatrix
        else: 
            cmatrix_record=cmatrix_record+cmatrix
        

    plt.hist(list_fold_acc,bins=25)
    plt.show()

    imp_record=imp_record/len(splits)
    imp_sort_index=imp_record.argsort()
    plt.barh(feat.columns[imp_sort_index], imp_record[imp_sort_index])
    plt.title("Feature importance by permutaion method")
    plt.ylabel("Features")
    plt.xlabel("Importance")
    plt.show()

    cmatrix_record=cmatrix_record/len(splits)
    print(cmatrix_record) #This is the "Average confusion matrix" 
    
def evaluate_combo(list_f_sel_tuple, dict_param, pipe, splits, feat, tar):
    """
    Evaluate one (feature_set, n_neighbors) across all CV folds.
    
    :param list_f_sel_tuple: A tuple indicating a combination. 
    :param dict_param: The dictionary of parameters used to adjust the pipeline. Expect the same formate as param_grid for GridSeaerchCV: {"{step_name}__{hyper-parameter_name}": value, ... }. 
    :param pipe: The model pipeline used. Expect that the DataSelector step to be samed as "DataSelector". 
    :param splits: A list of the pre generated splits. 
    :param feat: The feat df. 
    :param tar: The tar df. 
    :return: A dict with all the stats we want. 
    """
    list_f_sel = list(list_f_sel_tuple) 
    fold_acc = []
    fold_f1 = []

    for train_index, test_index in splits:
        x_tr, x_te = feat.iloc[train_index], feat.iloc[test_index]
        y_tr, y_te = tar.iloc[train_index], tar.iloc[test_index]

        y_tr = np.ravel(y_tr.values)
        y_te = np.ravel(y_te.values)
        
        pipe = clone(pipe)
        
        ori_dict_param_keys=list(dict_param.keys())
        
        dict_param["DataSelector__force"]=list_f_sel 
        
        pipe.set_params(**dict_param) 
            
        pipe.fit(X=x_tr, y=y_tr)
        y_p = pipe.predict(X=x_te)

        fold_acc.append(accuracy_score(y_true=y_te, y_pred=y_p))
        fold_f1.append(f1_score(y_true=y_te, y_pred=y_p))

    str_features = ",".join(list_f_sel)
    acc_mean = float(np.mean(fold_acc))
    acc_std  = float(np.std(fold_acc))
    f1_mean  = float(np.mean(fold_f1))
    f1_std   = float(np.std(fold_f1))
    above_73 = float((np.array(fold_acc) >= 0.73).sum() / (len(splits)))
    norm_above_73 = float(1-norm.cdf(0.73, loc=acc_mean, scale=acc_std))
    acc_mean_above_73 = float(1-norm.cdf(0.73, loc=acc_mean, scale=acc_std/np.sqrt(len(splits))))

    msg = (
        "_"*20 + "\n"
        + f"Currently used features {str_features}.\n"
        )
    
    for key in ori_dict_param_keys: 
        key_param_name=key.split("__")[-1]
        key_param_value=dict_param[key]
        msg=msg+ (
            f"With {key_param_name} at {key_param_value}. \n"
        )
    
    msg=msg+(   
            f"This combo has f1 mean {f1_mean} and f1 std {f1_std}, \n"
            + f"with acc mean {acc_mean} acc std {acc_std}, "
            + f"and sureness of beating 73% {above_73}.\n"
            + "_"*20
        )
    
    return_dict={
        #Hyper-parameters
        "features": str_features,
        #Performance
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "above_73": above_73,
        "norm_above_73": norm_above_73,
        "acc_mean_above_73": acc_mean_above_73,
        #Log
        "log": msg,
    }
    
    for key in ori_dict_param_keys: 
        return_dict[key.split("__")[-1]]=dict_param[key]

    return return_dict
    