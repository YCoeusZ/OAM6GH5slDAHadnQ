from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import numpy as np 
import pandas as pd
from sklearn.feature_selection import f_classif 
import itertools
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

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
    def __init__(self):
        """
        Creates manufactured data "mean", "F_w_mean", "above_3", "above_4", and "above_5". 
        """
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