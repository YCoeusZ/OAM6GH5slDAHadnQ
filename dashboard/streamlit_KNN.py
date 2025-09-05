import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.feature_selection import f_classif
import itertools
import sys
import importlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, make_scorer
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits
import matplotlib.pylab as plt
import os
from sklearn.inspection import permutation_importance
from scipy.stats import norm, t
from sklearn.base import clone
import streamlit as st 

sys.path.append("../")
from proj_mod import training
importlib.reload(training);

st.set_page_config(page_title="What-if Prediction of Customer Sentiment Improvement", layout="centered")
st.title("What-if Prediction of Customer Sentiment Improvement")

st.markdown("""
            ## Description 
            
            This streamlit dashboard provides insight into different strategies' impact on customer sentiment. 
            Histograms of percentage (values between 0 and 1) improvement of "happiness ratio" relative to both predicted and actual "happiness ratio" are provided for visualization. 
            
            The model used is the best KNN model with: 
            
            | Used Raw Features | Used (Manufactured) Features | KNN Number of Neighbors | 
            |-------------------|------------------------------|-------------------------| 
            | X1, X3, X4, X6    | X1, X6, F_w_mean             | 5                       |
            """)

with st.expander(label="Detail Description and Settings"): 
    st.markdown("""
                The process uses sklearn RepeatedStratifiedKFold with 5 splits, 20 repeats, and seed 420. 
                The data used, by default is kept at "../data/raw.csv". 
                
                **PLEASE give the process some time!** 
                
                The process is as following, in order: 
                
                * Trained model for each one of the splits is fitted with the train data of the split, after which, the model is kept. 
                * For each of the splits, prediction is made on both the original test set and the "improved" test set according the input values given. After which, the improvements of "happiness ratio" are recorded. 
                * Graph, mean, and std of the improvement of the "happiness ratio" of the (100, by default) splits are generated and provided. 
                """)
    
    col_split, col_repeats, col_seed = st.columns(spec=3)
    
    str_data=st.text_input(
        label="The relative path to the used data:", 
        value="../data/raw.csv"
    )
        
    with col_split:
         n_split=st.number_input(
             label="Split number:", 
             value=5, 
             min_value=2,
             step=1
         )
        
    with col_repeats: 
        n_repeats=st.number_input(
            label="Repeat number:", 
            value=20, 
            min_value=1, 
            step=1
        )
        
    with col_seed: 
        seed=st.number_input(
            label="Seed:", 
            value=420, 
            min_value=0, 
            step=1
        )

#Preparation 
# str_data="../data/raw.csv"
df=pd.read_csv(str_data)
#Set raw features used 
features=["X1","X3","X4","X6"] 
#Set target 
target=["Y"] 

#Set dataframes 
feat=df[features]
tar=df[target]

#Set splits
# n_split = 5
# n_repeats = 20
RSKF = RepeatedStratifiedKFold(n_splits=n_split, random_state=seed, n_repeats=n_repeats)

#Set hyper-parameters 
force=["X1","X6","F_w_mean"] 
nn=5 

#Make pipeline 
pipe=Pipeline([
    ("DataCreate", training.data_creator()),
    ("DataSelector", training.data_selector(force=force)),
    ("scale",StandardScaler()),
    ("KNN",KNeighborsClassifier(n_neighbors=nn))]
)

#Train 
enu_split=enumerate(list(RSKF.split(X=feat, y=tar)))
def train_once(index:int, train_index, pipe: Pipeline): 
    x_tr=feat.iloc[train_index]
    y_tr=tar.iloc[train_index]
    y_tr=np.ravel(y_tr.values)
    
    fitted=clone(pipe)
    fitted.fit(X=x_tr,y=y_tr)
    
    return index, fitted 
    
models=Parallel(n_jobs=-1,backend="loky",verbose=10)(
    delayed(train_once)(
        index=index, 
        train_index=train_index,
        pipe=pipe
    )
    for index, (train_index, _) in enu_split
)

model_dict=dict(models)

#Take input 
st.markdown(
    """## Input the wanted improvement of each of the features, the allowed interval for each of them is [0,5] and the value is expected to be an integer"""
)

with st.expander("Detail of \"improvement\""): 
    st.markdown("""
                As an example: 
                
                * Consider the situation where the original value of feature "X1" is 2, if one were to set improvement value of "X1" to 1, then "X1" will be changed in 3 with everything else kept the same. 
                
                * However, if original "X1" was 5, and the improvement value was still set to 1, then "X1" will remain 5. 
                """)

col1, col3, col4, col6 = st.columns(spec=4)

with col1: 
    x1_delta=st.number_input(
        label="Enter X1 improvement: ", 
        min_value=0, 
        max_value=5, 
        value=1,
        step=1
    )
with col3: 
    x3_delta=st.number_input(
        label="Enter X3 improvement: ", 
        min_value=0, 
        max_value=5, 
        value=0,
        step=1
    )
with col4: 
    x4_delta=st.number_input(
        label="Enter X4 improvement: ", 
        min_value=0, 
        max_value=5, 
        value=0,
        step=1
    )
with col6: 
    x6_delta=st.number_input(
        label="Enter X6 improvement: ", 
        min_value=0, 
        max_value=5, 
        value=0,
        step=1
    )

dict_deltas={
    "X1": x1_delta, 
    "X3": x3_delta, 
    "X4": x4_delta, 
    "X6": x6_delta
}

#Predict 
enu_split=enumerate(list(RSKF.split(X=feat, y=tar)))
def eva_once(index: int, test_index, in_dict_deltas: dict): 
    #Take in test features and create altered data
    x_te_o=feat.iloc[test_index].copy(deep=True)    
    x_te_i=x_te_o.copy(deep=True)
    arr_o=np.array([x_te_i[feature] for feature in features]).transpose()
    arr_delt=np.array([in_dict_deltas[feature] for feature in features])
    arr_i=arr_o+arr_delt
    arr_i=arr_i.clip(min=np.full(shape=4,fill_value=0),max=np.full(shape=4,fill_value=5))
    x_te_i=pd.DataFrame(dict(zip(x_te_i.columns, arr_i.transpose())))
    
    #Load in the right model 
    cur_pipe=model_dict[index]
    
    #Make predictions and produce expected improvement pct 
    pos_o=cur_pipe.predict(X=x_te_o).mean() 
    pos_i=cur_pipe.predict(X=x_te_i).mean()
    
    #Load in target values
    y_te_o=tar.iloc[test_index]
    y_te_o=np.ravel(y_te_o.values)
    pos_acc=y_te_o.mean()
    
    return (pos_i-pos_o)/pos_o, (pos_i-pos_acc)/pos_acc

results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
    delayed(eva_once)(
        index=index, 
        test_index=test_index,
        in_dict_deltas=dict_deltas
    )
    for index, (_, test_index) in enu_split 
)

imp_pct=np.array(results)

#Record 
df_whatif=pd.DataFrame(dict(zip(["rel pred", "rel truth"],imp_pct.transpose())))

#Graph
fig, axs = plt.subplots(1,2, figsize=(16,6))
#Relative to predicted happiness rate 
rp_mean=df_whatif["rel pred"].to_numpy().mean()
rp_std=df_whatif["rel pred"].to_numpy().std()
axs[0].hist(df_whatif["rel pred"])
axs[0].set_xlabel("Predicted improvement rate"+f"\n mean: {rp_mean:,.4f}; std: {rp_std:,.4f}", fontsize=12)
axs[0].set_title("Improvement relative to predicted happiness rate")

#Relative to recored happiness rate
th_mean=df_whatif["rel truth"].to_numpy().mean()
th_std=df_whatif["rel truth"].to_numpy().std()
axs[1].hist(df_whatif["rel truth"])
axs[1].set_xlabel("Predicted improvement rate"+f"\n mean: {th_mean:,.4f}; std: {th_std:,.4f}", fontsize=12)
axs[1].set_title("Improvement relative to recorded happiness rate")

st.pyplot(fig)

#Suggest Business Plan
st.markdown("""
            ## Predicting Sentiment according to "improvement budget" allotted 
            
            * "improvement budget" indicates the total amount of improvement to be spent on the set of raw features. For instance: If one were to set it to 2, then we may spend it on improving each of X1, X6 once, or improve X1 twice, and so on. This value is expected to be an integer larger or equal to 1. 
            """)

def comp_k(sum:int, k: int=len(features)): 
    for cuts in itertools.combinations(range(sum+k-1), k-1): 
        prev=-1 #First "cut"
        parts=[]
        #Each part should be current cut - previous cut -1 
        for cut in cuts + (sum+k-1,): #Last "cut" is always the very last entry 
            parts.append(cut-prev-1)
            prev=cut 
        yield np.array(parts)
        
# col_bgt, col_by = st.columns(2)

# with col_bgt:
imp_bgt=st.number_input(
        label="Improvement budget:", 
        value=1,
        min_value=1,
        step=1
)

df_bgt_results=pd.DataFrame(dict(zip(features, [[] for _ in features ])))
df_bgt_results["mean_rel_pred"], df_bgt_results["std_rel_pred"], df_bgt_results["mean_rel_act"], df_bgt_results["std_rel_act"]=[],[],[],[]

for lnr_comb in comp_k(sum=imp_bgt): 
    cur_dict_delta=dict(zip(features, lnr_comb))
    enu_split=enumerate(list(RSKF.split(X=feat, y=tar)))
    # print(cur_dict_delta)
    results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(eva_once)(
            index=index, 
            test_index=test_index,
            in_dict_deltas=cur_dict_delta
        )
        for index, (_, test_index) in enu_split 
    )
    imp_pct=np.array(results).transpose()
    # print(imp_pct)
    df_cur_results=pd.DataFrame(dict(zip(features, lnr_comb.reshape(-1,1))))
    df_cur_results["mean_rel_pred"], df_cur_results["std_rel_pred"]=imp_pct[0].mean(), imp_pct[0].std()
    df_cur_results["mean_rel_act"], df_cur_results["std_rel_act"]=imp_pct[1].mean(), imp_pct[1].std()
    df_bgt_results=pd.concat([df_bgt_results,df_cur_results])
    
st.markdown("""
            ### Predicted impact: 
            
            * X\{n\}: The improvement on this feature. 
            * mean_rel_pred: mean of the improvement relative to prediction. 
            * mean_rel_act: mean of the improvement relative to actual record. 
            * std_rel_pred: std of the improvement relative to prediction. 
            * std_rel_act: std of the improvement relative to actual record. 
            """)

options=["relative to prediction","relative to actual record"]
mapping={
    "relative to prediction": "mean_rel_pred","relative to actual record": "mean_rel_act"
}

# with col_by: 
choice=st.selectbox(
        label="Choose the order by:", 
        options=["relative to prediction","relative to actual record"], 
        index=1
)
    
order_by=mapping[choice]

df_show=df_bgt_results.sort_values(by=[order_by],ascending=False)

st.dataframe(data=df_show, use_container_width=True) 

df_best=df_show.iloc[[0]]

x1_imp, x3_imp, x4_imp, x6_imp=df_best.loc[0,"X1"].astype(int), df_best.loc[0,"X3"].astype(int), df_best.loc[0,"X4"].astype(int), df_best.loc[0,"X6"].astype(int)

sent_imp=df_best[order_by]

st.markdown("""
            ### Business Suggestion 
            """)

st.write(
    f"Given that the allotted improvement budget is {imp_bgt}, the model suggests that the budget to be spent with following linear combination: \n {x1_imp} on X1, {x3_imp} on X3, {x4_imp} on X4, {x6_imp}* on X6. \n This allocation has expected customer sentiment improvement of {sent_imp} {choice}. "
    )
