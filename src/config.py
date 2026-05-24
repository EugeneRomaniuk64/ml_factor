import torch

# Data Parameters

DATA_START = 199811
DATA_END = 201903
TRAIN_END = 201601
TEST_START = 201602

FEATURES = [
    'Advt_12M_Usd', 'Advt_3M_Usd', 'Advt_6M_Usd',
    'Asset_Turnover', 'Bb_Yld', 'Bv', 'Capex_Ps_Cf', 'Capex_Sales',
    'Cash_Div_Cf', 'Cash_Per_Share', 'Cf_Sales', 'Debtequity', 'Div_Yld',
    'Dps', 'Ebit_Bv', 'Ebit_Noa', 'Ebit_Oa', 'Ebit_Ta', 'Ebitda_Margin',
    'Eps', 'Eps_Basic', 'Eps_Basic_Gr', 'Eps_Contin_Oper', 'Eps_Dil', 'Ev',
    'Ev_Ebitda', 'Fa_Ci', 'Fcf', 'Fcf_Bv', 'Fcf_Ce', 'Fcf_Margin',
    'Fcf_Noa', 'Fcf_Oa', 'Fcf_Ta', 'Fcf_Tbv', 'Fcf_Toa', 'Fcf_Yld',
    'Free_Ps_Cf', 'Int_Rev', 'Interest_Expense', 'Mkt_Cap_12M_Usd',
    'Mkt_Cap_3M_Usd', 'Mkt_Cap_6M_Usd', 'Mom_11M_Usd', 'Mom_5M_Usd',
    'Mom_Sharp_11M_Usd', 'Mom_Sharp_5M_Usd', 'Nd_Ebitda', 'Net_Debt',
    'Net_Debt_Cf', 'Net_Margin', 'Netdebtyield', 'Ni', 'Ni_Avail_Margin',
    'Ni_Oa', 'Ni_Toa', 'Noa', 'Oa', 'Ocf', 'Ocf_Bv', 'Ocf_Ce', 'Ocf_Margin',
    'Ocf_Noa', 'Ocf_Oa', 'Ocf_Ta', 'Ocf_Tbv', 'Ocf_Toa', 'Op_Margin',
    'Op_Prt_Margin', 'Oper_Ps_Net_Cf', 'Pb', 'Pe', 'Ptx_Mgn',
    'Recurring_Earning_Total_Assets', 'Return_On_Capital', 'Rev', 'Roa',
    'Roc', 'Roce', 'Roe', 'Sales_Ps', 'Share_Turn_12M', 'Share_Turn_3M',
    'Share_Turn_6M', 'Ta', 'Tev_Less_Mktcap', 'Tot_Debt_Rev',
    'Total_Capital', 'Total_Debt', 'Total_Debt_Capital',
    'Total_Liabilities_Total_Assets', 'Vol1Y_Usd', 'Vol3Y_Usd'
]

# Purged Walk-Ahead Cross-Validation Parameters

CV_MIN_TRAIN_MONTHS = 72
CV_VAL_MONTHS = 12
CV_EMBARGO_MONTHS = 1

# Model Hyperparameters
LINEAR_MODEL_NAMES = ['ridge', 'lasso', 'elasticnet']

RIDGE_PARAMS = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

LASSO_PARAMS = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
}

ELASTICNET_PARAMS = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.2, 0.5, 0.8]
}


RF_PARAMS = {
    'n_estimators': [100, 300, 500],      
    'max_depth': [3, 5, 7],            
    'max_features': [0.3, 0.5, 'sqrt'],   
    'min_samples_leaf': [10, 50, 100]      
}

XGB_PARAMS = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.5, 0.8],
    'n_estimators': [300, 500],
    'colsample_bytree': [0.5, 0.8]
}

MLP_INPUT_SIZE = len(FEATURES)
MLP_NUM_EPOCHS = 100
MLP_PATIENCE = 10
MLP_STAGE1_LR = 0.001
MLP_STAGE1_BATCH_SIZE = 512

MLP_PARAMS_STAGE1 = {
    'hidden_sizes': [
        (32, 16, 8),
        (64, 32, 16),
        (128, 64, 32),
        (256, 128, 64),
        (32, 16, 8, 4),
        (64, 32, 16, 8)
        (128, 64, 32, 16),
        (256, 128, 64, 32)
    ],
    'dropout': [0.2, 0.3, 0.5]
}

MLP_PARAMS_STAGE2 = {
    'learning_rate': [0.001, 0.0001],
    'batch_size': [512, 1024] 
}

DEVICE = torch.device('cuda')
