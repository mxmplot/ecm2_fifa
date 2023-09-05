import numpy as np
import pandas as pd
from linearmodels import RandomEffects, PanelOLS
from linearmodels.panel import compare
from scipy import stats
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tools import add_constant
import numpy.linalg as la


def read_data(group_name):
    file_path = f"new/{group_name}/{group_name}_group_all_seasons_dummy_updated.csv"
    return pd.read_csv(file_path, index_col=['id', 'Season'])


group_names = ['red', 'yellow', 'green']
regressors = {
    'red': ['Gls', 'SoT%', 'Cmp%', 'Ast', 'Tkl%', 'Clr', 'Att.5', 'Succ%', 'PPM', 'CrdY', 'CrdR',
            'Won%', 'def', 'forw', 'mid'],
    'yellow': ['xG', 'Blocks', 'Tkl+Int', 'Err', 'Def Pen', 'Att.5', 'Succ%', 'Mn/Sub', 'xG+/-90', 'OG', 'Won%',
               'def', 'mid', 'forw'],
    'green': ['Sh/90', 'np:G-xG', 'Dist', 'npxG/Sh', 'Cmp%', 'A-xAG', 'Crs', 'TO', 'Tkl%', 'Tkl+Int', 'Err',
              'Def Pen', 'Succ%', '44986', 'onGA', 'xG+/-90', 'CrdY', 'CrdR', 'Won%', 'def', 'mid', 'forw']
}

def build_fixed_effects_regression(df, target, features):
    X = add_constant(df[features])
    y = df[target]
    fixed_effects_mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    fixed_effects_res = fixed_effects_mod.fit()
    return fixed_effects_res

def build_random_effects_regression(df, target, features):
    X = add_constant(df[features])
    y = df[target]
    random_effects_mod = RandomEffects(y, X, check_rank=False)
    random_effects_res = random_effects_mod.fit()
    return random_effects_res

def read_new_data(group_name):
    file_path = f"new/{group_name}/{group_name}_group_all_seasons_dummy_for_prediction.csv"
    return pd.read_csv(file_path, index_col=['id', 'Season'])

def get_prediction(res, df, features):
    X = add_constant(df[features])
    predictions = res.predict(X)
    return predictions

def mae_and_mse(pred, df):
    y = df['RATING']
    merg = pd.merge(y, pred, on='id')
    merg['diff'] = merg.apply(lambda row: (1 if (3 > row['RATING'] - row['predictions'] > -3) else 0), axis=1)
    print(str(merg['diff'].sum()/616 * 100) + '%')
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    return mae, mse


for group_name in group_names:
    df = read_data(group_name)
    df_pred = read_new_data(group_name)
    fixed_effects_res = build_fixed_effects_regression(df, 'RATING', regressors[group_name])
    random_effects_res = build_random_effects_regression(df, 'RATING', regressors[group_name])

    print(f"\nFixed Effects Regression Results for {group_name} group:")
    print(fixed_effects_res)

    print(f"\nRandom Effects Regression Results for {group_name} group:")
    print(random_effects_res)

    print(compare({'FE': fixed_effects_res, 'RE': random_effects_res}))

    print(f"\nPrediction for {group_name} group:")
    pred = get_prediction(random_effects_res, df_pred, regressors[group_name])
    print(pred)
    mae, mse = mae_and_mse(pred, df_pred)
    print(mae)
    print(mse)
