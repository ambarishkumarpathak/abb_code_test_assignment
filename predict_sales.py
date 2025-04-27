import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

sns.set()
plt.rcParams['figure.figsize'] = (15, 5)
warnings.filterwarnings("ignore")



class PredictSales(object):

    def __init__(self, q):
        self.q = q
        pass

    def change_dtype(self, data):
        data['Outlet_Establishment_Year'] = data['Outlet_Establishment_Year'].astype(str)

        return data


    def fill_missing_values(self, data):
        col_type = [str(el) for el in data.dtypes.tolist()]
        str_cols = [el[0] for el in zip(data.columns, col_type) if el[1]=='object']

        float_cols = [el[0] for el in zip(data.columns, col_type) if 'float' in el[1]]

        for col in str_cols:
            data[col].fillna("unknown", inplace=True)

        for col in float_cols:
            if any(data[col].isna().to_list()) and col != 'Item_Outlet_Sales':
                na_fill_df = data.copy()

                na_fill_df = na_fill_df.drop(columns=['Item_Identifier', 'Outlet_Identifier'], axis=1)
    
                na_fill_df = pd.get_dummies(na_fill_df,
                                             columns=[el for el in str_cols if el not in ['Item_Identifier', 'Outlet_Identifier']])
    
                non_na_rows = na_fill_df.loc[~na_fill_df[col].isna()]
                na_rows = na_fill_df.loc[na_fill_df[col].isna()]
    
                non_na_rows = non_na_rows.drop('Item_Outlet_Sales', axis=1)
                na_rows = na_rows.drop('Item_Outlet_Sales', axis=1)
                
                vals = DecisionTreeRegressor(min_samples_split=60).fit(non_na_rows.drop(col,
                                                                                        axis=1),
                                                                       non_na_rows[col]
                                                                      ).predict(na_rows.drop(col, axis=1))

                non_na_rows = data.loc[~data[col].isna()]
                na_rows = data.loc[data[col].isna()]
                na_rows[col] = vals

                data = pd.concat([non_na_rows, na_rows])
                
                # data[col].fillna(data[col].mean(), inplace=True)

        return data

    def derive_features(self, data):

        for i in range(4):
            col_name = 'II_' + str(i)
            data[col_name] = data.Item_Identifier.apply(lambda x: str(x[i:]) if i == 3 else str(x[i]))
        
        col_type = [str(el) for el in data.dtypes.tolist()]
        float_cols = [el[0] for el in zip(data.columns, col_type) if 'float' in el[1]]
        
        data['corrected_Item_Fat_Content'] = data['Item_Fat_Content']
        data['corrected_Item_Fat_Content'] = np.where(data.Item_Fat_Content == 'low fat', "Low Fat",
                                                            data.Item_Fat_Content)
        data['corrected_Item_Fat_Content'] = np.where(data.Item_Fat_Content == 'LF', "Low Fat",
                                                            data.corrected_Item_Fat_Content)
        data['corrected_Item_Fat_Content'] = np.where(data.Item_Fat_Content == 'reg', "Regular",
                                                            data.corrected_Item_Fat_Content)

        del data['Item_Fat_Content']
        
        for col in float_cols:
            if col != 'Item_Outlet_Sales':
                class_name = col+"_bucket"
                
                data[class_name] = data.groupby('Item_Type')[col].transform(lambda x: pd.qcut(x, q=self.q, labels=range(1, self.q+1),
                                                                                                        duplicates='drop'))
                data[class_name] = data[class_name].astype(str)

        return data


    @staticmethod
    def q25(x):
        return x.quantile(0.5)
    
    @staticmethod
    def q75(x):
        return x.quantile(0.9)

    def treat_tgt_outliers(self, data):
        quantiles = data.groupby(['Item_Weight_bucket', 'Item_Type']).agg(first_q=('Item_Outlet_Sales', self.q25),
                                                                                third_q=('Item_Outlet_Sales', self.q75)
                                                                               ).reset_index()
        
        quantiles['ub'] = 2 * (quantiles['third_q'] - quantiles['first_q']) + quantiles['third_q']
        quantiles['lb'] = quantiles['first_q'] - 2 * (quantiles['third_q'] - quantiles['first_q'])

        df = data.merge(quantiles, on=['Item_Weight_bucket', 'Item_Type'], how='inner')
        df['Item_Outlet_Sales'] = np.where(df.Item_Outlet_Sales > df.ub, df.ub, df.Item_Outlet_Sales)
        df['Item_Outlet_Sales'] = np.where(df.Item_Outlet_Sales < df.lb, df.lb, df.Item_Outlet_Sales)

        df = df.drop(columns=['first_q', 'third_q', 'ub', 'lb'])
        
        return df
        
        

    def get_opt_base_model(self, data):
        
        train_data = data.loc[data.flag=='train']
        test_data = data.loc[data.flag=='test']

        del train_data['flag']
        del test_data['flag']

        print(train_data.shape, test_data.shape)
        
        models = {'dt': DecisionTreeRegressor(min_samples_split=60, random_state=42),
                  'rf': RandomForestRegressor(min_samples_split=60, random_state=42),
                  'ab': AdaBoostRegressor(n_estimators=100, random_state=42),
                  'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                  'xgb': XGBRegressor(n_estimators=100, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, enable_categorical=True)}

        errs_train = []
        errs_test = []
        errs_all = []
        
       
        train_data = train_data.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        test_data_id = test_data[['Item_Identifier', 'Outlet_Identifier']]

        test_data = test_data.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
        
        col_type = [str(el) for el in train_data.dtypes.tolist()]
        cat_cols = [el[0] for el in zip(train_data.columns, col_type) if el[1] in ['object', 'category']]
        
        train_data['Outlet_Establishment_Year'] = train_data['Outlet_Establishment_Year'].astype(str)
        test_data['Outlet_Establishment_Year'] = test_data['Outlet_Establishment_Year'].astype(str)
        
        cat_cols = cat_cols + ["Outlet_Establishment_Year"]
        
        
        train_data = pd.get_dummies(train_data, columns=cat_cols)
        x_test = pd.get_dummies(test_data, columns=cat_cols)
        
        x_test = x_test.drop("Item_Outlet_Sales", axis=1)
        
        x, y = train_data.drop("Item_Outlet_Sales", axis=1), train_data["Item_Outlet_Sales"]
        
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15, random_state=42)
        # print(train_x.columns)
        
        for k in models.keys():
            # print(k)
            model = models[k]
            
            model.fit(train_x, train_y)
        
            pred_tr = model.predict(train_x)
            pred = model.predict(test_x)
            pred_all = model.predict(x)
        
            err_te = root_mean_squared_error(test_y, pred)
            err_tr = root_mean_squared_error(train_y, pred_tr)
            err_all = root_mean_squared_error(y, pred_all)
        
            errs_train.append(err_tr)
            errs_test.append(err_te)
            errs_all.append(err_all)

        opt_model = list(models.values())[errs_test.index(min(errs_test))]
        print(models.keys(), errs_test)

        # print("opt_model", opt_model)
        test_data_id['Item_Outlet_Sales'] = np.abs(opt_model.fit(x, y).predict(x_test))

        return opt_model, test_data_id, (test_y, pred), (train_y, pred_tr), (y, pred_all), (x, y), x_test

    def fine_tune_opt_model(self, x, y, x_test, opt_model):
        model = opt_model
        grid = dict() 
        grid['n_estimators'] = [100, 150, 200]
        grid['learning_rate'] = [0.01, 0.03, 0.05, 0.07, 0.1]
        grid['subsample'] = [0.7, 0.8, 0.9, 1.0]
        grid['max_depth'] = [2, 3, 7, 9]

        # define the evaluation procedure
        cv = 5
        # define the grid search procedure
        reg = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv)
        # execute the grid search
        reg.fit(x, y)

        pred = reg.predict(x_test)
        
        return pred

    def predict_sales_amt(self, train_data, test_data):

        train_data['flag'] = 'train'
        test_data['flag'] = 'test'
        data = pd.concat([train_data, test_data])
        
        data = self.fill_missing_values(data)
        data = self.derive_features(data)
        data = self.treat_tgt_outliers(data)

        opt_model, output, (test_y, pred), (train_y, pred_tr), (y, pred_all), (x, y), x_test = self.get_opt_base_model(data)

        # finetuned_model_result = self.fine_tune_opt_model(x, y, x_test, opt_model)
        # output['Item_Outlet_Sales'] = finetuned_model_result
        output.to_csv(f"first_output_{self.q}_15_pct_test_encoded_outliers_treated_no_fine_tune.csv", index=False)

        return output, (test_y, pred), (train_y, pred_tr), (y, pred_all)


        
        

