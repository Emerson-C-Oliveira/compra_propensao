import json
import pickle
import pandas as pd
import numpy as np
import inflection
import os 

from sklearn.ensemble     import RandomForestClassifier
from sklearn              import preprocessing   as pp
from lightgbm             import LGBMClassifier
from sklearn              import metrics         as m
from category_encoders    import OneHotEncoder

class HealthInsurance (object):
    
    def __init__( self ):
        # Use o caminho relativo para o diretório do script
        script_dir = os.path.dirname(__file__)  # Obtém o diretório do script atual
        self.home_path = script_dir  # Define o diretório como home_path

        # Use caminhos relativos para carregar os arquivos pickle
        self.gender_label_encoder = pickle.load(open (os.path.join(script_dir, 'parameter/gender_label_encoder.pkl'), 'rb'))
        self.vehicle_damage_label_encoder = pickle.load(open('parameter/vehicle_damage_label_encoder.pkl','rb'))
        self.mms_age = pickle.load(open(os.path.join(script_dir, 'parameter/mms_age_scaler.pkl'), 'rb'))
        self.mms_annual_premium = pickle.load(open(os.path.join(script_dir, 'parameter/annual_premium_scaler.pkl'), 'rb'))
        self.mms_vintage = pickle.load(open(os.path.join(script_dir, 'parameter/vintage_scaler.pkl'), 'rb'))
        self.target_encode_region_code = pickle.load(open(os.path.join(script_dir, 'parameter/target_encode_region_code.pkl'), 'rb'))
        self.fe_policy_sales_channel = pickle.load(open(os.path.join(script_dir, 'parameter/fe_policy_sales_channel_scaler.pkl'), 'rb'))
    
    def data_cleanning(self,df1):
         # Renomeie as colunas para snake_case
        df1.columns = [inflection.underscore(col) for col in df1.columns]
        return df1
                      
    def feature_engieneering(self,df2):
        df2['risk_age']= df2['age'].apply(lambda x: 0 if x>25 else 1)    
        
        df2['age_insured'] = ((df2['age'] >= 32) & (df2['age'] <= 52) & (df2['previously_insured'] == 0)).astype(int)

        return df2
    
    def data_preparation(self,df5):
        # # Carregue os LabelEncoders salvos
        # gender_label_encoder = pickle.load(open('../propensity_score/parameter/gender_label_encoder.pkl', 'rb'))
        # vehicle_damage_label_encoder = pickle.load(open('../propensity_score/parameter/vehicle_damage_label_encoder.pkl', 'rb'))
        # # Carregue os MinMaxScalers salvos
        # mms_age = pickle.load(open('../propensity_score/parameter/mms_age_scaler.pkl', 'rb'))
        # mms_annual_premium = pickle.load(open('../propensity_score/parameter/annual_premium_scaler.pkl', 'rb'))
        # mms_vintage = pickle.load(open('../propensity_score/parameter/vintage_scaler.pkl', 'rb'))
        # # Carregue o dicionário de frequency encoding para 'policy_sales_channel'
        # fe_policy_sales_channel = pickle.load(open('../propensity_score/parameter/fe_policy_sales_channel_scaler.pkl', 'rb'))
        # # Carregue o dicionário de target encoding para 'region_code'
        # target_encode_region_code = pickle.load(open('../propensity_score/parameter/target_encode_region_code.pkl', 'rb'))

        # Aplique o Label Encoding às colunas apropriadas
        df5['gender'] = self.gender_label_encoder.transform(df5['gender'])
        df5['vehicle_damage'] = self.vehicle_damage_label_encoder.transform(df5['vehicle_damage'])

        # Aplique o MinMaxScaler às colunas apropriadas
        df5['age'] = self.mms_age.transform(df5[['age']])
        df5['annual_premium'] = self.mms_annual_premium.transform(df5[['annual_premium']])
        df5['vintage'] = self.mms_vintage.transform(df5[['vintage']])
        
        # Mapeie os valores em 'policy_sales_channel' com base nas frequências relativas
        df5['policy_sales_channel'] = df5['policy_sales_channel'].map(self.fe_policy_sales_channel)

        # Mapeie os valores em 'region_code' com base nas taxas de resposta média
        df5['region_code'] = df5['region_code'].map(self.target_encode_region_code)

        # Crie as colunas one-hot para 'vehicle_age'
        df5['below_1_year'] = df5['vehicle_age'].apply(lambda x: 1 if x == '< 1 Year' else 0)
        df5['between_1_2_year'] = df5['vehicle_age'].apply(lambda x: 1 if x == '1-2 Year' else 0)
        df5['over_2_years'] = df5['vehicle_age'].apply(lambda x: 1 if x == '> 2 Years' else 0)
        df5.drop('vehicle_age', axis=1, inplace=True)

        df5 = df5.fillna(0)

        # Feature Selection
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage','policy_sales_channel',  'previously_insured', 'age_insured']
        
        return df5[ cols_selected ]

    def get_prediction(self, model, original_data, test_data):
        # model prediction
        pred = model.predict_proba(test_data)
        
        # Extract the probabilities of the positive class (column 1)
        positive_class_probabilities = pred[:, 1]
        
        # Create a new column 'prediction' in the original data
        original_data['prediction'] = positive_class_probabilities
        
        return original_data.to_json(orient='records', date_format='iso')