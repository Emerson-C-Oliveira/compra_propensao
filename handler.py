import pickle
import pandas as pd
from flask import Flask,request,Response
from healthinsurance import HealthInsurance

model = pickle.load(open('rf_model_bp.pkl','rb'))

app = Flask(__name__)
@app.route('/healthinsurance/predict',methods=['POST'])

def insurance_all_predict():
    test_json = request.get_json()

    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate Rossmann class
        pipeline = HealthInsurance()
        
        # data cleaning
        df1 = pipeline.data_cleanning( test_raw )
        
        # feature engineering
        df2 = pipeline.feature_engieneering( df1 )
        
        # data preparation
        df3 = pipeline.data_preparation( df2 )
        
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response
    
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    app.run('0.0.0.0')