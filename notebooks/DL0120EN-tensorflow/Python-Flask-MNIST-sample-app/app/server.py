from watson_machine_learning_client import WatsonMachineLearningAPIClient
import math
import PIL
from PIL import Image
import numpy as np
from flask import Flask, request, json, jsonify
import os


# 1.  Fill in wml_credentials.

wml_credentials = {
  "apikey": "pIFaFqASWOGHoAtWAZIaSYfS_xEhAYTEtUJnAxMEDqlD",
  "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance - crn:v1:bluemix:public:pm-20:us-south:a/20cf85337b99434a95ba2b141886466e:91a35b53-d3a3-48b1-9eba-0922f333fa20::",
  "iam_apikey_name": "auto-generated-apikey-aaf8a72e-b930-46d0-b1a9-6b405a9f9724",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/20cf85337b99434a95ba2b141886466e::serviceid:ServiceId-c3f4527b-e310-4fe6-972e-43e1f4cf5d22",
  "instance_id": "91a35b53-d3a3-48b1-9eba-0922f333fa20",
  "url": "https://us-south.ml.cloud.ibm.com"
}

# vcap = json.loads( os.getenv( "VCAP_SERVICES" ) )
# wml_credentials = vcap["pm-20"][0]["credentials"] 


client = WatsonMachineLearningAPIClient( wml_credentials )

#
# 2.  Fill in one or both of these:
#     - model_deployment_endpoint_url
#     - function_deployment_endpoint_url
#
model_deployment_endpoint_url    = "https://us-south.ml.cloud.ibm.com/v3/wml_instances/91a35b53-d3a3-48b1-9eba-0922f333fa20/published_models/1d7d2304-0087-4b7e-803e-5113365bb256/deployments/0b7e5589-f03e-4ca7-a8ac-633185e64522/online";
function_deployment_endpoint_url = "";

def createPayload( canvas_data ):
    dimension      = canvas_data["height"]
    img            = Image.fromarray( np.asarray( canvas_data["data"] ).astype('uint8').reshape( dimension, dimension, 4 ), 'RGBA' )
    sm_img         = img.resize( ( 28, 28 ), Image.LANCZOS )
    alpha_arr      = np.array( sm_img.split()[-1] )
    norm_alpha_arr = alpha_arr / 255
    payload_arr    = norm_alpha_arr.reshape( 1, 784 )
    payload_list   = payload_arr.tolist()
    return { "values" : payload_list }


app = Flask( __name__, static_url_path='' )

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int( os.getenv( 'PORT', 8000 ) )

@app.route('/')
def root():
    return app.send_static_file( 'index.html' )

@app.route( '/sendtomodel', methods=['POST'] )
def sendtomodel():
    try:
        print( "sendtomodel..." )
        if model_deployment_endpoint_url:
            canvas_data = request.get_json()
            payload = canvas_data
            result = client.deployments.score( model_deployment_endpoint_url, payload )
            print( "result: " + json.dumps( result, indent=3 ) )
            return jsonify( result )
        else:
            err = "Model endpoint URL not set in 'server.py'"
            print( "\n\nError:\n" + err )
            return jsonify( { "error" : err } )
    except Exception as e:
        print( "\n\nError:\n" + str( e ) )
        return jsonify( { "error" : str( e ) } )
    
@app.route( '/sendtofunction', methods=['POST'] )
def sendtofunction():
    try:
        print( "sendtofunction..." )
        if function_deployment_endpoint_url:
            canvas_data = request.get_json()
            payload = canvas_data
            result = client.deployments.score( function_deployment_endpoint_url, payload )
            print( "result: " + json.dumps( result, indent=3 ) )
            return jsonify( result )
        else:
            err = "Function endpoint URL not set in 'server.py'"
            print( "\n\nError:\n" + err )
            return jsonify( { "error" : err } )
    except Exception as e:
        print( "\n\nError:\n" + str( e ) )
        return jsonify( { "error" : str( e ) } )

@app.route( '/sendtowebserver', methods=['POST'] )
def sendtowebserver():
    try:
        print( "sendtowebserver..." )
        if model_deployment_endpoint_url:
            canvas_data = request.get_json()
            payload = createPayload( canvas_data )
            result = client.deployments.score( model_deployment_endpoint_url, payload )
            print( "result: " + json.dumps( result, indent=3 ) )
            return jsonify( result )
        else:
            err = "Model endpoint URL not set in 'server.py'"
            print( "\n\nError:\n" + err )
            return jsonify( { "error" : err } )
    except Exception as e:
        print( "\n\nError:\n" + str( e ) )
        return jsonify( { "error" : str( e ) } )

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=port, debug=True)
