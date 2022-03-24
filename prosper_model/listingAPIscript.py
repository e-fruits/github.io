import requests
import pandas as pd
import json
import config

url = "https://api.prosper.com/v1/security/oauth/token"

headers = {'accept': "application/json", 'content-type': "application/x-www-form-urlencoded"}
response = requests.request("POST"
                            , url
                            , data="grant_type=password&client_id="+config.client_id+"&client_secret="+config.secret_id+"&username="+config.user+"&password="+config.pw
                            , headers=headers)
print(response.text)

tokens = json.loads(response.text)
access_token = tokens['access_token']
#print(access_token)

query = {'biddable':'true',  'limit':'500', 'include_credit_bureau_values':'transunion'}
headers = { 'accept': "application/json", 'authorization': "bearer "+ access_token}
response = requests.get('https://api.prosper.com/listingsvc/v2/listings/', params=query, headers=headers)
#print(response.json())

content = response.content.decode('utf-8') # list of ugly strings
j = json.loads(content)
#j = j[0]

df = pd.json_normalize(j['result'])

#print(df.head(5))
