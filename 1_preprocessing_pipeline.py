# -*- coding: utf-8 -*-
"""1. Preprocessing Pipeline.ipynb

"""

# pre-process pipeline for all datasets. comment out test-train split and take all data as train for final model building.

import pandas as pd
import scipy.stats as stats
#from ua_parser import user_agent_parser
import pprint
import numpy as np
from ua_parser.user_agent_parser import Parse
from category_encoders import MEstimateEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv(r"C:\Users\UserAdmin\Desktop\shraddha\Doceree-Complete-DataSet\Doceree_Complete_DataSet\Doceree-HCP_Train.csv",encoding = 'latin-1')
data_test_final = pd.read_csv(r"C:\Users\UserAdmin\Desktop\shraddha\Doceree-Complete-DataSet\Doceree_Complete_DataSet\Doceree-HCP_Test.csv",encoding = 'latin-1')

print(data.shape)
print(data_test_final.shape)

"""#### 1. Updating target variable"""

# updating target variable - is_h
data['USERPLATFORMUID'] = data['USERPLATFORMUID'].fillna('Unidentified')
data = data[~data.IS_HCP.isna()]
data1 = pd.merge(data,data.groupby(['USERPLATFORMUID'],as_index = False).agg(is_h = ('IS_HCP','max')),on = 'USERPLATFORMUID',how = 'left')
data1['is_h'] = np.where(data['USERPLATFORMUID']=='Unidentified',data1['IS_HCP'],data1['is_h'])

tax_codes = pd.read_csv(r"nucc_taxonomy_230.csv",encoding = 'latin-1')

data1 = pd.merge(data1,tax_codes[['Code','Classification']],how = 'left',left_on = 'TAXONOMY',right_on = 'Code')

a = data1.groupby('USERPLATFORMUID')['Classification'].agg(pd.Series.mode).reset_index(drop = False)
a.columns = ['USERPLATFORMUID','TAX1']
a

data1 = pd.merge(data1,a,on = 'USERPLATFORMUID',how = 'left')
data1

data1['TAX1'] = data1['TAX1'].astype(str)

# setting TAXONOMY mode of TaXONOMy for each userplatformuid
data1['TAX1'] = np.where(data['USERPLATFORMUID']=='Unidentified',data1['Classification'],data1['TAX1'])

data1.TAX1 = data1.TAX1.apply(lambda x:x.replace('[]',''))
data1['TAX1']=np.where(data1['TAX1']=='',None,data1['TAX1'])

data1['TAX1'].value_counts()

data1['TAXONOMY_Pred'] = np.where((data1['is_h']==0),'Not Doctor',np.where((data1['is_h']==1)&(data1.TAXONOMY.isna()),'Unknown',data1['Classification']))

data1['TAXONOMY_Pred'].isna().sum()

x = pd.DataFrame(data1['TAXONOMY_Pred'].value_counts())
x = x[x.TAXONOMY_Pred<10].index
data1['TAXONOMY_Pred'] = np.where(data1['TAXONOMY_Pred'].isin(x),'Others',data1['TAXONOMY_Pred'])

data1.TAXONOMY_Pred.nunique()

"""#### 2. Creating new features

###### 2.1 IP
"""

def find_class_ip(ip1):
    if ip1>=1 and ip1<=127:
        return 'A'
    elif ip1>=128 and ip1<=191:
        return 'B'
    elif ip1>=192 and ip1<=223:
        return 'C'
    elif ip1>=224 and ip1<=239:
        return 'D'
    elif ip1>=240 and ip1<=255:
        return 'E'
    else:
        return 'I'

# creating IP Class - train
data1['IP1']=data1['BIDREQUESTIP'].apply(lambda x:int(x.split('.')[0]))
data1['IP2']=data1['BIDREQUESTIP'].apply(lambda x:int(x.split('.')[1]))
data1['IP3']=data1['BIDREQUESTIP'].apply(lambda x:int(x.split('.')[2]))
data1['IP4']=data1['BIDREQUESTIP'].apply(lambda x:int(x.split('.')[3]))
data1['IP_class'] = data1['IP1'].apply(find_class_ip)

# creating IP Class- - final-test
data_test_final['IP1']=data_test_final['BIDREQUESTIP'].apply(lambda x:int(x.split('.')[0]))
data_test_final['IP2']=data_test_final['BIDREQUESTIP'].apply(lambda x:int(x.split('.')[1]))
data_test_final['IP3']=data_test_final['BIDREQUESTIP'].apply(lambda x:int(x.split('.')[2]))
data_test_final['IP4']=data_test_final['BIDREQUESTIP'].apply(lambda x:int(x.split('.')[3]))
data_test_final['IP_class'] = data_test_final['IP1'].apply(find_class_ip)

# def find_public_private_networks(row):
#     ip1 = row[0]
#     ip2 = row[1]
#     if ((ip1==10) | (ip1==169 & ip2 ==254) | (ip1 == 172 &(ip2>=16 & ip2<=31)) | (ip1==192 & ip2 ==168)):
#         return 'Private'
#     else:
#         return 'Public'

# #data1['IP_Public_Priv'] = data1[['IP1','IP2']].apply(lambda _row: find_public_private_networks(_row[0], _row[1]))
# data1['IP_Public_Priv'] = data1[['IP1','IP2']].apply(find_public_private_networks,axis=1)
# data1['IP_Public_Priv'].value_counts()

"""###### 2.2 User agents"""

def ua_par(ua):
    if ua==None:
        return None,None,None,None,None
    else:
        try:
            dic = Parse(ua)
            return dic['device']['brand'],dic['device']['family'],dic['device']['model'], dic['os']['family'],dic['user_agent']['family']
        except:
            return None,None,None,None,None

data1['USERAGENT']=data1['USERAGENT'].fillna(' ')
data1['device_brand'], data1['device_family'], data1['device_model'], data1['os'], data1['ua'] =   zip(*data1['USERAGENT'].map(ua_par))

data_test_final['USERAGENT']=data_test_final['USERAGENT'].fillna(' ')
data_test_final['device_brand'], data_test_final['device_family'], data_test_final['device_model'], data_test_final['os'], data_test_final['ua'] =   zip(*data_test_final['USERAGENT'].map(ua_par))

"""###### 2.3 Website URLs"""

from urllib.parse import urlparse

def get_url_features(url):
    urlp = urlparse(url)
    le = len(url)
    sch = urlp.scheme
    dig = len([i for i in url if i.isdigit()])
    is_query = urlp.query !=''
    is_param = urlp.params !=''
    is_frag = urlp.fragment !=''
    host_len = len(urlp.netloc)
    no_of_prd = len([i for i in url if i == '.'])
    no_of_subd = len(urlp.path.split('/'))
    base_path = urlp.netloc.split('.')[-2]
    end_path = urlp.netloc.split('.')[-1]
    is_www = urlp.netloc.split('.')[0]=='www'
    no_of_dash = len([i for i in url if i == '-'])
    has_home = 'home' in url
    return le,sch,dig,is_query,is_param,is_frag,host_len,no_of_prd,no_of_subd,base_path,end_path,is_www,no_of_dash,has_home

data1['le'],data1['sch'],data1['dig'],data1['is_query'],data1['is_param'],data1['is_frag'],data1['host_len'],data1['no_of_prd'],data1['no_of_subd'],data1['base_path'],data1['end_path'],data1['is_www'],data1['no_of_dash'],data1['has_home'] =zip(*data1['URL'].map(get_url_features))

data_test_final['le'],data_test_final['sch'],data_test_final['dig'],data_test_final['is_query'],data_test_final['is_param'],data_test_final['is_frag'],data_test_final['host_len'],data_test_final['no_of_prd'],data_test_final['no_of_subd'],data_test_final['base_path'],data_test_final['end_path'],data_test_final['is_www'],data_test_final['no_of_dash'],data_test_final['has_home'] =zip(*data_test_final['URL'].map(get_url_features))

"""###### Preprocessing keywords"""

data1['KEYWORDS'] = data1['KEYWORDS'].apply(lambda x:x.replace('|',' '))

data_test_final['KEYWORDS'] = data_test_final['KEYWORDS'].apply(lambda x:x.replace('|',' '))

"""###### 3. Test-train split"""

data1.is_h.value_counts()

import numpy as np
from sklearn.model_selection import train_test_split

train,test = train_test_split(data1, test_size=0.2, stratify=data1[['is_h','TAXONOMY_Pred']], random_state=101)

train.shape

test.shape

# train.to_parquet(r'train1.parquet')
# test.to_parquet(r'test1.parquet')
# data1.to_parquet(r'data1.parquet')
# data_test_final.to_parquet(r'test_submit.parquet')

# run when training on complete data
# train = data1.copy()
# test = data1.copy()

"""###### 4. Feature selection"""

#device type
train.DEVICETYPE = np.where(train.DEVICETYPE=='Unknown',train.DEVICETYPE.mode(),train.DEVICETYPE)
test.DEVICETYPE = np.where(test.DEVICETYPE=='Unknown',train.DEVICETYPE.mode(),test.DEVICETYPE)

#PLATFORM_ID
x = pd.DataFrame(train[['PLATFORM_ID']].value_counts(normalize = True)).reset_index()
y_plat = x[x[0]>0.009].PLATFORM_ID.values
# y = [ 2,  7,  9,  5,  6, 10]
train['PLATFORM_ID'] = np.where(train['PLATFORM_ID'].isin(y_plat),train['PLATFORM_ID'],'Others')
test['PLATFORM_ID'] = np.where(test['PLATFORM_ID'].isin(y_plat),test['PLATFORM_ID'],'Others')

x = pd.DataFrame(train[['USERCITY']].value_counts(normalize = True)).reset_index()
y_city = x[x[0]>0.009].USERCITY.values
y_city
train['USERCITY_top'] = np.where(train['USERCITY'].isin(y_city),train['USERCITY'],'Others')
test['USERCITY_top'] = np.where(test['USERCITY'].isin(y_city),test['USERCITY'],'Others')

train['USERZIPCODE'] = train['USERZIPCODE'].fillna('Unknown')
train['USERZIPCODE'] = train['USERZIPCODE'].astype(str).apply(lambda x:x.split('.')[0])
test['USERZIPCODE'] = test['USERZIPCODE'].fillna('Unknown')
test['USERZIPCODE'] = test['USERZIPCODE'].astype(str).apply(lambda x:x.split('.')[0])
x = pd.DataFrame(train[['USERZIPCODE']].value_counts(normalize = True)).reset_index()
y_zip = x[x[0]>0.009].USERZIPCODE.values
y_zip
train['USERZIPCODE_top'] = np.where(train['USERZIPCODE'].isin(y_zip),train['USERZIPCODE'],'Others')
test['USERZIPCODE_top'] = np.where(test['USERZIPCODE'].isin(y_zip),test['USERZIPCODE'],'Others')

# PLATFORMTYPE
top = ['Online Medical Journal','Online Learning Portal']
train['PLATFORMTYPE_top'] = np.where(train['PLATFORMTYPE'].isin(top),train['PLATFORMTYPE'],'Others')
test['PLATFORMTYPE_top'] = np.where(test['PLATFORMTYPE'].isin(top),test['PLATFORMTYPE'],'Others')

train.reset_index(inplace = True,drop=True)
test.reset_index(inplace = True,drop=True)

train['device_brand'] = train['device_brand'].fillna('Unknown')
test['device_brand'] = test['device_brand'].fillna('Unknown')
x = pd.DataFrame(train['device_brand'].value_counts()).reset_index()
y_dev = x[x['device_brand']>500]['index'].values
y_dev
train['device_brand_top'] = np.where(train['device_brand'].isin(y_dev),train['device_brand'],'Others')
test['device_brand_top'] = np.where(test['device_brand'].isin(y_dev),test['device_brand'],'Others')

train['device_family'] = train['device_family'].fillna('Unknown')
test['device_family'] = test['device_family'].fillna('Unknown')
x = pd.DataFrame(train['device_family'].value_counts()).reset_index()
y_devf = x[x['device_family']>500]['index'].values
y_devf
train['device_family_top'] = np.where(train['device_family'].isin(y_devf),train['device_family'],'Others')
test['device_family_top'] = np.where(test['device_family'].isin(y_devf),test['device_family'],'Others')

train['device_model'] = train['device_model'].fillna('Unknown')
test['device_model'] = test['device_model'].fillna('Unknown')
x = pd.DataFrame(train['device_model'].value_counts()).reset_index()
y_devm = x[x['device_model']>500]['index'].values
y_devm
train['device_model_top'] = np.where(train['device_model'].isin(y_devm),train['device_model'],'Others')
test['device_model_top'] = np.where(test['device_model'].isin(y_devm),test['device_model'],'Others')

train['os'] = train['os'].fillna('Unknown')
test['os'] = test['os'].fillna('Unknown')
x = pd.DataFrame(train['os'].value_counts()).reset_index()
yos = x[x['os']>500]['index'].values
yos
train['os_top'] = np.where(train['os'].isin(yos),train['os'],'Others')
test['os_top'] = np.where(test['os'].isin(yos),test['os'],'Others')

train['ua'] = train['ua'].fillna('Unknown')
test['ua'] = test['ua'].fillna('Unknown')
x = pd.DataFrame(train['ua'].value_counts()).reset_index()
yua = x[x['ua']>10]['index'].values
yua
train['ua_top'] = np.where(train['ua'].isin(yua),train['ua'],'Others')
test['ua_top'] = np.where(test['ua'].isin(yua),test['ua'],'Others')

train['base_path'] = train['base_path'].fillna('Unknown')
test['base_path'] = test['base_path'].fillna('Unknown')
x = pd.DataFrame(train['base_path'].value_counts()).reset_index()
ybase_path = x[x['base_path']>100]['index'].values
ybase_path
train['base_path_top'] = np.where(train['base_path'].isin(ybase_path),train['base_path'],'Others')
test['base_path_top'] = np.where(test['base_path'].isin(ybase_path),test['base_path'],'Others')

MEE_encoder = MEstimateEncoder()
train_mee = MEE_encoder.fit_transform(train[['USERCITY','USERZIPCODE','IP1', 'IP2', 'IP3','base_path']], train['is_h'],random_state=101, randomized=True)
test_mee = MEE_encoder.transform(test[['USERCITY','USERZIPCODE','IP1', 'IP2', 'IP3','base_path']])
train_mee.columns = ['USERCITY_MEE','USERZIPCODE_MEE','IP1_MEE', 'IP2_MEE', 'IP3_MEE','base_path_MEE']
test_mee.columns = ['USERCITY_MEE','USERZIPCODE_MEE','IP1_MEE', 'IP2_MEE', 'IP3_MEE','base_path_MEE']

ce_encoder = CountEncoder()
train_ce = ce_encoder.fit_transform(train[['USERCITY','USERZIPCODE','IP1', 'IP2', 'IP3','base_path']], train['is_h'],random_state=101, randomized=True)
test_ce = ce_encoder.transform(test[['USERCITY','USERZIPCODE','IP1', 'IP2', 'IP3','base_path']])
train_ce.columns = ['USERCITY_ce','USERZIPCODE_ce','IP1_ce', 'IP2_ce', 'IP3_ce','base_path_ce']
test_ce.columns = ['USERCITY_ce','USERZIPCODE_ce','IP1_ce', 'IP2_ce', 'IP3_ce','base_path_ce']

#KEY_WORDS
tf_idf = TfidfVectorizer()
#applying tf idf to training data
X_train_tf = tf_idf.fit_transform(train['KEYWORDS'])
train_df = pd.DataFrame(X_train_tf.toarray(), columns=tf_idf.get_feature_names_out())
#applying tf idf to training data
X_test_tf = tf_idf.transform(test['KEYWORDS'])
test_df = pd.DataFrame(X_test_tf.toarray(), columns=tf_idf.get_feature_names_out())

train_tf = pd.merge(train[['ID']],train_df,left_index = True,right_index = True,how = 'inner')
test_tf = pd.merge(test[['ID']],test_df,left_index = True,right_index = True,how = 'inner')

X_data_test_final_tf = tf_idf.transform(data_test_final['KEYWORDS'])
data_test_final_df = pd.DataFrame(X_data_test_final_tf.toarray(), columns=tf_idf.get_feature_names_out())
data_test_final_tf = pd.merge(data_test_final[['ID']],data_test_final_df,left_index = True,right_index = True,how = 'inner')



data_test_final.DEVICETYPE = np.where(data_test_final.DEVICETYPE=='Unknown',train.DEVICETYPE.mode(),data_test_final.DEVICETYPE)
data_test_final['PLATFORM_ID'] = np.where(data_test_final['PLATFORM_ID'].isin(y_plat),data_test_final['PLATFORM_ID'],'Others')
data_test_final['USERCITY_top'] = np.where(data_test_final['USERCITY'].isin(y_city),data_test_final['USERCITY'],'Others')
data_test_final['USERZIPCODE'] = data_test_final['USERZIPCODE'].fillna('Unknown')
data_test_final['USERZIPCODE'] = data_test_final['USERZIPCODE'].astype(str).apply(lambda x:x.split('.')[0])
data_test_final['USERZIPCODE_top'] = np.where(data_test_final['USERZIPCODE'].isin(y_zip),data_test_final['USERZIPCODE'],'Others')
data_test_final['PLATFORMTYPE_top'] = np.where(data_test_final['PLATFORMTYPE'].isin(top),data_test_final['PLATFORMTYPE'],'Others')
data_test_final.reset_index(inplace = True,drop=True)
data_test_final['device_brand_top'] = np.where(data_test_final['device_brand'].isin(y_dev),data_test_final['device_brand'],'Others')
data_test_final['device_family_top'] = np.where(data_test_final['device_family'].isin(y_devf),data_test_final['device_family'],'Others')
data_test_final['device_model_top'] = np.where(data_test_final['device_model'].isin(y_devm),data_test_final['device_model'],'Others')
data_test_final['os_top'] = np.where(data_test_final['os'].isin(yos),data_test_final['os'],'Others')
data_test_final['ua_top'] = np.where(data_test_final['ua'].isin(yua),data_test_final['ua'],'Others')
data_test_final['base_path_top'] = np.where(data_test_final['base_path'].isin(ybase_path),data_test_final['base_path'],'Others')

data_test_final_mee = MEE_encoder.transform(data_test_final[['USERCITY','USERZIPCODE','IP1', 'IP2', 'IP3','base_path']])
data_test_final_mee.columns = ['USERCITY_MEE','USERZIPCODE_MEE','IP1_MEE', 'IP2_MEE', 'IP3_MEE','base_path_MEE']

data_test_final_ce = ce_encoder.transform(data_test_final[['USERCITY','USERZIPCODE','IP1', 'IP2', 'IP3','base_path']])
data_test_final_ce.columns = ['USERCITY_ce','USERZIPCODE_ce','IP1_ce', 'IP2_ce', 'IP3_ce','base_path_ce']
data_test_final = pd.merge(data_test_final,data_test_final_ce,left_index = True,right_index = True,how = 'inner')

train_mee_tf = pd.merge(train,train_mee,left_index = True,right_index = True,how = 'inner').merge(train_tf,on = 'ID',how='left')
test_mee_tf = pd.merge(test,test_mee,left_index = True,right_index = True,how = 'inner').merge(test_tf,on = 'ID',how='left')
data_test_final_mee_tf = pd.merge(data_test_final,data_test_final_mee,left_index = True,right_index = True,how = 'inner').merge(data_test_final_tf,on = 'ID',how='left')

train_mee_tf.to_parquet(r'train_mee_tf.parquet')
test_mee_tf.to_parquet(r'test_mee_tf.parquet')
data_test_final_mee_tf.to_parquet(r'data_test_final_mee_tf.parquet')

train_mee_v = pd.merge(train,train_mee,left_index = True,right_index = True,how = 'inner').merge(train_v,on = 'ID',how='left')
test_mee_v = pd.merge(test,test_mee,left_index = True,right_index = True,how = 'inner').merge(test_v,on = 'ID',how='left')
data_test_final_mee_v = pd.merge(data_test_final,data_test_final_mee,left_index = True,right_index = True,how = 'inner').merge(data_test_final_v,on = 'ID',how='left')

train_mee_v.to_parquet(r'train_mee_v.parquet')
test_mee_v.to_parquet(r'test_mee_v.parquet')
data_test_final_mee_v.to_parquet(r'data_test_final_mee_v.parquet')

train_ce_tf = pd.merge(train,train_ce,left_index = True,right_index = True,how = 'inner').merge(train_tf,on = 'ID',how='left')
test_ce_tf = pd.merge(test,test_ce,left_index = True,right_index = True,how = 'inner').merge(test_tf,on = 'ID',how='left')
data_test_final_ce_tf = pd.merge(data_test_final,data_test_final_ce,left_index = True,right_index = True,how = 'inner').merge(data_test_final_tf,on = 'ID',how='left')

train_ce_tf.to_parquet(r'train_ce_tf.parquet')
test_ce_tf.to_parquet(r'test_ce_tf.parquet')
data_test_final_ce_tf.to_parquet(r'data_test_final_ce_tf.parquet')

train_ce_v = pd.merge(train,train_ce,left_index = True,right_index = True,how = 'inner').merge(train_v,on = 'ID',how='left')
test_ce_v = pd.merge(test,test_ce,left_index = True,right_index = True,how = 'inner').merge(test_v,on = 'ID',how='left')
data_test_final_ce_v = pd.merge(data_test_final,data_test_final_ce,left_index = True,right_index = True,how = 'inner').merge(data_test_final_v,on = 'ID',how='left')

train_ce_v.to_parquet(r'train_ce_v.parquet')
test_ce_v.to_parquet(r'test_ce_v.parquet')
data_test_final_ce_v.to_parquet(r'data_test_final_ce_v.parquet')

