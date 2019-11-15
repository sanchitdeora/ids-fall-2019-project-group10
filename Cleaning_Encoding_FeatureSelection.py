import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing


df = pd.read_csv("11142019_train_subsampled.csv")

# Conversion because the number does not fit long in C
df['fullVisitorId'] = df['fullVisitorId'].astype(float)


'''
Code used for exploration for the purpose of data cleaning

df['totals.bounces'].value_counts()

df.isna().sum(axis=0)

'''


# Columns to be dropped (the number represents the percent of null values)
# 100                   100                    100                     77                  100                           99.99                         <does not serve any purpose>
# 'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.metro', 'geoNetwork.networkLocation', 'trafficSource.campaignCode', 'trafficSource.adwordsClickInfo.isVideoAd'


# Set the na values of following columns to 
# 'geoNetwork.networkDomain': '(not set)' => 'unknown.unknown'
# 'totals.bounces': NA => -1
# 'trafficSource.referralPath' => 'UNK'
# 'totals.transactions' => 0
# 'totals.transactionRevenue' => 0
# 'totals.totalTransactionRevenue' => 0
# 'trafficSource.isTrueDirect' => False
# 'trafficSource.keyword' => 'UNK'
# 'trafficSource.adwordsClickInfo.slot' => 'UNK'
# 'trafficSource.adwordsClickInfo.page' => 0
# 'trafficSource.adwordsClickInfo.gclId' => 'UNK'
# 'trafficSource.adwordsClickInfo.adNetworkType' => 'UNK'
# 'trafficSource.adContent' => 'UNK'
# 'totals.timeOnSite' => 0 or -1
# 'totals.sessionQualityDim' => 0
# 'totals.pageviews' => mean of the column
# 'totals.newVisits' => 0


##############################################
# Applying the above mentioned transformations
##############################################

df = df.drop(columns=['geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.metro', 'geoNetwork.networkLocation', 'trafficSource.campaignCode', 'trafficSource.adwordsClickInfo.isVideoAd'])


df['geoNetwork.networkDomain'] = df['geoNetwork.networkDomain'].map(lambda value: 'unknown.unknown' if value == '(not set)' else value)

fill_0_columns = ['totals.transactions', 'totals.transactionRevenue', 'totals.totalTransactionRevenue', 'trafficSource.adwordsClickInfo.page', 'totals.timeOnSite', 'totals.sessionQualityDim', 'totals.newVisits']
df[fill_0_columns] = df[fill_0_columns].fillna(0)

df = df.fillna({
    'totals.bounces': -1, 
    'trafficSource.referralPath': 'UNK', 
    'trafficSource.isTrueDirect': False, 
    'trafficSource.keyword': 'UNK', 
    'trafficSource.adwordsClickInfo.slot': 'UNK', 
    'trafficSource.adwordsClickInfo.gclId': 'UNK', 
    'trafficSource.adwordsClickInfo.adNetworkType': 'UNK', 
    'trafficSource.adContent': 'UNK',
})

df['totals.pageviews'] = df[['totals.pageviews']].fillna(df['totals.pageviews'].mean())


################################################
# Code for feature selection
# Only needs to be run once
# The model takes upto 25 GB of memory during and after runtime because of the number of estimators
################################################


forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

y = df['new.returningCustomer']
X = df.drop(columns=['new.returningCustomer'])

# Encoding the columns with Boolean values
lb = preprocessing.LabelBinarizer()
X['trafficSource.isTrueDirect'] = lb.fit_transform(X['trafficSource.isTrueDirect'])
X['device.isMobile'] = lb.transform(X['device.isMobile'])

# Encoding the categorical features
for column in X.columns:
    if (X[column].dtype != np.int64 and X[column].dtype != np.float64):
        print(f"Encoding column: {column}")

        le = preprocessing.LabelEncoder()
        X[column] = le.fit_transform(X[column])
        del le


forest.fit(X, y)


# Filtering the dataframe by removing the features with least importance
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

drop_cols = []

for i in range(len(indices)):
    # print(f"{X.columns[indices[i]]}:\t {importances[indices[i]]}")
    if importances[indices[i]] == 0:
        drop_cols += [X.columns[indices[i]]]

df = df.drop(columns=drop_cols)
