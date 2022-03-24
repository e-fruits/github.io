import numpy as np
import pandas as pd
import sqlite3

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from pickle import dump

####################
# uncomment the lines depending on your source (SQLite or csv)
# SQLLite
# connection = sqlite3.connect("loan_database")  # connect to sql db
# df = pd.read_sql_query('SELECT * FROM joined_data;', connection)
# connection.execute("VACUUM;")

# from CSV (during development)
df = pd.read_csv('joined_data.csv', low_memory=False)
print('import done')

#####################

# replace special values with null based on Prosper documentation
# we aren't going to worry about mixed type features as a simplifying assumption
df.replace(to_replace=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, 999],
           value = np.nan,
           inplace = True)
print("replaced special values")

# convert all column names to lowercase
df.columns = df.columns.str.strip().str.lower()

# drop some un-needed columns
# df.drop(['unnamed: 0', 'level_0', 'unnamed: 0.1'], inplace=True, axis=1)
df.drop(['unnamed: 0'], inplace=True, axis=1)

#drop Experian fields
exp_fields_to_drop = pd.read_excel('tu_exp_fields.xlsx', sheet_name='EXP')
exp_fields_to_drop = exp_fields_to_drop['Field']
df.drop(exp_fields_to_drop, inplace=True, axis=1)

# create year column & leave as string (will one-hot encode later)
df['year'] = df['loan_origination_date'].str[:4]

# store as a vector for filter later
year = df['loan_origination_date'].str[:4].astype(int)

# drop columns with 'date' in name since we have captured origination year
df.drop(df.filter(regex='date').columns, inplace=True, axis=1)
df.drop(df.filter(regex='paid').columns, inplace=True, axis=1)

print('Removed dates and paid columns')

# create training dataframe
# we still need to keep to the side to identify records later
loan_numbers = df['loan_number']

# create default flag vector
default_flag = np.where(df['loan_status'] == 2, 1, 0)

# remove columns we know are not known at origination or that we do not want in model
df.drop(['age_in_months', 'days_past_due', 'loan_number', 'days_past_due', 'principal_balance',
         'debt_sale_proceeds_received', 'next_payment_due_amount', 'loan_default_reason',
         'loan_default_reason_description', 'index', 'member_key', 'listing_number', 'amount_funded',
         'amount_remaining', 'percent_funded', 'partial_funding_indicator', 'funding_threshold',
         'estimated_return', 'estimated_loss_rate', 'lender_yield', 'effective_yield', 'listing_category_id',
         'income_range', 'lender_indicator', 'group_indicator', 'group_name', 'channel_code',
         'amount_participation', 'investment_typeid', 'investment_type_description', 'loan_status',
         'loan_status_description', 'listing_status_reason', 'borrower_city', 'borrower_metropolitan_area',
         'first_recorded_credit_line', 'investment_type_description', 'tuficorange', 'listing_term', 'listing_amount',
         'borrower_apr']
        , inplace=True
        , axis=1)

# identify non numeric columns to one-hot encode
str_cols = list(df.select_dtypes(include=['object', 'string']).columns)
#print(str_cols)

# add loan term to features to one-hot encode. We want to treat as categorical since only three possible terms.
str_cols.append('term')

# write function to one-hot encode specific features
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], dummy_na=True)
    result = pd.concat([original_dataframe, dummies], axis=1)
    result = result.drop([feature_to_encode], axis=1)
    return result

# perform one hot encoding on string features
for feature in str_cols:
    df = encode_and_bind(df, feature)

print('Finished One-Hot encoding')

# filter to 2017 and beyond since that is when TransUnion started being used
full_df = df
full_default_flag = default_flag
# default_flag = default_flag[df['year'].astype(int) >= 2017]
default_flag = default_flag[year >= 2017]
# df = df[df['year'].astype(int) >= 2017]
df = df[year >= 2017]

print('Finished filtering by year')

#capture feature names to ID later
feature_names = pd.Series(df.columns.values)
feature_names.to_csv('feature_names_considered.csv', index=False)
# dump(feature_names, open('feature_names.pkl', 'wb'))

# filter by prosper rating
df_AA = df[df['prosper_rating_AA'] == 1]
df_A = df[df['prosper_rating_A'] == 1]
df_B = df[df['prosper_rating_B'] == 1]
df_C = df[df['prosper_rating_C'] == 1]
df_D = df[df['prosper_rating_D'] == 1]
df_E = df[df['prosper_rating_E'] == 1]
df_HR = df[df['prosper_rating_HR'] == 1]

# convert to array to pass to the model
df_AA = df_AA.values
df_A = df_A.values
df_B = df_B.values
df_C = df_C.values
df_D = df_D.values
df_E = df_E.values
df_HR = df_HR.values

# Fill n/a and inf values with 0 now that missing flag is set
df_AA[~np.isfinite(df_AA)] = 0
df_A[~np.isfinite(df_A)] = 0
df_B[~np.isfinite(df_B)] = 0
df_C[~np.isfinite(df_C)] = 0
df_D[~np.isfinite(df_D)] = 0
df_E[~np.isfinite(df_E)] = 0
df_HR[~np.isfinite(df_HR)] = 0
print('Defined model datasets done')

# start modeling
# define model hyperparameters and cv
def logistic_cv(x_train, y_true, class_wgts, folds=5, regs = [.05], max_iterations=500):
    return LogisticRegressionCV(Cs=regs, cv=folds, penalty='l1', class_weight=class_wgts, scoring='f1',
                                max_iter=max_iterations, solver='saga', random_state=1990).fit(x_train, y_true)


# find optimal class weights and regularization strength
weights = np.linspace(0.04, 0.07, 4)
regs = [.01, .05, .1]
gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'class_weight': [{0: x, 1: 1.0-x} for x in weights], 'C': regs, 'penalty': ['l1'], 'random_state': [1990],
        'solver': ['saga'], 'max_iter': [750]
    },
    scoring='f1',
    cv=3
)

# prosper rating AA
scaler_AA = StandardScaler().fit(df_AA)
train = scaler_AA.transform(df_AA)
y = default_flag[df['prosper_rating_AA'] == 1]
model_AA = logistic_cv(train, y, {0: .04, 1: .96}, folds=5, regs = [.01], max_iterations=750)
features_AA = np.where(model_AA.coef_ != 0)
print('The AA model variables & coefficients are: ',
      list(zip(np.array(feature_names)[features_AA[1].astype(int)],
               model_AA.coef_[np.where(model_AA.coef_ != 0)])))

# uncomment the next two lines if you want to save the model and scaler
# dump(model_AA, open('model_AA.pkl', 'wb'))
# dump(scaler_AA, open('scaler_AA.pkl', 'wb'))

# prosper rating A
scaler_A = StandardScaler().fit(df_A)
train = scaler_A.transform(df_A)
y = default_flag[df['prosper_rating_A'] == 1]
# model_A = gsc.fit(train, y)
model_A = logistic_cv(train, y, {0: .04, 1: .96}, folds=5, regs = [.01], max_iterations=750)
features_A = np.where(model_A.coef_ != 0)
print('The A model variables & coefficients are: ',
      list(zip(np.array(feature_names)[features_A[1].astype(int)],
               model_A.coef_[np.where(model_A.coef_ != 0)])))

# uncomment the next two lines if you want to save the model and scaler
# dump(model_A, open('model_A.pkl', 'wb'))
# dump(scaler_A, open('scaler_A.pkl', 'wb'))


# prosper rating B
scaler_B = StandardScaler().fit(df_B)
train = scaler_B.transform(df_B)
y = default_flag[df['prosper_rating_B'] == 1]
model_B = logistic_cv(train, y, {0: .04, 1: .96}, folds=5, regs = [.01], max_iterations=500)
# model_B = gsc.fit(train, y)
features_B = np.where(model_B.coef_ != 0)
print('The B model variables & coefficients are: ',
      list(zip(np.array(feature_names)[features_B[1].astype(int)],
               model_B.coef_[np.where(model_B.coef_ != 0)])))

# uncomment the next two lines if you want to save the model and scaler
# dump(model_B, open('model_B.pkl', 'wb'))
# dump(scaler_B, open('scaler_B.pkl', 'wb'))

# prosper rating C
scaler_C = StandardScaler().fit(df_C)
train = scaler_C.transform(df_C)
y = default_flag[df['prosper_rating_C'] == 1]
model_C = logistic_cv(train, y, {0: .04, 1: .96}, folds=5, regs = [.01], max_iterations=500)
# model_C = gsc.fit(train, y)
features_C = np.where(model_C.coef_ != 0)
print('The C model variables & coefficients are: ',
      list(zip(np.array(feature_names)[features_C[1].astype(int)],
               model_C.coef_[np.where(model_C.coef_ != 0)])))

# uncomment the next two lines if you want to save the model and scaler
# dump(model_C, open('model_C.pkl', 'wb'))
# dump(scaler_C, open('scaler_C.pkl', 'wb'))

# prosper rating D
scaler_D = StandardScaler().fit(df_D)
train = scaler_D.transform(df_D)
y = default_flag[df['prosper_rating_D'] == 1]
model_D = logistic_cv(train, y, {0: .04, 1: .96}, folds=5, regs = [.01], max_iterations=750)
# model_D = gsc.fit(train, y)
features_D = np.where(model_D.coef_ != 0)
print('The D model variables & coefficients are: ',
      list(zip(np.array(feature_names)[features_D[1].astype(int)],
               model_D.coef_[np.where(model_D.coef_ != 0)])))

# uncomment the next two lines if you want to save the model and scaler
# dump(model_D, open('model_D.pkl', 'wb'))
# dump(scaler_D, open('scaler_D.pkl', 'wb'))

# prosper rating E
scaler_E = StandardScaler().fit(df_E)
train = scaler_E.transform(df_E)
y = default_flag[df['prosper_rating_E'] == 1]
model_E = logistic_cv(train, y, {0: .04, 1: .96}, folds=5, regs = [.05])
#model_E = gsc.fit(train, y)
features_E = np.where(model_E.coef_ != 0)
print('The E model variables & coefficients are: ',
      list(zip(np.array(feature_names)[features_E[1].astype(int)],
               model_E.coef_[np.where(model_E.coef_ != 0)])))

# uncomment the next two lines if you want to save the model and scaler
# dump(model_E, open('model_E.pkl', 'wb'))
# dump(scaler_E, open('scaler_E.pkl', 'wb'))

# prosper rating HR
scaler_HR = StandardScaler().fit(df_HR)
train = scaler_HR.transform(df_HR)
y = default_flag[df['prosper_rating_HR'] == 1]
model_HR = logistic_cv(train, y, {0: .04, 1: .96}, folds=5, regs = [.1], max_iterations = 1000)
# model_HR = gsc.fit(train, y)
features_HR = np.where(model_HR.coef_ != 0)
print('The HR model variables & coefficients are: ',
      list(zip(np.array(feature_names)[features_HR[1].astype(int)],
               model_HR.coef_[np.where(model_HR.coef_ != 0)])))

# uncomment the next two lines if you want to save the model and scaler
# dump(model_HR, open('model_HR.pkl', 'wb'))
# dump(scaler_HR, open('scaler_HR.pkl', 'wb'))

### PROBABILITIES ARE BIASED, BUT CAN BE USED FOR THRESHOLDS

full_df[~np.isfinite(full_df)] = 0
train = full_df
pred = dict.fromkeys(['AA', 'A', 'B', 'C', 'D', 'E', 'HR', 'nan'])

pred['AA'] = model_AA.predict_proba(scaler_AA.transform(train[train['prosper_rating_AA'] == 1].values))[:, 1]
pred['A'] = model_A.predict_proba(scaler_A.transform(train[train['prosper_rating_A'] == 1].values))[:, 1]
pred['B'] = model_B.predict_proba(scaler_B.transform(train[train['prosper_rating_B'] == 1].values))[:, 1]
pred['C'] = model_C.predict_proba(scaler_C.transform(train[train['prosper_rating_C'] == 1].values))[:, 1]
pred['D'] = model_D.predict_proba(scaler_D.transform(train[train['prosper_rating_D'] == 1].values))[:, 1]
pred['E'] = model_E.predict_proba(scaler_E.transform(train[train['prosper_rating_E'] == 1].values))[:, 1]
pred['HR'] = model_HR.predict_proba(scaler_HR.transform(train[train['prosper_rating_HR'] == 1].values))[:, 1]
pred['nan'] = model_C.predict_proba(scaler_C.transform(train[train['prosper_rating_nan'] == 1].values))[:, 1]

pred['AA'] = pd.qcut(pred['AA'], q=3, labels=['Plus', 'Mid', 'Minus'], duplicates='drop')
pred['A'] = pd.qcut(pred['A'], q=3, labels=['Plus', 'Mid', 'Minus'], duplicates='drop')
pred['B'] = pd.qcut(pred['B'], q=3, labels=['Plus', 'Mid', 'Minus'], duplicates='drop')
pred['C'] = pd.qcut(pred['C'], q=3, labels=['Plus', 'Mid', 'Minus'], duplicates='drop')
pred['D'] = pd.qcut(pred['D'], q=3, labels=['Plus', 'Mid', 'Minus'], duplicates='drop')
pred['E'] = pd.qcut(pred['E'], q=3, labels=['Plus', 'Mid', 'Minus'], duplicates='drop')
pred['HR'] = pd.qcut(pred['HR'], q=3, labels=['Plus', 'Mid', 'Minus'], duplicates='drop')
pred['nan'] = pd.qcut(pred['nan'], q=3, labels=['Plus', 'Mid', 'Minus'], duplicates='drop')

print('Created final predictions')
# final = full_df.values

full_df['predict'] = 0
full_df.loc[full_df['prosper_rating_AA'] == 1, 'predict'] = pred['AA']
full_df.loc[full_df['prosper_rating_A'] == 1, 'predict'] = pred['A']
full_df.loc[full_df['prosper_rating_B'] == 1, 'predict'] = pred['B']
full_df.loc[full_df['prosper_rating_C'] == 1, 'predict'] = pred['C']
full_df.loc[full_df['prosper_rating_D'] == 1, 'predict'] = pred['D']
full_df.loc[full_df['prosper_rating_E'] == 1, 'predict'] = pred['E']
full_df.loc[full_df['prosper_rating_HR'] == 1, 'predict'] = pred['HR']
full_df.loc[full_df['prosper_rating_nan'] == 1, 'predict'] = pred['nan']

print(full_df['predict'].head(10))

test = pd.DataFrame(zip(loan_numbers, full_df['predict']))
test.to_csv('notches.csv')

# full_df.drop(['predict'], axis=1, inplace=True)

