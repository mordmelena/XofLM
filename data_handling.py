import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/chexpert.csv')

print(df.columns)

df = df.drop(['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'Lung Opacity', 'No Finding',
       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'], axis=1)

# print(df.tail())
# df.to_csv('data/p_e.csv') # here are pleural effusion labels

text_dict = {'study_id': [], 'report': [], }

''' This code goes through the files and folders  '''
def traverse():
    directory = '/Users/elenamordmillovich/Documents/ Project Clinical Notes/files/'
    if os.path.exists(directory):
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.txt'):
                    with open(os.path.join(dirpath, filename)) as f:
                        text = f.read()
                        write_text_into_dic(text)
                        write_filename_into_dic(filename)
    else:
        print("File does not exist")

''' This function returns a dataframe made from the dictionary'''
def dic_to_csv():
    df_text = pd.DataFrame(text_dict).reset_index(drop=True)
    # print(df_text)
    return df_text

''' This helper function writes filenames/study_id into dictionary'''
def write_filename_into_dic(filename):
    filename = filename[1:-4]
    try:
        text_dict['study_id'].append(int(filename))
    except KeyError:
        text_dict['study_id'] = [filename]

''' This helper function writes reort texts into dictionary'''
def write_text_into_dic(text):
    try:
        text_dict['report'].append(text)
    except KeyError:
        text_dict['report'] = [text]


traverse() # run
''' check out the result, pandas don't show it properly'''
for i in range(0, 5):
    for r in text_dict.items():
        print(r)

dic_to_csv().to_csv('data/reports.csv') # here are reports and ids

'''explore the counts '''
df_rep = pd.read_csv('data/reports.csv')
df_p = pd.read_csv('data/p_e.csv')
print(df_p.head())
rep_sorted = df_rep.sort_values('study_id')
p_e_sorted = df_p.sort_values('study_id')
print(rep_sorted.shape[0]) # 227835
print(p_e_sorted.shape[0]) # 227827
count = p_e_sorted['Pleural Effusion'].isna().sum()
print(count) # 140555

s_rep = rep_sorted['study_id']
s_p = p_e_sorted['study_id']
df_diff = pd.concat([s_rep, s_p]).drop_duplicates(keep=False)
print(df_diff)

''' compare the reports and lable DFs, filter out difference'''
def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]


new_rep = filter_rows_by_values(rep_sorted, "study_id", ['50798377','52035334', '53071062', '53607029', '54168089', '54231141', '56724958', '58235663'])

print(new_rep.shape[0])
print(p_e_sorted.shape[0])

# reports = pd.read_csv('data/reports.csv')
# print(df.dtypes)
# print(reports.dtypes)
inner_join = pd.merge(p_e_sorted, rep_sorted, how='inner', on='study_id')
print(inner_join.shape)
print(inner_join.columns)
inner_join.to_csv('data/inner_join.csv')

dataset = pd.read_csv('data/inner_join.csv')
print(dataset.columns)
dataset = dataset.drop(['Unnamed: 0', 'Unnamed: 0_x', 'subject_id', 'Unnamed: 0_y'], axis = 1)
dataset = dataset[dataset['Pleural Effusion'].notna()]
dataset.to_csv('data/dataset.csv')

'''checked the split file -- there are no test expamples with a report'''
split = pd.read_csv('data/split.csv')
print(split.columns)
split = split.drop(['dicom_id', 'subject_id'], axis=1)
print(split.columns)
join = pd.read_csv('data/inner_join.csv')
print('split: ', split.shape[0])
print('join: ',  join.shape[0])


reports = pd.read_csv('data/reports.csv')
print('reports: ', reports.shape[0])
dataset = pd.read_csv('data/dataset.csv', index_col=[0])
print("dataset length: ", dataset.shape[0])
print("dataset columns: ", dataset.columns)

print('zeros: ', dataset['Pleural Effusion'].value_counts()[0.0])
print(dataset['Pleural Effusion'].value_counts()[1.0])
print(dataset['Pleural Effusion'].value_counts()[-1.0])

data_zero_label = dataset[dataset['Pleural Effusion'] == 0.0]
print(data_zero_label.shape[0])
data_zero_label.to_csv('data/zero_labels.csv') # zero-labeled reports from dataset

'''negbio'''
df = pd.read_csv('data/negbio.csv')
df = df.drop(['subject_id', 'Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'Lung Opacity', 'No Finding',
       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'], axis=1)
print(df.columns)
print(df.shape[0])
count = df['Pleural Effusion'].isna().sum()
print(count)
print('zeros: ', df['Pleural Effusion'].value_counts()[0.0])

'''Drop the zeros from dataset, save as dataset_no_zeros, split balanced'''
dataset_no_zeros = dataset[dataset['Pleural Effusion'] != 0.0]
print(dataset_no_zeros.head())
print(dataset_no_zeros.shape[0])
dataset_no_zeros.to_csv('data/dataset_no_zeros.csv')
print(dataset_no_zeros.columns)
texts = dataset_no_zeros[['report']] # 60114
labels = dataset_no_zeros[['Pleural Effusion']]  # 60114
labels = labels.replace(1.0, 1) # maybe stratify wants 0 and 1...
labels = labels.replace(-1.0, 0)
x_train, x_test, y_train, y_test = train_test_split(texts, labels,  stratify=labels, test_size=0.1 )
print(labels.value_counts()[0], labels.value_counts()[1])
print(labels.dtypes)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)# (42079, 1) (18035, 1) (42079, 1) (18035, 1)
print(y_train.value_counts()[0], y_train.value_counts()[1], y_test.value_counts()[0],y_test.value_counts()[1]) #stratify works

'''do split test into val and test'''



