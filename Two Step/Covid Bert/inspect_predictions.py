import pandas as pd

# df = pd.read_csv("/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/Covid Bert/val_predictions_covid_bert.csv")

# # retain only the columns text and predicted_labels and true_labels
# df = df[['text', 'predicted_labels', 'true_labels']]

# # print the first 5 rows of the dataframe
# print(df.head())

# # print the last 5 rows of the dataframe
# print(df.tail())

# # rename the csv file and store the dataframe in a new csv file
# df.to_csv("/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/Covid Bert/val_predictions_covid_bert_cleaned.csv", index=False)

# store the predicted_labels in val_CA_with_label_desc.csv in a new column called predicted_labels_from_covid_bert

df = pd.read_csv("/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Two_Step/Covid Bert/val_predictions_covid_bert_cleaned.csv")
val_data = pd.read_csv("/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc.csv")
# store in string list format
predicted_labels_from_covid_bert = df['predicted_labels'].apply(lambda x: x.split(' ')).tolist()
# change labels to predicted_labels_from_covid_bert
val_data['labels'] = predicted_labels_from_covid_bert
val_data.to_csv("/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc_with_predicted_labels_from_covid_bert.csv", index=False)
