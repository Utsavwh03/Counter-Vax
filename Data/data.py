import pandas as pd 
data = pd.read_csv("/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/train_CA_with_label_desc.csv")
# print the number of rows
print(len(data))

val_data = pd.read_csv("/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/Data/val_CA_with_label_desc.csv")

print(len(val_data))