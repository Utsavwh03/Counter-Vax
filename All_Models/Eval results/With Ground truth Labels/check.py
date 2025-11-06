import pandas as pd

df = pd.read_csv("/home/sohampoddar/Restored Data/HDD2/sohampoddar/utsav/All_Models/Eval results/With Ground truth Labels/phi3mini4k_with_labels_with_labels.csv")

print(df['prediction'][1])
print("--------------------------------")
print(df['reference'][1])
print("--------------------------------")
print(df['labels'][1])