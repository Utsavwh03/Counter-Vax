import torch
import openai 
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(3) if torch.cuda.is_available() else "No GPU")

device = torch.device("cuda:1")

import os
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)


import pandas as pd
import os

# read the data
train_data=pd.read_csv('/home/sohampoddar/HDD2/utsav/Dataset/train.csv')

train_data.__len__

test_data=pd.read_csv('/home/sohampoddar/HDD2/utsav/Dataset/test.csv')
test_data.__len__

import csv
from openai import OpenAI
client = OpenAI()

# choose random 10 samples
data=train_data.sample(n=5)
texts = data['text']
label= data['labels']
results=[]

### Prompt 1

with open('Prompts/counter_examples_brief.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Label', 'Prompt', 'Generated Counter Argument']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for lab, text in zip(label, texts):
        # Create a prompt with the text
        # if there are more than one labels then only do else continue

        prompt = f"Generate a brief counter argument for the anti-vaccine tweet :\n{text}\n\n"
        # print(prompt)
        # Generate the completion
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        generated_text = completion.choices[0].message.content
        # print(generated_text)
        # Write the result to the CSV file
        writer.writerow({
            'Label': lab,
            'Prompt': prompt,
            'Generated Counter Argument': generated_text
        })

        # Also store in the list for future use if needed
        results.append(generated_text)

print("All results have been saved to 'counter_arguments_prompt1.csv'.")


### Prompt 2
# import csv
# from openai import OpenAI
client = OpenAI()
results=[]

with open('Prompts/counter_examples_prompt2.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Label', 'Prompt', 'Generated Counter Argument']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for lab, text in zip(label, texts):
        # Create a prompt with the text
        # if there are more than one labels then only do else continue

        prompt = f"Summarize a quick rebuttal for this anti-vaccine tweet:\n{text}\n\n"
        # print(prompt)
        # Generate the completion
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        generated_text = completion.choices[0].message.content
        # print(generated_text)
        # Write the result to the CSV file
        writer.writerow({
            'Label': lab,
            'Prompt': prompt,
            'Generated Counter Argument': generated_text
        })

        # Also store in the list for future use if needed
        results.append(generated_text)

print("All results have been saved to 'counter_arguments_prompt2.csv'.")


### Prompt 3
client = OpenAI()
results=[]

with open('Prompts/counter_examples_prompt3.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Label', 'Prompt', 'Generated Counter Argument']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for lab, text in zip(label, texts):
        # Create a prompt with the text
        # if there are more than one labels then only do else continue

        prompt = f"Give a short counterpoint to the statement in this anti-vaccine tweet:\n{text}\n\n"

        # print(prompt)
        # Generate the completion
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        generated_text = completion.choices[0].message.content
        # print(generated_text)
        # Write the result to the CSV file
        writer.writerow({
            'Label': lab,
            'Prompt': prompt,
            'Generated Counter Argument': generated_text
        })

        # Also store in the list for future use if needed
        results.append(generated_text)

print("All results have been saved to 'counter_arguments_prompt3.csv'.")


