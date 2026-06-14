import pdfplumber
import re
import os
import pandas as pd
from transformers import pipeline

#Testing different classifiers
#classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier = pipeline("zero-shot-classification", model="KBLab/megatron-bert-large-swedish-cased-165-zero-shot") # Swedish words. potentially better classification. https://kb-labb.github.io/posts/2023-02-12-zero-shot-text-classification/
# classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli") # way too slow in comparison to the previous two. After 30 min I cancel the run
df = pd.read_csv('labels.csv')
#print(df.to_string)
#print(df.head())
#print(df['label'].value_counts())

item_purchased = df['name']
#print(f"The items that are in the data frame: \n {item_purchased}")
#labels = ["utilities", "healthy food", "junk food", "energy drink"]
labels = ["hushåll", "hälsosam mat", "skräpmat", "energidryck"] # for the swedish version of the classifier
#result = classifier(item_purchased, labels)

#print(f"Result of the labeling:\n {result}")
directory = 'ICA Kvitton'  # set directory path
items = []
total_price = 0

# Iterating over the files in the directory, according to https://www.geeksforgeeks.org/python/how-to-iterate-over-files-in-directory-using-python/
print("--- Reading Receipts ---")
for entry in os.listdir(directory):
    if entry.endswith(".pdf"):  
        full_path = os.path.join(directory, entry)
        with pdfplumber.open(full_path) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text()

    # There are two "categories" that are 13 digits on the receipts: Artikelnummer and AID. The and-statement filters out the "AID" digits and allows us to only extract the rows with actual Artikelnummer
        for i in text.splitlines():
            if re.search(r'\d{13}', i) and re.search(r"[+-]?[0-9]+\,[0-9]+", i):
                i_to_list = i.split()

                # print(i)
                # print(i_to_list)
                last_element = i_to_list[-1]
                # print(last_element)
                replace_comma = last_element.replace(",",".")
                # print(replace_comma)
                price_to_float = float(replace_comma)
                total_price += price_to_float
                # print(f"{price_to_float:.2f}")
                # print(i_to_list[-1::])
                name_list = i_to_list[:-5]
                full_name = " ".join(name_list)
                current_item = {"name": full_name, "price": price_to_float}
                items.append(current_item)
                # print(f"Item: {full_name} | Price: {price_to_float}")

# Print out the items as tuples in a list, along with the total sum of all items ever purchased
print(items[0])
# print(f"Total: {total_price} kr")

# Making the list of items to a table https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
pandas_df = pd.DataFrame(items)
# print(f"This is what's in pandas_df: {pandas_df}")
unique_items = pandas_df["name"].unique()
# print(f"This is what's in unique_items: {unique_items}")
unique_items_list = list(unique_items)
result = classifier(unique_items_list, labels, hypothesis="Varan på kvittot är {}")

print(f"The result from the classifier: {result}")
print(type(result))
# result is a dictionary. Converting it to a list
if isinstance(result, dict):
    result = [result]
print(type(result))

item_to_label = {}

for res in result:
    item_name = res["sequence"]
    label_prediction = res["labels"][0] # Index 0 because the "highest guess"/best label is always the first index
    item_to_label[item_name] = label_prediction
pandas_df["label"] = pandas_df["name"].map(item_to_label)

labeled_items = pandas_df[["name","label"]]
print(f"List of labeled foods: \n {labeled_items}")
# Each label gets a weight assigned to it. This is used for health scoring
#health_scoring = {"healthy food": 1.0, "junk food": 0.0, "energy drink": 0.0, "not food": 0.5}
health_scoring = {"hälsosam mat": 1.0, "skräpmat": 0.0, "energidryck": 0.0, "hushåll": 0.5}
pandas_df["health_value"] = pandas_df["label"].map(health_scoring)

average_health_score = pandas_df["health_value"].mean()

print(f"Average health score: {average_health_score:.2f}")
