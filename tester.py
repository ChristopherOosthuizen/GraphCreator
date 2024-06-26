from src import GraphCreation
import os
os.environ['OPENAI_API_KEY']= open("api_key","r").read().strip()
os.environ['TOKENIZERS_PARALLELISM'] = "true"
import pandas as pd
import numpy as np
data = pd.read_csv("fresh.csv")
data.head()

def pick_first_url(row):
    row = str(row).split("\n")[0]
    print(row)
    print()
    return row
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.06, random_state=120)
test.head()
test['source'] = test['source'].apply(pick_first_url)
test.head()['source']
data = GraphCreation.bm.bench_mark_from_dataset(test,"source","answer_0","question", chunks_precentage_linked=0, eliminate_all_islands=False, inital_repeats=30, ner=False, ner_type="llm")
data.to_csv("test.csv",index=False)