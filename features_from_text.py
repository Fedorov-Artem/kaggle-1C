import pandas as pd
from nltk.tokenize import word_tokenize
import re
import FeaturesTimeSeries
from sklearn.preprocessing import LabelEncoder


# Here I extract features from text description of items and categories
# This is the only information I have about new items
#
# Decided not to build TF-IDF or doc2vec
# Instead I rely on comments written in brackets, and use them to fix categories in a few cases
# Here I also also introduce a broader category - call it "type"
# As some categories contain few items, type is going to be useful for prediction
# Info about sales results here is only used to calculate time period, when an item was first sold

# paths to the data
path_items = 'Input/items.csv'
path_categories = 'Input/item_categories.csv'
path_sales = 'Input/sales_train_v2.csv'

df = pd.read_csv(path_items)


# this function is to clean and tokenize the text


def cleantext(text):
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub('\W+', ' ', text)
    text = text.lower()
    text = word_tokenize(text)
    return text


# Two specific features for prediction of new product sales
# Many item_id's have almost the same text description - for example computer games for each platform
# Here I calculate number of item_id's with almost the same description that were released same month
# This helps distinguish between major releases and for example just new editions of old games
# Second feature is time since first item_id with similar description was released

# To do this I will join sales info and calculate the release month
df_sales = pd.read_csv(path_sales)
df_sales = df_sales.groupby(['item_id']).agg({'date_block_num': 'min'})
df_sales = df_sales.rename(columns={'date_block_num': 'first_month'}).reset_index()
df = pd.merge(df, df_sales, on=['item_id'], how='left')
df['first_month'] = df['first_month'].fillna(34)

# Now clean the text and calculate the features
df['text'] = [re.split('\[|\(|\.|,|–', w)[0] for w in df['item_name']]
df['text'] = df['text'].str.strip().str.lower()
df['first_month_same_text'] = df.groupby('text')['first_month'].transform('min').astype('int16')
df['same_text_this_month'] = df.groupby(['text', 'first_month'])['item_id'].transform('nunique').astype('int16')
df['month_after_same_text'] = df['first_month'] - df['first_month_same_text']

# building wide categories (called them 'types')
df_categories = pd.read_csv(path_categories)
df_categories['type_name'] = [re.split('\(|-', w)[0] for w in df_categories['item_category_name']]
df_categories['type_code'] = LabelEncoder().fit_transform(df_categories['type_name'])
df_categories['is_digital'] = 0
df_categories.loc[df_categories['item_category_name'].str.contains('Цифра', regex=False), 'is_digital'] = 1
df_categories = df_categories[['item_category_id', 'type_name', 'type_code', 'is_digital']]
df = pd.merge(df, df_categories, on='item_category_id', how='left')

# fixing some categories/types manually
# item_id 20949 has very high sales, so I make a cpecial category\type for it, so that it didn't move the average encodings
df.loc[df['item_id'] == 20949, 'type_code'] = df['type_code'].max() + 1
df.loc[df['item_id'] == 20949, 'item_category_id'] = df['item_category_id'].max() + 1

# Making a few straightforward features from tokenized text - two of them will be only used to fix some categories
df['item_name_clean'] = df['item_name'].apply(cleantext)
df['is_region'] = [1 if 'регион' in x else 0 for x in df['item_name_clean']]
df['is_firm'] = [1 if 'фирм' in x else 0 for x in df['item_name_clean']]
df['lang'] = [0 if 'русская' in x or 'русские' in x else (2 if 'английская' in x or 'англ' in x else 1) for x in
              df['item_name_clean']]
df.loc[~df['type_name'].str.contains('Игры', regex=False), 'lang'] = 1

# manual fix for some categories
df.loc[(df['is_region'] == 1) & (df['item_category_id'] == 40), 'item_category_id'] = df['item_category_id'].max() + 1
df.loc[(df['is_firm'] == 1) & (df['item_category_id'] == 55), 'item_category_id'] = 56
df.loc[df['type_code'] == 7, 'type_code'] = 6

# delete all the information that will not be used afterwards
df = df[['item_id', 'item_category_id', 'type_code', 'lang', 'month_after_same_text', 'same_text_this_month', 'is_digital']]

# save the result file
target_file_path = 'Input/' + 'items_and_vectors.csv'
df.to_csv(target_file_path, encoding='utf-8', index=False)
