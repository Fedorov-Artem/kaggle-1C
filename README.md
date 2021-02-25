# coursera-1C
Task.
In the project I solve a kaggle challenge. Its brief description is available on the kaggle site (https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data).
This is a time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. The task is to forecast the total amount of products sold in every shop for the test set. The list of shops and products (they are called items) slightly changes every month.
Most of the items are CDs/DVDs with some information - either a computer game, movie, music, programs, etc.

Input data.
I have information about total sales in each shop per day from 01.2013 to 10.2015 and need to predict total sales for the next month, 11.2015. For the test period I have a list of items, each of them was sold at least once. I’m also provided with a list of shops that were selling in November 2015.
In the data set there are about 20 thousand items divided into about 80 item categories, and a list of about 60 different shops. I get a brief text description for each item, category and shop.
.
It is important to note that the task is to predict the target value clipped to [0,20].

Solution overview.
I am given per-day data and my task is to predict total sales for the whole month. So, first of all I aggregate data to get per-month totals and for each month add rows with zero sales for items not sold in this shop, but sold in another shop the same month. Then I build models capable of predicting sales for a given month using data from the previous periods. I build several such models to predict data for four periods and use all of them for cross-validation.
To achieve best possible results, I build several model types: for target values clipped to [0, 25], to [0, 10], and to [0, 50]. Then I use predictions made by these models as features and train a linear regression to make a final prediction.

Preparing features.
Data healing. For example, I remove from the dataset shops that constantly start and stop working, as I am not asked to predict sales for such shops and they only add noise to the models. Then, I use comments in brackets in the item text description to fix categories for a number of items, e.t.c.
I make most features from some sort of target aggregation from previous periods (months). I make features from item sales in the same shop, average item sales in all shops, average sales per category e.t.c.
Some features are time-based, like the number of months since the item was first sold.
Features from item name

Special features.
Probably the most challenging task for this data set is to predict sales for new items that were never sold before. The most problematic items are those of one of computer games categories. Such items could have either very high sales in case of a major new release or very low sales in case of a new edition of an old game. I made a feature with average sales for the new items only, per category and per shop, improved the situation, but does not solve the problem. Text-based features helped me to tackle the issue, as I have text description of new items. Major releases usually include several items with similar names, as they include versions for different platforms. I made a feature “same_text_this_month” that significantly decreased error for the new computer games.
Another special feature is for the second month. Sales for the second month significantly depend on the day when sales of a new item began last month. If they began close to the end of the month, first month sales would be low and second month sales would be high. So, I made two features “first_day” - for the first day when item was sold this shop and “average_per_day_after_start” for average sales per day after the item was first time sold in each shop. These two features significantly decreased the loss function for second month sales.


Code and files.
Solution itself consists of 4 files with python code:
features_from_text.py - this code should be run first, it prepares the items_and_vectors.csv file. Text description of items and categories are only processed in here;
feature_engineering.py - this file should be run after items_and_vectors.csv file is prepared. Here the sales data itself is processed and most features are prepared;
all_models_run.py - code to define and train all the first level models. Model predictions are saved in *.csv files, cat_25 model predictions are already good, but I run a second-level model to get even better results;
second_level.py - code to build a second-level model and export the final prediction.

