# Task
In this project I solve a kaggle challenge. Its brief description is available on the [kaggle site](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data).
>You are provided with daily historical sales data. The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.

This is a time-series dataset with some additional metadata information - brief test descriptions of every shop, product (they are called "items") and product category. Yes, this is real world sales data. As for the test set, there is a full list of items sold at least in one shop during that month and a full list of shops that worked at that time. The task is to predict real sales value per item per shop clipped to [0,20].

# Solution overview
The main idea of this solution is to aggregate data to get per-month totals from per-day data and for each month add rows with zero sales for items not sold in this shop, but sold in another shop the same month. Then it becomes possible to create features from sales during previous periods, like sales of each item in the same shop last month, average sales of the same item in all shops last month, average sales of items of the same category in the shop, average sales of an item for the last 3 or 6 months e.t.c. Now it becomes quite logical to cross-validate the model on the data from the last month of the train set - and this is enough to build a decently working model. 

I used a number of tricks to significantly improve the model's performance, and will briefly describe all of them here in the readme.

# Multiple models and a second level model
To begin with, I used not just the last month, but the last 3 months for cross-validation, as results for a single month very much depend on how well sales for a several new items are predicted.

And then, I built several models and then used a second-level model (linear regression) to get the final result. Cross-validation results for the three months are used as a training set for that second level model.
As was mentioned earlier, the task is to predict the actual sales clipped to [0,20]. Most items are sold rarely and have target values below 10, while some very popular items are sold much more often than the limit of 20 units. This means, that if we clip the target to 20 the model would have no examples for results higher than 20 and would predict the target value lower than 20 even in cases when it can be predicted for sure that the actual value is much higher. Clipping the target to a value much higher than 20 would lead to the model underperforming for most items that are unpopular [the metric is RMSE]. I choose to fit three models to target values clipped to [0,10], [0,25] and [0,50]. The second of these models produces decent predictions without two other models, but performance of the second-level model is better than of any single model.

# Data cleaning
Organizers intentionally left the data as is, so there is much space for data cleaning in the contest. 

Two first shops in the list can be easily recognized as duplicates.  Their names start with an exclamation mark, and then there is the same text description as for two other shops. There are also items with their descriptions beginning with exclamation mark symbols.

But most importantly, it turns out that there are two more "shops" that are not actual shops, but sales during some events. These "shops" never work for two months in a row, and they do not appear in the test set. Removing data for those "shops" from the train set increased the model's performance.

# Special features for new items
The most challenging task for this data set is to predict sales for new items that were never sold before. A few features with average sales for the new items only, per category and per shop, somewhat improved the situation. Another way to improve the model's prediction for new features is using text-based features, as the only thing we know about new items is their brief text description. All the features for new items were set to be always equal to zero for any item sold before, as the model performs much better on data for sales on previous months, and special features for new items do not improve predictions for other items.

# Special features for items sold second month
Prediction for second month sales significantly depend on a day when sales of a new item started previous month. If sales started close to the end of the month, data for sales last month would only consist of sales during the last few days, and that would lead to the model underestimating sales for the second month. Information on day, when sales started, is not present in aggregated per month data, but it is present in initial per day data. So, I've added a few features for second month sales, like "average sales per day in this shop" and "average sales per day in all shops", that significantly improved results.

# Trick to improve predictions for new shops
In the test set, there is one shop that is works second month after working for about two weeks previous month. Models underestimated sales in that shop as all the features for sales last month / last three months / last 6 months only had items sold for two weeks. So, for each shop and month, I've calculated the number of days that shop worked and divided all the "lag features" (made from sales in previous periods) by the actual number of days shop worked during that period.

# Code and files.
Solution itself consists of 4 files with python code:
* features_from_text.py - code to process text descriptions of items and categories and to prepare items_and_vectors.csv file with features made from text;
* feature_engineering.py - code to prepare features from the sales data and to join them with the features made from text;
* all_models_run.py - code to define and train all the first level models;
* second_level.py - code to build a second-level model and export the final prediction.

