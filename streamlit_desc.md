We start with plotting frequency count for each rating in dataset. \
We can observe
- Ratings are disproptiontely distributed, having more 4, 5 ratings than 1,2

We are visualizing the number of reviews by age. \
\
While exploring the data, I found out that the range for age in dataset was  ```min = 18 to max = 99```. \
\
As we are looking at product reviews dataset, I decided to put in into buckets of age groups for easier and relevant analysis. \
\
We can observe that age data is ***normally distributed***

Here I have plotted number of reviews against two important variables - \
- Product Type 
- Age Groups

For counting positive and negative reviews, I used feature column - *ratings* available in the dataset. I used threshold to distinguish between negative and positive reviews. \
```ratings < 3 = Negative``` \
```ratings >= 3 = Positive```

This is an important analysis to observe average/median review length by each age group. \

We can see that generally the number is equivalent but for certain age groups there are more negative reviews than positive ones. 

Most important analysis of the dataset \
\
In this window we are visualiazing the ***Pain Points*** of the customers as well as ***What is working and What is Not?***

Word cloud gives us a quick yet informative picture of most commonly mentioned words in reviews. Ability to see Top Positive and Negative featured words by each Product **Type** gives us deeper insights for business teams to aspects they should primarilt focus upon.

If I had more time available, these are the additional tasks I would have implemented as part of my future plan - 
- Topic Modelling for product types
- Product ID level analysis (I have observed that each product category/type has same ProductID used by different age groups, so there is some scope for analysis here)
- More modularized and cleaner code