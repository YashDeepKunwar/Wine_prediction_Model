# Wine Quality Prediction

A classification project on the UCI Red Wine Quality dataset — predicting whether 
a wine is poor, average, or great quality based on its chemical properties.

## What this project actually covers

This isn't just "train a model, report accuracy." The real value of this project 
is in the investigation:

- Compared six algorithms (Logistic Regression, Decision Tree, Random Forest, 
  Gradient Boosting, SVM, and a neural network) using cross-validation and 
  paired t-tests, not just a single train/test split.
- Found that outlier removal (IsolationForest) was disproportionately stripping 
  out rare quality categories — fixed by running outlier detection per-category 
  instead of across the whole dataset.
- Discovered that several models with the highest raw accuracy (~86-88%) were 
  doing so by never predicting the "poor" quality class at all — just defaulting 
  to the majority class. Diagnosed this with confusion matrices, not just accuracy.
- Tested class_weight rebalancing and SMOTE to address this, and documented the 
  real tradeoff: aggressive rebalancing fixed minority-class detection but tanked 
  overall accuracy; a tuned middle ground (custom class weights + GridSearchCV) 
  gave the best practical balance.
- Verified the final model's feature importances against real winemaking chemistry 
  (alcohol, volatile acidity, sulphates) rather than just trusting the numbers.

## Final model
Random Forest, tuned with GridSearchCV, custom class weighting for the minority 
"poor quality" class. ~86% accuracy on held-out test data, with a documented, 
honest limitation: the dataset only contains 59 poor-quality wine samples, which 
caps how well any model can learn that class regardless of technique.

## Tech
Python, pandas, scikit-learn, seaborn/matplotlib

## Dataset
[UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) (red wine)
