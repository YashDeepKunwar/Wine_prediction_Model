# Wine Quality Prediction

Random Forest classifier on the UCI Red Wine dataset — with a documented investigation 
into class imbalance, outlier-removal bias, and the accuracy-vs-recall tradeoff.

## What this project covers

This isn't just "train a model, report accuracy." The real value of this project 
is in the investigation:

- Compared six algorithms (Logistic Regression, Decision Tree, Random Forest, 
  Gradient Boosting, SVM, and a neural network) using cross-validation and 
  paired t-tests, not just a single train/test split.
- Found that outlier removal (IsolationForest) was disproportionately stripping 
  out rare quality categories — fixed by running outlier detection per-category 
  instead of across the whole dataset.
- Discovered that several models with the highest raw accuracy (~86-88%) were 
  achieving that by never predicting the "poor" quality class at all — just 
  defaulting to the majority class. Diagnosed this with confusion matrices, 
  not just accuracy.
- Tested `class_weight` rebalancing and SMOTE to address this, and documented 
  the real tradeoff: aggressive rebalancing fixed minority-class detection but 
  tanked overall accuracy; a tuned middle ground (custom class weights + 
  GridSearchCV) gave the best practical balance.
- Verified the final model's feature importances against real winemaking 
  chemistry (alcohol, volatile acidity, sulphates) rather than just trusting 
  the numbers.

## Final model

Random Forest, tuned with `GridSearchCV`, custom class weighting for the 
minority "poor quality" class. ~86% accuracy on held-out test data, with a 
documented, honest limitation: the dataset only contains 59 poor-quality wine 
samples, which caps how well any model can learn that class regardless of 
technique.

## Pipeline

1. Load and explore the data (distributions, correlations, feature relationships)
2. Bin `quality` into three categories: poor (<5), average (5–6), great (≥7)
3. Remove outliers per-category with `IsolationForest` (to avoid disproportionately 
   removing rare classes)
4. Split into train/test, scale features where needed
5. Compare six classifiers with cross-validation + significance testing
6. Diagnose misleading high-accuracy results via confusion matrices
7. Address class imbalance with weighted classes / SMOTE
8. Tune the final model with `GridSearchCV`
9. Evaluate on held-out test data and inspect feature importances

## Tech

Python, pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib

## Dataset

[UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) (red wine)
