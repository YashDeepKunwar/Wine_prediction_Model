Ended up locking in Logistic Regression as the final model after running GridSearchCV, with custom class weights after way too much trial and error.

Basically tested six models—LogReg, SVM, MLP, Gradient Boosting, RF, and a basic Decision Tree. Ran full cross-val and significance testing on all of them. Decision Tree was clearly the worst of the bunch (p < 0.05 across the board), but the other four were basically dead even on raw accuracy. Statistically, there was no real difference between LogReg, RF, MLP, and SVC.

The confusion matrix was what actually broke the tie, not accuracy. SVC and default RF both had great looking ~86-88% accuracy scores, but when you look under the hood they were basically cheating. Since "average" wine makes up like 80% of the dataset, the models just got lazy and predicted average almost every single time. High score, but zero poor wines detected—so pretty much useless in practice.

A few things I tried to fix it:

* Outlier removal backfired at first. IsolationForest was aggressively stripping out poor and great wines just because they're rare by nature. Ended up fixing that by running outlier filtering per-category instead of dumping the whole dataset into it at once.
* Slapping `class_weight='balanced'` on fixed the poor wine detection (caught like 8/10), but completely tanked overall accuracy down to ~61%. It got way too paranoid and started guessing poor/great everywhere, messing up a ton of normal average wines.
* Tried SMOTE next, but it gave pretty much the same result as class weighting. Stacking them together didn't help either—mostly just redundant.
* Settled on custom class weights (~3-4x on poor, 2x on great). It's a compromise, but it keeps overall accuracy around ~70% while still actually flagging poor wines instead of pretending they dont exist.

Feature importance (and LogReg coefficients, checked those too) both pointed to alcohol, volatile acidity, and sulphates as the big three. Makes complete sense chemically—alcohol tracks ripeness/body, volatile acidity is that sour vinegar defect, and sulphates act as preservatives.

Real talk on limitations though: there are only 59 poor-quality wines in the entire dataset. Every single rebalancing trick—SMOTE, manual weights, combos—hit the exact same wall at around 70-80% recall on the poor class. At this point it’s pretty clearly a data scarcity problem, not a modeling issue. Swapping algorithms wont fix it; we just need more bad wine samples.
