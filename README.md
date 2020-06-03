# [Implementation of Naïve Bayes Classifier for Titanic Dataset](https://www.kaggle.com/samyukthagopalsamy/naive-bayes-classifier)
- One highly practical Bayesian learning method is the Naïve Bayes learner, often called the Naïve Bayes classifier.
- It is based on simplifying the assumption that the attribute values are conditionally independent given the target value. 
- There is no explicit search through the space of possible hypotheses. Instead, the hypothesis is formed without searching, simply by counting the frequency of various data combinations within the training examples.

- [Titanic dataset](https://www.kaggle.com/jamesleslie/titanic-cleaned-data) (cleaned) from Kaggle is chosen.
- Using [Naïve Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier), we try to predict which passengers survived the Titanic shipwreck.
- The dataset is split 80% as training dataset and 20% as test or validation set.
- The attributes having continuous numerical values are converted into categorical variables.
- The model is evaluated using accuracy, precision and recall metrics.

### Titanic dataset attributes used
- Age: age
- Sex: Sex (male, female)
- Fare: Passenger Fare (British pound) 
- Parch: Number of Parents/Children Aboard 
- Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
- SibSp: Number of Siblings/Spouses Aboard 
- Title: title of their name (Miss, Mr, Mrs, Master)
- Family_Size: size of the family.
- Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- Survival: Survived (0 = No; 1 = Yes) 

The Naïve Bayes classifier assigns either 0 (not survived) or 1 (survived) to the target value for each instance, based on the probability estimates learned from the training data. <br>The actual and predicted outcomes are appended to the output data frame.<br><br>
- Values of Confusion Matrix are calculated - True Positives(TP), True Negatives(TN), False Positives(FP), False Negatives(FN)<br>
- Accuracy: Percentage of test set tuples that are correctly classified.<br>
- Precision: Exactness – percentage of tuples that the classifier labeled as positive are actually positive.<br>
- Recall: Completeness – percentage of positive tuples that the classifier label as positive.<br>

