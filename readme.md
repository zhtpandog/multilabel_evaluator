# Multilabel Classification Evaluator #
This module offers a variety of multilabel evaluation metrics with simple input.  
It is tested in Python 2.7 environment.  
## Usage ##
```
from MLB_Evaluation import evaluation
evaluation(truth1, pred1, truth2, pred2, save=False, saveid='mlb0')
```
## Input ##
__truth1__: A list of list of string. Each list of string represents a set of correct labels of that record. Example:  
```
[
[lbl1, lbl5],
[lbl2, lbl3, lbl7],
...
]
```
__pred1__: A list of list of list. The outer list has each item as a record. The inner list has dimension you set to be with each item as a binary list of [label, proba]. Sorted by proba from high to low. Example:  
```
[
[[lbl1, 0.8], [lbl2, 0.2], ...[lbl9, 0.1]], 
[[lbl1, 0.9], [lbl2, 0.7], ...[lbl9, 0.4]], 
...
]
```
__truth2__: A list of list of binary integer (0/1). Each position in the inner list correponds to a label. A 1 meaning containing that label and a 0 means not. Example:  
```
[
[1, 0, 0, ..., 0, 1],
[0, 1, 1, ..., 0, 0],
...
]
```
__pred2__: A list of list of float numbers. Each position in the inner list corresponds to a label. Each float entry is a proba for that label. Example:  
```
[
[0.8, 0.2, 0.5, ..., 0.1, 0.4],
[0.9, 0.7, 0.1, ..., 0.4, 0.6],
...
]
```
__save__: Save the evaluation result to file or not?  
__saveid__: If save, what's the file name?  

Caution:  
1. If using __truth1__ and __pred1__, then __truth2__ and __pred2__ should be None, and vise versa.  
2) __saveid__ only used when __save__ is True.  

## Ouput ##
Evaluation results in a format you-can-understand-even-I-am-too-lazy-to-explain-here.  

## What's inside ##
Average NDCG Score  
Coverage Error  
Label Ranking Average Precision Score  
Label Ranking Loss  
Recall @ k  










