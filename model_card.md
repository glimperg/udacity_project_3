# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created by Gido Limperg. This is a random forest classifier model (using the default sklearn hyperparameters).

## Intended Use
The model can be used to predict the salary range of a person, given certain attributes of that person.
The prediction is binary: either less than (or equal to) $50k, or more than $50k.

## Data
The model is trained on the [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income). 
The dataset consists of 32561 rows of data corresponding to various attributes of different people.
The data has been cleaned by removing all spaces in the csv file. The data was split into a train and test set using an 80-20 split.
Before training, a one hot encoder was applied to the categorical features, as well as a label binarizer on the labels.

## Metrics
The metrics used to evaluate the model are precision, recall and F1 score.
The overall model performance is the following (using random state 420):
    - Precision: 0.7276
    - Recall: 0.6272
    - F1: 0.6737

## Ethical Considerations
The training data is based on public census data from the USA. This is important to consider when using the model.

## Caveats and Recommendations
- The Census Income Dataset originates from 1994, so it's important to keep this in mind when using the model. After all, the income values might not be representative for current day examples (e.g. due to inflation).
- The default hyperparameters were used for the random forest classifier. In order to improve accuracy of the model, it's recommended to optimize the hyperparameters.