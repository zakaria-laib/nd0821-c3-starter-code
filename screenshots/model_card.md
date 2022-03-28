# Model Card - Census Bureau Income

## Model Details
* The Census Bureau classifier model for prediction task is to 
determine whether a person makes over 50K a year.
* Developed by Zakaria Laib 2022

## Intended Use
* Prediction task is to determine whether a person makes over 50K a year in real-time inference.

## Training Data
* Census dataset :  0.7 of the dataset

## Evaluation Data
* Census dataset:  0.3 of the dataset

## Metrics
* Evaluation metrics include : 
  * precision: 0.7136539524599226
  * recall: 0.5894977168949772
  * fbeta: 0.6456614153538386

## Ethical Considerations
*Certain features used could be deemed discriminatory (race, education level).

## Caveats and Recommendations
* Use more data for training for better generalization.
* Use more complex hyperparameters optimization. 
