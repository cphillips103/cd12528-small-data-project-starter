## Small Data Solutions
## Christopher Phillips
## November 8, 2023
## For Udacity Small Data Program

**Part 1: Travel Planner Problem**
A travel planning company asks their customers to share pictures of past vacations / holidays so their staff can get a sense of what kind of trips they enjoy. The company offers three basic categories of trips:
- Exploring in the Forest
- Adventure in the Desert
- Relaxing on the Beach 

As part of a new online trip planning software, the company is creating an AI bot that will automatically figure out from the uploaded photos which category is likely to be most appealing to the customer.  The challenge is, the company has fewer than 500 photos that are categorized and they feel it will be difficult to train a model using that little data. 

You will use the starter code and use transfer learning to build a model to categorize images with at least a 90% accuracy rate. 

Credit: Udacity.com


**Conclusions:**

Training with 30 Epochs:

Highest Accuracy 0.9333

![Base Results](https://github.com/[username]/[reponame]/blob/[branch]/travel_plan_30_epochs.jpg?raw=true)

Training with 200 Epochs:

Highest Accuracy 0.9333

![Augmented Results](https://github.com/[username]/[reponame]/blob/[branch]/travel_plan_200_epochs.jpg?raw=true)

**Part 2: Loan Funding Prediction Problem**
A loan company has a fairly large dataset that they want to use to train a model that predicts whether or not a loan should be funded.  The problem they face is the dataset they are using has a large class imbalance... they don't have enough examples of loans that were denied. The company wants to augment the "denied loan" class of data in order to train a model that performs better.  

You will use the starter code and use a Variational Autoencoder to generate an additional set of synthetic data for just the denied loan category of data. You will then augment the original dataset with the additional synthetic data and test to see if there were improvements.  

Credit: Udacity.com

**Conclusions:**

Before Augmented Data:

Macro Accuracy 0.48

![Base Results](https://github.com/[username]/[reponame]/blob/[branch]/loan_status_base.jpg?raw=true)


After Augmented Data:

Macro Accuracy 0.9333

![Augmented Results](https://github.com/[username]/[reponame]/blob/[branch]/loan_status_augmented.jpg?raw=true)

## Getting Started

1. Clone the repository
1. Install the dependencies listed below

### Dependencies

```
Python 3.10.8
Pytorch 1.13.1
torchvision 0.14.1
Numpy 1.24.1
sklearn 1.2.1
```

## Testing

**Part 1: Travel Planner Problem**
There is a test_model function included in TestModel.py. The code is already in place in the transfer_learning.py file to run this, you simply need to provide the correct parameters: a DataLoader for the test data, a trained model, and a list of class names. An image viewer will show the test images and the predicted class.  You should expect at least 9/10 images to be correctly identified.  

**Part 2: Loan Funding Prediction Problem** 
There is a test_model function included in TestModel.py. The function takes a path where the augmented dataset should be placed.  The script produces precision, recall, and an F1-score for both classes of data. You should expect results similar to this:
               
                precision    recall  f1-score   support
           0       0.79      0.55      0.65     61222
           1       0.63      0.84      0.72     56241 


## Project Instructions

**Part 1: Travel Planner Problem**
1. Create 2 Transforms (Train and Validation)
1. Create 2 DataLoaders (Train and Validation)
1. Import the VGG16 model 
1. Freeze the existing layers in the model
1. Replace the classification layer with your own
1. Using the provided train_model function, train a new model with the appropriate criterion, optimizer, and scheduler.
1. Using the set of images labeled "new" categorize them and see how your model performs

**Part 2: Loan Funding Prediction Problem**
1. Load the file "loan_continuous.csv" and split out just the records with Loan Status = 1
1. Use the test_model method to baseline the precision, recall and F1 score of the loan_continuous.csv file
1. Using the provided DataBuilder class, create Datasets for training and validation 
1. Using the provided Autoencoder class and the provided CustomLoss class, write methods to train and validate the model
1. After training the model, use the generate_fake method to generate 50000 additional records
1. Combine the generated data with the original dataset 
1. Use the test_model method to test whether the dataset augmented with synthetic data performs better on precision, recall, and F1 score for both loan statuses
