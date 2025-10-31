This project estimates the loan approval status via the majority voting amongst the implemented models.

# Models
1. KNN
2. Random Forest
3. Support Vector Machine
4. XGBoost
5. Naive Bayes
6. Neural Network

# Result

F1 Score on train data by tuned COMBINED MODEL Classifier : 81.4621409921671

F1 Score on test data by tuned COMBINED MODEL Classifier  : 84.070796460177

# Dataset

From [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset), the model is fed with background/demographic of the loan applicants and the loan approval status, and learns the underlying patterns between the independent variables and the dependent variables

| Label              |
| ------------------ |
| Loan_ID            |
| Gender             |
| Married            |
| Dependents         |
| Education          |
| Self_Employed      |
| ApplicantIncome    |
| CoapplicantIncome  |
| LoanAmount         |
| Loan_Amount_Term   |
| Credit_History     |
| Property_Area      |
| Loan_Status        |

# Techniques

## Data Preprocessing
1. Handle missing values

   - Fills in missing value with default value.
2. Manual Label Encoding

   - Converts data into numerical format.
3. Feature Selection using Mutual Information

   - Shows which independent variables(features) provide the most contributions towards the dependent(target) variable.
   - Then, select the ones with the greatest strength as the features to be fed to the model.
4. Standardization

   - Rescales continuous numerical data to a common range, improving model training speed (gradient descent rate).
   - Prevent features with larger scales from dominating
     > Income(1000 - 100,000) vs Age(0 - 100)
5. SMOTE(Synthetic Minority Oversampling Technique)

   - Oversampling is applied to the minority class in the training data set to handle class imbalance.
   - Random point of data is created(synthesized) for the minority class along the line of neighborhood.
   - Helps reduce overfitting, and improve model performance.
6. Check duplicates

   - Drops duplicates if there is.

## Modelling

1. Hyperparameter tuning via **GridSearchCV**

   - GridSearchCV optimizes model configuration by searching through a predefined set of hyperparameter combinations.
   - Improves model performance.
2. Combined probability
   - Using `Voting='soft'` to get average predicted probabilities instead of hard votes, which makes a tie less likely to happen.
