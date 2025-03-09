# Cardiac Arrhythmia Detection and Classification

## Introduction

Cardiac arrhythmia refers to any irregularity in the heart's rhythm. Some arrhythmias are so serious that they require immediate medical attention. Because analyzing ECG data involves sifting through large amounts of information, human error is a real risk. This can lead to misdiagnoses, which in turn might cause serious health complications or even legal issues for healthcare providers. By using computer-assisted analysis, we can improve the detection and classification of arrhythmia, ultimately supporting better clinical decisions.

Even with all the money spent on healthcare, the U.S. system still struggles with preventable errors and inconsistent diagnoses. Computer-based decision support systems offer a way to analyze patient data every day, reducing human error and improving overall diagnostic accuracy.

---

## Objective

In this project, we use supervised machine learning to analyze ECG features along with other physiological data to detect and classify arrhythmias. We developed two models:

1. **Model 1 (Detection):** Determines whether a patient is normal or shows signs of any arrhythmia.
2. **Model 2 (Classification):** Breaks down the arrhythmia cases into multiple specific classes.

Our work uses publicly available data from the UCI Machine Learning Repository, and we compare several statistical and machine learning approaches.

---

## Scope of Work

1. **Data Cleaning**  
   - Handle missing or invalid values.
   - Impute missing data (using the mean, for example).

2. **Feature Selection**  
   - Use the Boruta feature selection algorithm to pick out the most important predictors.

3. **Modeling with Various Algorithms**  
   - Apply a range of supervised learning methods (like Decision Trees, Random Forest, SVM, XGBoost, Neural Networks, etc.) for both detection and classification.

4. **Comparing the Results**  
   - Evaluate the models based on metrics such as accuracy, sensitivity, and specificity, and discuss the trade-offs.

---

## Approach

### Data Description

Our dataset consists of ECG readings and other physiological measurements from patients. Each patient is labeled as either **normal** or assigned to one of several arrhythmia classes.

- **Model 1:** All arrhythmia types are grouped into one category, making it a binary problem (normal vs. arrhythmia).
- **Model 2:** We perform a multi-class classification, distinguishing between different arrhythmia types and the normal condition.

### Why This Approach?

1. **Detection (Binary Classification):**  
   - We simply label patients as "normal" or "arrhythmia" to quickly flag any potential issues.
  
2. **Classification (Multi-Class):**  
   - We go a step further by identifying the specific type of arrhythmia, which can be crucial for making detailed treatment decisions.

#### Data Cleaning

- We replaced missing values with nulls and then imputed them using the mean.
- Uninformative columns (or those with no variance) were removed.
- For detection, classes 2–16 (as an example) were merged into one "arrhythmia" category.
- For classification, very small classes were either merged with similar groups or removed if they added too much noise.

#### Feature Selection

We used the Boruta algorithm to automatically select the features that provide the most useful information. This step helps reduce overfitting and makes our models easier to interpret.

---

## Implementation Details

We experimented with several algorithms for both tasks:

1. Linear Discriminant Analysis (LDA)
2. Random Forest
3. Support Vector Machine (SVM) – using both linear and RBF kernels
4. Decision Tree
5. XGBoost
6. K-Nearest Neighbors (KNN)
7. Logistic Regression (Lasso and Ridge) for the detection model
8. Gradient Boosting for the detection model
9. Neural Network for the detection model

Hyperparameter tuning was done using grid search and cross-validation to ensure we got the best performance possible.

---

### Model 1: Detection of Arrhythmia

For this binary classification task, our dataset includes:
- **Normal:** 245 samples  
- **Arrhythmia:** 207 samples  

Here’s a quick summary of how the models performed:

| **Algorithm**                    | **Best Hyperparameters**                           | **Confusion Matrix**         | **Accuracy** | **Comments**                                             |
|----------------------------------|----------------------------------------------------|------------------------------|--------------|----------------------------------------------------------|
| **Lasso Logistic Regression**    | C = 0.3594                                         | [[33, 19], [7, 54]]          | 0.77         | A balanced model with a few misclassifications.          |
| **Ridge Logistic Regression**    | C = 0.0464                                         | [[34, 18], [7, 54]]          | 0.78         | Slightly better than Lasso.                              |
| **KNN**                          | k = 7                                              | [[31, 21], [3, 58]]          | 0.79         | Shows good performance with fewer errors.                |
| **Random Forest**                | max_features = 7                                   | [[43, 9], [6, 55]]           | 0.87         | Best overall accuracy among our models.                  |
| **Gradient Boosting**            | learning_rate=0.01, max_depth=1, n_estimators=2500 | [[43, 9], [7, 54]]           | 0.86         | Almost as good as Random Forest.                         |
| **Decision Tree**                | max_depth = 8                                      | [[34, 18], [11, 50]]         | 0.74         | Works okay with pruning, but not as strong.              |
| **SVC (Linear)**                 | C = 0.05                                           | [[35, 17], [8, 53]]          | 0.78         | Good, though might benefit from adjusting the threshold. | 
| **Neural Network**               | alpha=0.1, hidden_layer_sizes=(7,)                 | [[42, 10], [42, 19]]         | 0.54         | Underperformed – might need more  data.                  |

**Observations:**  
The Random Forest model stood out with an accuracy of 0.87, while Gradient Boosting was close behind. The Neural Network, however, didn’t perform as well, suggesting that more additional data could be necessary.

---

### Model 2: Classification of Arrhythmia (Multi-Class)

In this task, we classify patients into multiple categories (for example, Class 0, Class 1, Class 2, Class 3, and Class 4). These classes represent different types of arrhythmias along with normal.

**Overall Highlights:**

- **XGBoost** delivered the best performance, with high sensitivity (ranging from 0.55 to 0.90) and solid balanced accuracy.
- **Random Forest** also achieved good results (around 0.85 accuracy), although some classes had lower recall.
- **SVM (RBF)** showed a high AUC of about 0.92 but had issues with sensitivity in certain classes.
- **Decision Trees** and **KNN** were more affected by data imbalance and sometimes overfit the data.

#### Example Metrics (Summarized)

| **Algorithm**     | **Accuracy** | **Key Observations**                                                                                          |
|-------------------|--------------|---------------------------------------------------------------------------------------------------------------|
| **LDA**           | ~0.85 AUC    | Tends to misclassify some arrhythmia cases as normal.                                                         |
| **Random Forest** | 0.85         | Overall good accuracy, though some classes have lower recall.                                                 |
| **SVM (RBF)**     | 0.92 AUC     | Strong AUC, but sensitivity varies between classes.                                                           |
| **Decision Tree** | ~0.76        | Moderate performance; overfitting can be an issue.                                                            |
| **XGBoost**       | **Best**     | Highest sensitivity and balanced accuracy; robust against imbalanced data.                                    |
| **KNN**           | ~0.73        | Non-parametric approach that sometimes misclassifies minority classes as normal.                              |

---

## Interpretation & Observations

- **Data Imbalance:**  
  Some arrhythmia classes have fewer examples, which can lead to them being misclassified as normal.

- **Tree-Based Models:**  
  Methods like Random Forest and XGBoost tend to perform better because they combine the results of multiple trees, reducing variance.

- **Feature Engineering:**  
  Including more domain-specific features (like wavelet transforms or measures of heart rate variability) might further improve performance.

---

## Conclusion

1. **Detection Model (Model 1):**  
   - The Random Forest model achieved the highest accuracy (0.87), making it a promising tool for quickly screening patients for arrhythmia.

2. **Classification Model (Model 2):**  
   - XGBoost performed best in the multi-class setting, but data imbalance remains a challenge.

3. **Clinical Relevance:**  
   - In a clinical setting, high sensitivity is crucial. It’s better to have some false positives than to miss a true arrhythmia case.

4. **Future Work:**  
   - We plan to collect more data for underrepresented classes, explore additional ensemble methods, and incorporate further clinical features to boost model reliability.

---

## References

- [UCI Machine Learning Repository: Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/arrhythmia)
- Research papers on ECG analysis and arrhythmia detection

---

## Acknowledgments

We thank the dataset providers for making their data publicly available and the open-source community for their invaluable tools and libraries.
