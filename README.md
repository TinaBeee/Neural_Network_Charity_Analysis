# Neural Network Analysis of charity investments
Using TensorFlow to design a neural network that uses binary classification to determine if a charity that receives funding will be successful or not 

## Resources
- Data: charity_data.csv 
- Software: Jupyter Notebook 6.3.0, scikit-learn, Plotly, TensorFlow and Pandas libraries

## Overview
Dataset includes investments in more than 34,000 organizations, including a binary column stating whether or not the money was used effectively. After cleaning the dataset and using sklearn's `OneHotEncoder()` to standardize the dataset, we're setting the outcome column as the target, with the remaining columns set as features to split the data into train and testing data. We then use sklearn's `StandardScaler()` and TensorFlow to determine accuracy using different nodes and activation functions.

## Data Processing
- The outcome column ('is_successful') was set as the target 
- The 'EIN' and 'name' columns were removed from the dataset, as they do not aid in analyzing the outcome
- The remaining nine columns are the features considered in the analysis

## Evaluating the model
Use of different combination of nodes, layers and activation functions to determine whether results can be optimized to achieve 75% accuracy.

- Original attempt included two layers, with 8 and 5 nodes, respectively. Activation function for both layers was relu, sigmoid for the output:

  <img width="563" alt="Screen Shot 2022-03-05 at 22 59 02" src="https://user-images.githubusercontent.com/90064437/156909873-b657b9b6-0802-4f6b-a9d4-458ba2395129.png">
  
- First optimization attempt included three layers with 20, 18, and 11 nodes and a combination of relu and tanh activation functions:

  <img width="549" alt="Screen Shot 2022-03-05 at 23 01 36" src="https://user-images.githubusercontent.com/90064437/156909924-ed7d8569-248a-4ef8-b3d8-39faa2e04cb8.png">

- Second optimization attempt included only two layers, using 9 and 6 nodes and the tanh function exclusively:

  <img width="547" alt="Screen Shot 2022-03-05 at 23 03 05" src="https://user-images.githubusercontent.com/90064437/156909950-f2091989-05c6-4634-84da-d4a58b192216.png">
  
- Third optimization attempt used three layers, including 7, 6, and 5 nodes, using a mix of relu and tanh:

  <img width="548" alt="Screen Shot 2022-03-05 at 23 04 41" src="https://user-images.githubusercontent.com/90064437/156909987-c7b463c3-1244-421b-b5d4-af6391d31fee.png">

- Fourth attempt used three layers with 25, 18 and 9 nodes each and relu activation:

  <img width="548" alt="Screen Shot 2022-03-05 at 23 05 52" src="https://user-images.githubusercontent.com/90064437/156910011-96248f6c-7802-4462-8323-9b6cfb75b1b2.png">


## Summary
None of the models were able to reach an accuracy score of 75%. Using different cut-off points for bucketing some of the feature data (increasing or decreasing the number of buckets) also did not change the outcome. The use of different activation functions similarly did not have an impact on the accuracy. 
