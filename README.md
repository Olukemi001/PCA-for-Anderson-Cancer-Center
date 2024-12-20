## Project Overview

This project applies **Principal Component Analysis (PCA)** on the breast cancer dataset from **sklearn.datasets** to reduce dimensionality and identify key features for addressing the growing number of referrals at the Anderson Cancer Center. The dataset is processed, reduced to two principal components, and a logistic regression model is implemented to predict the classification. The results are then visualized for better insights.

The project is modular, with separate modules handling specific tasks, including PCA processing, logistic regression modeling, and data visualization. Each module can be modified or extended independently, ensuring scalability and flexibility.

## Folder Structure

Here is the structure of the project folder:

```
PCA_for_Anderson_Cancer_Center/
│
├── pca_processor.ipynb           # Module for PCA implementation
├── logistic_regression_model.ipynb  # Module for logistic regression model
├── pca_visualizer.ipynb           # Module for PCA visualization
├── main.ipynb                   # Main module that integrates all components
├── conversion_module.ipynb       # Converts Jupyter notebook to Python script
├── README.md                  # This README file
│
```

## Installation Requirements

To run the project in a Jupyter notebook, ensure you have the following Python libraries installed:

- `sklearn`
- `pandas`
- `matplotlib`
- `numpy`


## How to Run the Project in Jupyter Notebook

1. **Clone the repository** or **download the project files** to your local machine.
2. Open a Jupyter notebook and navigate to the project folder.
3. **Execute the modules** in the following order:
    - `pca_processor.ipynb`: Contains PCA logic for loading and processing the dataset.
    - `logistic_regression_model.ipynb`: Contains the logistic regression implementation for prediction.
    - `pca_visualizer.ipynb`: Contains the logic to visualize PCA components.
    - `main.ipynb`: Integrates all modules and runs the entire process.
    - `conversion_module.ipynb `: This module is used to convert a Jupyter notebook into a Python script. It iterates over notebooks and saves them as .py files.
4. The results will be displayed in the notebook or Python console, including accuracy and visualizations.

### Example of Running in Jupyter Notebook

```python
# Step 1: Run the PCA processor module
from pca_processor import PCAProcessor
pca_processor = PCAProcessor(n_components=2)
data, target = pca_processor.load_data()
pca_df = pca_processor.apply_pca(data)

# Step 2: Run the Logistic Regression Model
from logistic_regression_model import LogisticRegressionModel
log_reg_model = LogisticRegressionModel()
accuracy = log_reg_model.train_and_evaluate(pca_df, target)

# Step 3: Visualize PCA Components
from pca_visualizer import PCAVisualizer
PCAVisualizer.plot_pca(pca_df, target)
```

## Code Explanation

### Module 1: PCA Processor (`pca_processor.ipynb`)

This module is responsible for loading the breast cancer dataset, scaling the data, and applying PCA to reduce the dimensions.

- **`load_data()`**: Loads the dataset and returns the features and target values.
- **`apply_pca(data)`**: Scales the data and applies PCA to reduce it to the specified number of components (default is 2).

### Module 2: Logistic Regression Model (`logistic_regression_model.ipynb`)

This module implements logistic regression to train and evaluate a classification model on the PCA-reduced data.

- **`train_and_evaluate(X, y)`**: Splits the data into training and testing sets, trains the logistic regression model, and evaluates its accuracy.

### Module 3: PCA Visualizer (`pca_visualizer.ipynb`)

This module visualizes the PCA components in a 2D scatter plot.

- **`plot_pca(pca_df, target)`**: Plots the first two PCA components and uses color encoding based on the target (malignant or benign).

### Module 4: Main Module (`main.ipynb`)

This module integrates the PCA processor, logistic regression model, and PCA visualizer into a single workflow. It performs the following steps:
1. Loads the dataset.
2. Applies PCA.
3. Trains a logistic regression model.
4. Visualizes the PCA components.

### Conversion Module (`conversion_module.ipynb`)

This module is used to convert a Jupyter notebook into a Python script. It iterates over notebooks and saves them as `.py` files.

## Error Handling

All modules include error handling using `try-except` blocks to catch potential issues. Some common errors include:

1. **Data Loading Errors**: If the dataset cannot be loaded, the function will print an error message and return `None`.
2. **PCA Errors**: If PCA fails (e.g., due to insufficient components or invalid data), an error message is printed and `None` is returned.
3. **Model Training Errors**: If logistic regression encounters issues (e.g., convergence problems or incorrect data), it will catch the error and return `None`.

### Possible Errors and Solutions

1. **FileNotFoundError**: If the dataset file is not found, ensure the file path is correct or that the data is available.
   - Solution: Verify dataset availability or use sklearn’s `load_breast_cancer()` function, which loads the dataset directly.

2. **ValueError during PCA**: If the number of components is greater than the available features, PCA will fail.
   - Solution: Set the number of components to 2 as shown in the example or reduce the `n_components` to match the features.

3. **Logistic Regression Convergence Warning**: If the logistic regression model doesn’t converge, increase the number of iterations in `LogisticRegression(max_iter=10000)` or adjust the data preprocessing.

4. **Plotting Errors**: If matplotlib cannot generate the plot, ensure the environment supports plotting (e.g., Jupyter notebook or a GUI-based IDE).


## Conclusion

By using modular components, this project allows for flexibility and easy adaptation to other datasets or algorithms. The workflow can be extended with other machine learning models or feature selection techniques. This project provides a comprehensive example of how PCA can be applied to a real-world problem, and it is a useful tool for data analysis and machine learning in healthcare and research fields. 

