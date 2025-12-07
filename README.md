# Wine Quality Analysis

This notebook analyzes a red wine quality dataset (`WineQT.csv`) to predict and classify wine quality based on its physicochemical properties. 

## Dataset

The `WineQT.csv` dataset contains various wine properties and a quality score for various types of the Portuguese wine type "Vinho Verde". The data can be found on Kaggle (https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download), as well as on the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/186/wine+quality).

## Key Functions

1.  **`data_loading_cleaning(filename)`**: Loads `WineQT.csv`, handles missing values, and renames columns.
2.  **`choose_best_ccp_alpha(X, y, ...)`**: Helps find the optimal pruning parameter (`ccp_alpha`) for Decision Trees using cross-validation.
3.  **`analyze_wine_quality_models(data_df, ...)`**: The main analysis function. It transforms the continuous `quality` score into a binary `Besorolas` ('Rossz' / 'Jó'), splits data, trains and visualizes Decision Trees, trains Random Forests, and evaluates models using cross-validation.

## How to Run

1.  Install necessary libraries (e.g., `pyearth`, `pandas`, `scikit-learn`).
2.  Place `WineQT.csv` in the notebook directory.
3.  Execute cells sequentially, starting with the `if __name__ == "__main__":` block at the end.

Expected runtime is around 31 sec
