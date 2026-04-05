# Machine Learning Assignment - SVM, KNN, and Ensemble Methods

This repository contains solutions to a machine learning take-home assignment covering Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and ensemble methods including boosting and bagging techniques.

## Repository URL

[github.com:MrGladiator14/assignment33PM.git](https://github.com/MrGladiator14/assignment33PM.git)

## Project Structure

```
assignment33PM/
├── pyproject.toml                 # Project configuration and dependencies
├── uv.lock                        # Dependency lock file
├── README.md                      # This file
├── .python-version                # Python version specification
├── .gitignore                     # Git ignore rules
├── D33_PM_TakeHome_SVM_KNN_CS.docx.pdf  # Assignment document
├── partA.ipynb                    # Part A: SVM and KNN implementation
├── partB.ipynb                    # Part B: Boosting vs Bagging analysis
├── partC.md                       # Part C: Written responses
├── partD.ipynb                    # Part D: Advanced ensemble methods
└── catboost_info/                 # CatBoost training artifacts
```

## Dependencies

This project uses Python 3.13+ with the following main dependencies:

- **scikit-learn**: Machine learning algorithms (SVM, KNN)
- **catboost**: Gradient boosting library
- **lightgbm**: LightGBM gradient boosting framework
- **xgboost**: XGBoost gradient boosting framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **seaborn**: Data visualization
- **ipykernel**: Jupyter notebook support

## Setup

1. Install dependencies using uv:

   ```bash
   uv sync
   ```
2. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```
3. Run the main script:

   ```bash
   python main.py
   ```

## Assignment Parts

- **Part A**: Implementation and analysis of SVM and KNN classifiers
- **Part B**: Comparative study of boosting vs bagging ensemble methods
- **Part C**: Theoretical questions and written explanations
- **Part D**: Advanced ensemble techniques and performance optimization
