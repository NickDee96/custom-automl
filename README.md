# Twitter Success Predictor

This repository contains the code for a machine learning project aimed at predicting the performance of Twitter posts. The project is detailed in a series of articles that cover the development of the machine learning model, the construction of the training pipeline, the process of model tuning using Bayesian Optimization, and the visualization of the training pipeline.

## Articles

1. **Predicting Twitter Success: The Power of Machine Learning**: Introduces the project and explains the importance of predicting Twitter post performance for businesses and individuals. Provides an overview of the machine learning model and the features it considers.

2. **Behind the Scenes of a Machine Learning Pipeline: Training Multiple Models**: Focuses on the various models used in the project, the reasons for their selection, and their contribution to the overall project. Discusses the challenges encountered and the solutions implemented during the pipeline construction.

3. **Fine-Tuning Machine Learning Models with Bayesian Optimization**: Delves into the use of Bayesian Optimization for model tuning. Explains the concept of Bayesian Optimization, its application in model tuning, and how it enhances model performance.

4. **Visualizing the Journey: Insights from Our Machine Learning Training Pipeline**: Emphasizes the importance of visualization in understanding and communicating machine learning models. Showcases various visualizations from the project, explaining their significance and the insights they provide.

## Repository Structure

```
custom-Automl/
┣ data/
┃ ┗ twitter_news.csv
┣ output/
┃ ┗ metrics.csv
┣ data_loader.py
┗ trainer.py
```

## Files

- `data_loader.py`: Contains functions for loading and preparing the data for machine learning. This includes functions for loading data from a CSV file, calculating thresholds for engagement levels, assigning engagement levels, encoding text data into embeddings, encoding labels into integers, and splitting data into training and test sets.

- `trainer.py`: Contains functions for training and evaluating machine learning models. This includes functions for getting the size of a model, training and evaluating a model, and preparing data for machine learning. It also includes the main script for training and evaluating multiple models, and visualizing the results.

## Usage

To use this project, you can clone the repository and run the `trainer.py` script. This will train and evaluate multiple machine learning models on the Twitter data, and output the results to the `output/metrics.csv` file. It will also display a plot of the results.

Please note that you will need to install the required Python libraries, which include `pandas`, `sklearn`, `sentence_transformers`, `psutil`, `plotly`, `joblib`, `xgboost`, `lightgbm`, and `catboost`.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the terms of the MIT license.