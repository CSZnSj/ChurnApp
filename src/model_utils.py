# model_utils.py

import sys
sys.path.append('/home/sajjad/Projects/ChurnApp')


from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel, GBTClassificationModel
from pyspark.ml import Estimator, Model
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from typing import Any, Dict
from src.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

def get_classifier(
        model_name: str, 
        feature_col: str = "FEATURES",
        label_col: str = "label") -> Estimator:
    """
    Returns the classifier based on the given model name.

    Args:
        model_name (str): Name of the model ('random_forest', 'logistic_regression', 'gbt').
        feature_col (str): The name of the feature column.
        label_col (str): The name of the label column.

    Returns:
        Estimator: Initialized classifier of the specified type.

    Raises:
        ValueError: If the model name is not recognized.
    """
    classifiers = {
        "rf": RandomForestClassifier(featuresCol=feature_col, labelCol=label_col),
        "lr": LogisticRegression(featuresCol=feature_col, labelCol=label_col),
        "gbt": GBTClassifier(featuresCol=feature_col, labelCol=label_col)
    }

    if model_name not in classifiers:
        logger.error(f"Model '{model_name}' is not supported. Choose from {list(classifiers.keys())}")
        raise

    logger.info(f"Returning classifier for model '{model_name}'")

    return classifiers[model_name]

def get_param_grid(
        model_name: str) -> ParamGridBuilder:
    """
    Returns a parameter grid for hyperparameter tuning based on the provided model name.

    Args:
        model_name (str): The name of the model ('random_forest', 'logistic_regression', 'gbt').

    Returns:
        ParamGridBuilder: The grid with hyperparameters to be tuned.

    Raises:
        ValueError: If the model name is not supported.
    """
    param_grids = {
        "rf": ParamGridBuilder() \
            .addGrid(RandomForestClassifier().numTrees, [10, 20, 50]) \
            .addGrid(RandomForestClassifier().maxDepth, [5, 10, 20]) \
            .build(),

        "lr": ParamGridBuilder() \
            .addGrid(LogisticRegression().regParam, [0.01, 0.1, 0.5]) \
            .addGrid(LogisticRegression().elasticNetParam, [0.0, 0.5, 1.0]) \
            .build(),

        "gbt": ParamGridBuilder() \
            .addGrid(GBTClassifier().maxIter, [10, 20, 50]) \
            .addGrid(GBTClassifier().maxDepth, [5, 10, 15]) \
            .build()
    }

    if model_name not in param_grids:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(param_grids.keys())}.")
    
    logger.info(f"Returning param_grid for model '{model_name}'")

    return param_grids[model_name]

def get_evaluator(
        metric_name: str, 
        label_col: str = "label") -> MulticlassClassificationEvaluator:
    """
    Returns an evaluator based on the provided metric name.

    Args:
        metric_name (str): The metric to evaluate the model ('accuracy', 'f1', 'precision', 'recall', 'auc').
        label_col (str): The name of the label column.

    Returns:
        object: Initialized evaluator for the specified metric.

    Raises:
        ValueError: If the metric name is not supported.
    """
    evaluators = {
        "accuracy": MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy"),
        "f1": MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1"),
        "precision": MulticlassClassificationEvaluator(labelCol=label_col, metricName="weightedPrecision"),
        "recall": MulticlassClassificationEvaluator(labelCol=label_col, metricName="weightedRecall"),
        "auc": BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
    }

    if metric_name not in evaluators:
        logger.error(f"Metric '{metric_name}' is not supported. Choose from {list(evaluators.keys())}.")
        raise

    logger.info(f"Returning {metric_name} evaluator")

    return evaluators[metric_name]

def train_model(
        model: Estimator, 
        df: DataFrame,
        evaluator: MulticlassClassificationEvaluator, 
        param_grid: Dict) -> Any:
    """
    Train any model using cross-validation.

    Args:
        model (Estimator): The classifier or regressor model to be trained.
        df (DataFrame): Input training DataFrame containing features and labels.
        evaluator (MulticlassClassificationEvaluator): The evaluator used to evaluate the model's performance based on the specified metric
        param_grid (Dict): A dictionary of hyperparameters for cross-validation.
        feature_col (str): The column name containing feature vectors.

    Returns:
        CrossValidatorModel: Trained model with cross-validation.

    Raises:
        Exception: If an error occurs during model training.
    """
    try:
        # Set up cross-validation
        logger.info("Setting up cross-validation.")
        crossval = CrossValidator(estimator=model,
                                  estimatorParamMaps=param_grid,
                                  evaluator=evaluator,
                                  numFolds=3,)

        # Train the model with cross-validation
        logger.info(f"Training the {model.__class__.__name__} with cross-validation.")
        model_cv = crossval.fit(df)
        logger.info("Model training completed successfully.")
        return model_cv
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def evaluate_model(
        model: Estimator, 
        df: DataFrame,
        evaluator: MulticlassClassificationEvaluator, 
        metric_name: str = "f1") -> float:
    """
    Evaluate the trained model on a given dataset.

    Args:
        model (CrossValidatorModel): Trained model.
        df (DataFrame): DataFrame to evaluate the model on.
        evaluator (MulticlassClassificationEvaluator): The evaluator used to evaluate the model's performance based on the specified metric
        metric_name (str): The evaluation metric to use (default is "f1").

    Returns:
        float: Evaluation score of the model.

    Raises:
        Exception: If an error occurs during model evaluation.
    """
    try:
        logger.info("Evaluating model performance.")
        predictions = model.transform(df)
        score = evaluator.evaluate(predictions)
        logger.info(f"Model evaluation completed. {metric_name.capitalize()} Score: {score:.4f}")
        return score
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

def save_model(model, output_path: str):
    """
    Save the trained model to the specified path.

    Args:
        model (CrossValidatorModel): Trained model.
        output_path (str): Path to save the model.

    Raises:
        Exception: If an error occurs during model saving.
    """
    try:
        logger.info(f"Saving the model to {output_path}.")
        model.bestModel.save(output_path) # model.bestModel.write().overwrite().save(output_path)
        logger.info(f"Model successfully saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(model_name: str, model_path: str) -> Model:
    """
    Load a trained model based on the provided model name.

    Args:
        model_name (str): The name of the model ('random_forest', 'logistic_regression', 'gbt').
        model_path (str): The path to the saved model.

    Returns:
        Model: The loaded model object.

    Raises:
        ValueError: If the model name is not supported.
        Exception: If there is an error during model loading.
    """
    try:
        logger.info(f"Loading model '{model_name}' from {model_path}")
        
        model_loaders = {
            "rf": RandomForestClassificationModel.load,
            "lr": LogisticRegressionModel.load,
            "gbt": GBTClassificationModel.load
        }
        
        if model_name not in model_loaders:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(model_loaders.keys())}.")
        
        # Load the model using the appropriate loader
        model = model_loaders[model_name](model_path)
        logger.info(f"Model '{model_name}' loaded successfully from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Error loading model '{model_name}' from {model_path}: {str(e)}")
        raise
