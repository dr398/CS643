
 # Programming Assignment 2 
 # Course : CS643 Fall 2022
 # Student : Dhruval Rana
 
"Spark application to run tuned model with testfile"

import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
"main function for application"
if __name__ == "__main__":
    
    # Creates spark application
    spark = SparkSession.builder 
        .getOrCreate()

    # Loads and analyzes the data file into an RDD of LabeledPoint.
    if len(sys.argv) > 3:
        print("Usage: wine_test_data_prediction.py <input_data_file> <model_path>", file=sys.stderr)
        sys.exit(-1)
    elif len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if not("/" in input_path):
            input_path = "data/csv/" + input_path
              model_path="/code/data/model/testdata.model"
              print("----Input file for test data is---")
              print(input_path)
    else:
              current_dir = os.getcwd() 
        print(current_dir)
              input_path = os.path.join(current_dir, "data/csv/testdata.csv")
              model_path= os.path.join(current_dir, "data/model/testdata.model")

    # reads csv file in DataFram 
    df = (spark.read.format("csv").option('header', 'true').option("sep", ";").option("inferschema",'true').load(input_path)) 
    df1 = data_cleaning(df)
    # Splits the data into training and test sets (30% held out for testing)
    required_features = ['fixed acidity','volatile acidity','citric acid','chlorides','total sulfur dioxide','density','sulphates','alcohol',]
    predictions = rf.transform(df1)
    print(predictions.show(5))
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')

    # prints the accuracy of above model
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score = ', metrics.weightedFMeasure())
    sys.exit(0)
