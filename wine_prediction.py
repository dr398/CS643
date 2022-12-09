
 # Programming Assignment 2 
 # Course : CS643 Fall 2022
 # Student : Dhruval Rana
 
"Spark application to train and tune model using wine prediction data"

import sys

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def data_cleaning(df):
    # cleaning header 
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

    

"main function for the application"
if __name__ == "__main__":
    
    # Creates the  spark application
    spark = SparkSession.builder \
        .appName('wine_prediction_cs643') \
        .getOrCreate()

    # creates the spark context to report logging info
    sc = spark.sparkContext
          sc.setLogLevel('ERROR')

    # Load and analyze the data file into an RDD of LabeledPoint.
    if len(sys.argv) > 3:
        print("Usage: pyspark_wine_training.py <input_file>  <valid_path> <s3_output_bucketname>", file=sys.stderr)
             sys.exit(-1)
    elif len(sys.argv) == 3:
         input_path = sys.argv[1]
         valid_path = sys.argv[2]
         output_path = sys.argv[3] + "testmodel.model"
    else:
          input_path = "s3://cs643project2/TrainingDataset.csv"
          output_path="s3://cs643project2/testmodel.model"

  
    
    train_data_set = data_cleaning(df)

    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(valid_path))
    
    valid_data_set = data_cleaning(df)



    # removes column not adding much value to prediction
    # removed 'residual sugar','free sulfur dioxide',  'pH',
    required_features = ['fixed acidity','volatile acidity','citric acid','chlorides','total sulfur dioxide','density','sulphates','alcohol',]
    
    # creates vector column name feature by only required_features list columns
    assembler = VectorAssembler(inputCols=required_features, outputCol='features')
    
    # creates classification with the given input values 
    indexer = StringIndexer(inputCol="quality", outputCol="label")

    # splits given data for training and testing
    # caches data so that it can be faster to use
                 train_data_set.cache()
                 valid_data_set.cache()
    
    # Chooses RandomForestClassifier for training
    rf = RandomForestClassifier(labelCol='label', 
         featuresCol='features',
         numTrees=150,
         maxBins=8, 
         maxDepth=15,
         seed=150,
         impurity='gini')
    
    # prints accuracy of above model
    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score = ', metrics.weightedFMeasure())

    
    # Retrain the model on mutiple parameters 
    cvmodel = None
    paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxBins, [9, 8, 4])\
    .addGrid(rf.maxDepth, [25, 6 , 9])\
    .addGrid(rf.numTrees, [500, 50, 150])\
    .addGrid(rf.minInstancesPerNode, [6])\
    .addGrid(rf.seed, [100, 200, 5043, 1000])\
    .addGrid(rf.impurity, ["entropy","gini"])\
    .build()
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                        
  
    cvmodel = crossval.fit(train_data_set)
    
               #saves the best model to new param `model` 
               model = cvmodel.bestModel
               print(model)
               # prints accuracy of best model
               predictions = model.transform(valid_data_set)
               results = predictions.select(['prediction', 'label'])
               accuracy = evaluator.evaluate(predictions)
               print('Test Accuracy1 = ', accuracy)
               metrics = MulticlassMetrics(results.rdd.map(tuple))
               print('Weighted f1 score = ', metrics.weightedFMeasure())

               # saves the best model to s3
               model_path =output_path
               model.write().overwrite().save(model_path)
               sys.exit(0)
