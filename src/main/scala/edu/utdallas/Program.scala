package edu.utdallas

import org.apache.log4j.{ Logger, Level }
import scala.collection.mutable
import org.apache.spark.sql.{ SparkSession }
import org.apache.spark.sql.types.StringType
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.{ Pipeline, PipelineStage, PipelineModel }
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{ StringIndexer, VectorAssembler, OneHotEncoderEstimator }
import org.apache.spark.ml.tuning.{ CrossValidator, CrossValidatorModel, ParamGridBuilder }
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
  
object Program {

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    lazy val log = Logger.getLogger("infoLogger")

    val session = SparkSession.builder.appName("Program").master("local").getOrCreate()
    val dfReader = session.read

    val inputData = dfReader.option("header", "true")
      .option("inferSchema", value = true)
      //.csv("s3://edu.utdallas/house.csv")
      .csv("src/main/resources/house.csv")
      .cache

    val test = dfReader.option("header", "true")
      .option("inferSchema", value = true)
      //.csv("s3://edu.utdallas/house.csv")
      .csv("src/main/resources/test.csv")

    // *************************** START PREPROCESSING *************************** \\

    val dfOfInputCols = inputData.select("SalePrice", "GrLivArea", "OverallQual", "GarageFinish",
      "Fireplaces", "MasVnrArea", "PoolQC")
    inputData.unpersist()

    val stringTypeCols = dfOfInputCols.schema.fields.filter(x => x.dataType == StringType)
    val nonStringTypeCols = dfOfInputCols.schema.fields.filter(x => x.dataType != StringType)

    val indexers = stringTypeCols.map { column =>
      new StringIndexer()
        .setInputCol(column.name)
        .setOutputCol(column.name + "_indexed")
        .setHandleInvalid("keep")
    }

    val nonStringTypeColNames = nonStringTypeCols.map(column => column.name)

    val inputCols = new mutable.ArrayBuffer[String]()
    val stages = new mutable.ArrayBuffer[PipelineStage]()
    val lrStages = new mutable.ArrayBuffer[PipelineStage]()

    for (indexer <- indexers) {
      stages += indexer
      lrStages += indexer
    }

    val encoders = stringTypeCols.map { column =>
      new OneHotEncoderEstimator()
        .setInputCols(Array(column.name + "_indexed"))
        .setOutputCols(Array(column.name + "_ClassVec"))
    }

    for (encoder <- encoders) {
      inputCols += encoder.getOutputCols(0)
      stages += encoder
      lrStages += encoder
    }

    for (name <- nonStringTypeColNames) {
      inputCols += name
    }

    /**
     * Define a vector assembler:
     * Assemble the feature columns into a feature vector.
     */
    val assembler = new VectorAssembler()
      .setInputCols(inputCols.toArray)
      .setOutputCol("features")

    // *************************** END PREPROCESSING *************************** \\

    val targetColumn = "SalePrice"
    val learningRate = 0.001
    val minInfoGain = .6
    val seed = 4752L
    /**
     * Train a GBT model
     */
    val gbt = new GBTRegressor()
      .setLabelCol(targetColumn)
      .setFeaturesCol("features")
      .setPredictionCol("Predicted " + targetColumn)
      .setLossType("squared")
      .setStepSize(learningRate)
      .setMinInfoGain(minInfoGain)
      .setCheckpointInterval(2)
      .setSeed(seed)

    stages += assembler
    stages += gbt

    val pipeline = new Pipeline().setStages(stages.toArray)

    /**
     * Define evaluation metric:
     * Tells CrossValidator how well we are doing comparing true values to predictions,
     */
    val evaluator = new RegressionEvaluator()
      .setLabelCol(targetColumn)
      .setPredictionCol("Predicted " + targetColumn)
      .setMetricName("rmse")

    /**
     * Define a grid of hyperparameters to test:
     * maxDepth: max depth of each decision tree in the GBT ensemble
     * maxIter: number of trees in each GBT ensemble
     */
    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxDepth, Array(10))
      .addGrid(gbt.maxIter, Array(10))
      .build()

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("SalePrice")
      .setMaxIter(1)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    lrStages += assembler
    lrStages += lr

    val lrPipeline = new Pipeline().setStages(lrStages.toArray)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)
      .setParallelism(10)

    val Array(train, validation) = inputData.randomSplit(Array[Double](0.7, 0.3), seed)

    println("Training started")
    train.cache()
    val cvModel = cv.fit(train)
    train.unpersist()
    println("Training ended")

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    println("Validation started")
    validation.cache()
    val predictions = bestModel.transform(validation)
    validation.unpersist()
    println("Validation ended")

//    println("Testing started")
//    test.cache()
//    val predictions = bestModel.transform(test)
//    test.unpersist()
//    println("Testing ended")

    println("Final results")
    cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics).foreach(println)
    val results = predictions.select("Id", targetColumn, "Predicted " + targetColumn).toDF()
    results.createOrReplaceTempView("resultsTable")
    session.sql("SELECT Id, SalePrice, `Predicted SalePrice` FROM resultsTable LIMIT 20").show()
    val submission = session.sql("SELECT Id, SalePrice, `Predicted SalePrice` FROM resultsTable").toDF
    submission.write.format("csv").option("header", "true").mode("Overwrite").save("./Submission")
    //submission.write.format("csv").mode("Overwrite").save("s3n://edu.utdallas.house/Submission")

    //val lrModel = lrPipeline.fit(train)
    //val linearModel = lrModel.stages(0).asInstanceOf[PipelineModel]
    //println(lrModel.stages(0).asInstanceOf[LinearRegressionModel])
    session.stop()
  }
}