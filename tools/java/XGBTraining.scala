package com.test.train

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType

object XGBTraining {

  val assembler = new VectorAssembler()
    .setInputCols(Array("call_counts","short_call_count","risk_avg","risk_sum","risk_max","risk_min","cost_avg",
      "cost_sum","cost_max","cost_min","suspicion_avg","suspicion_sum","suspicion_max","suspicion_min","no_answered"))
    .setOutputCol("features")


  val cols = Array("call_counts", "short_call_count", "risk_avg", "risk_sum", "risk_max", "risk_min", "cost_avg",
    "cost_sum", "cost_max", "cost_min", "suspicion_avg", "suspicion_sum", "suspicion_max", "suspicion_min", "no_answered","fraud")

  val booster = new XGBoostClassifier(
    Map(
      "n_estimators" -> 450,
      "eta" -> 0.3f,
      "max_depth" -> 6,
      "gamma" -> 0,
      "min_child_weight" -> 2,
      "colsample_bytree" -> 0.2,
      "colsample_bylevel" -> 0.1,
      "subsample" -> 0.9,
      "reg_lambda" -> 0,
      "reg_alpha" -> 0,
      "seed" -> 33,
      "objective" -> "binary:logistic",
      "learning_rate" -> 0.0001,
      "random_state" -> 42,
      "eval_metric" -> "auc",
      "scale_pos_weight" -> 300,
      "allow_non_zero_for_missing"-> true
    )
  )
  booster.setFeaturesCol("features")
  booster.setLabelCol("fraud")


  def training(ds:DataFrame): Unit = {
    var new_batch = ds.drop("batch")
    for (colName <- cols) {
      new_batch = new_batch.withColumn(colName, col(colName).cast(DoubleType))
    }

    val Array(training, test) = new_batch.randomSplit(Array(0.8, 0.2), 123)


    val pipeline = new Pipeline()
      .setStages(Array(assembler, booster))
    val model = pipeline.fit(training)

    val prediction = model.transform(test)
    prediction.show(false)

    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setLabelCol("fraud")
    evaluator.setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(prediction)
    println("The model accuracy is : " + accuracy)
  }

}
