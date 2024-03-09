package com.test.job;

import com.alibaba.fastjson.JSONObject;
import com.test.utils.CommonFlinkJobUtils;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@Data
@NoArgsConstructor
@AllArgsConstructor
public class CommonFlinkXGBProcess extends ProcessFunction<JSONObject, String> {
    private String[] features;
    private String label;
    private Booster booster;

    private String outputValue;
    private String outputFeatureName;
    @Override
    public void processElement(JSONObject jsonObject, ProcessFunction<JSONObject, String>.Context context, Collector<String> collector) {
        List<Float> data = new ArrayList<>();
        for (int i = 0; i < features.length; i++) {
            data.add(jsonObject.getFloat(features[i]));
        }
        DMatrix matrix = CommonFlinkJobUtils.getDMatrix(data);
        float predictResult = 0;
        try {
            predictResult = booster.predict(matrix)[0][0];
            if(predictResult > 0.4f){
                jsonObject.put(label, outputValue);
                jsonObject.put("featureName", outputFeatureName);
                collector.collect(jsonObject.toJSONString());
                log.info("XGB Model predicts fraud: {}", jsonObject.toJSONString());
            }
        } catch (XGBoostError e) {
            log.error("XGB model predict error", e);
        }

    }
}
