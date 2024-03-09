package com.test.job;

import com.alibaba.fastjson.JSONObject;
import com.test.common.config.DataScienceRunConfig;
import com.test.utils.CommonFlinkJobUtils;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import ml.dmlc.xgboost4j.java.Booster;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import static com.test.utils.CommonFlinkJobUtils.MODEL_CACHE;

@Builder
@Slf4j
public class CommonFlinkRunner {
    private String jobName;
    private DataScienceRunConfig config;

    public void run() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        String filter = config.getQuery().getFilter();
        String label = config.getAnalytic().getLabel();
        String[] features = config.getAnalytic().getFeatures().split(",");
        Booster booster = CommonFlinkJobUtils.getXGBModel(MODEL_CACHE.get(config.getAnalytic().getModelRedisKey()));
        String outputValue = config.getAnalytic().getOutput().getOutputValue();
        String outputFeatureName = config.getAnalytic().getOutput().getOutputFeatureName();
        CommonFlinkXGBProcess process = new CommonFlinkXGBProcess(features, label, booster, outputValue, outputFeatureName);
        //source
        KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
                .setBootstrapServers(config.getQuery().getMetadata().get("bootstrapServer"))
                .setTopics(config.getQuery().getMetadata().get("topic"))
                .setStartingOffsets(OffsetsInitializer.latest())
                .setValueOnlyDeserializer(new SimpleStringSchema())
                .build();

        SingleOutputStreamOperator<JSONObject> dataStreamSource = env.fromSource(kafkaSource,
                WatermarkStrategy.noWatermarks(), config.getQuery().getDataSource())
               .map(JSONObject::parseObject)
               .filter(jsonObject -> CommonFlinkJobUtils.filterByFeature(jsonObject, filter));


        // transform
        SingleOutputStreamOperator<String> singleOutputStreamOperator = dataStreamSource
                .process(process);

        //sink
        KafkaSink<String> sink = KafkaSink.<String>builder()
                .setBootstrapServers(config.getAnalytic().getOutput().getOutputMetadata().get("bootstrapServer"))
                .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                        .setTopic(config.getAnalytic().getOutput().getOutputMetadata().get("topic"))
                        .setValueSerializationSchema(new SimpleStringSchema())
                        .build()
                )
                .build();
        singleOutputStreamOperator.sinkTo(sink);
        env.execute();

    }

}
