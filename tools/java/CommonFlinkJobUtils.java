package com.test.utils;


import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.TypeReference;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.io.ByteStreams;
import com.google.common.primitives.Floats;
import com.test.common.dataset.FraudFeatureContext;
import com.test.common.util.LoadFraudFeatureContextResource;
import com.test.common.util.OJDBCUtils;
import com.test.common.util.RedisClusterUtil;
import com.test.common.util.UnirestUtil;
import lombok.extern.slf4j.Slf4j;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.*;
import java.util.concurrent.TimeUnit;

@Slf4j
public class CommonFlinkJobUtils {


    private static final String blockUniqueIdsUrl = "https://antifraud.test.com/api/v1/post";
    private static final String fraToken = "Basic XXXX";



    public static final LoadingCache<String, byte[]> MODEL_CACHE = CacheBuilder.newBuilder()
            .concurrencyLevel(8)
            .initialCapacity(10)
            .maximumSize(10)
            .expireAfterWrite(24, TimeUnit.HOURS)
            .build(new CacheLoader<String, byte[]>() {
                @Override
                public byte[] load(String redisKey) {
                    try {
                        return RedisClusterUtil.getJedis().get(redisKey.getBytes(StandardCharsets.UTF_8));
                    } catch (Exception e) {
                        log.error("get binary model from redis error:{}", e.getMessage());
                    }
                    return loadLocalXGBModel();
                }
            });

    public static final LoadingCache<String, FraudFeatureContext> FEATURE_CONTEXT_CACHE = CacheBuilder.newBuilder()
            .concurrencyLevel(8)
            .initialCapacity(10)
            .maximumSize(10)
            .expireAfterWrite(2, TimeUnit.MINUTES)
            .build(new CacheLoader<String, FraudFeatureContext>() {
                @Override
                public FraudFeatureContext load(String key) {
                    try {
                        String countryIdf = RedisUtils.getIdfFromRedis("fraud_country_idf_v3_prod");
                        String domainIdf = RedisUtils.getIdfFromRedis("fraud_domain_idf_v3_prod");
                        String siteIdf = RedisUtils.getIdfFromRedis("fraud_site_idf_v3_prod");
                        String joinCallIdf = RedisUtils.getIdfFromRedis("fraud_join_call_idf_v3_prod");
                        JSONObject countryIdfJsonObject = JSONObject.parseObject(countryIdf);
                        JSONObject domainIdfJsonObject = JSONObject.parseObject(domainIdf);
                        JSONObject siteIdfJsonObject = JSONObject.parseObject(siteIdf);
                        JSONObject joinCallIdfJsonObject = JSONObject.parseObject(joinCallIdf);
                        Map<String, Map<String, Double>> countryIdfMap = JSONObject.parseObject(countryIdfJsonObject.toJSONString(), new TypeReference<Map<String, Map<String, Double>>>() {
                        });
                        Map<String, Double> siteIdfMap = JSONObject.parseObject(siteIdfJsonObject.getObject("site_idf", JSONObject.class).toJSONString(), new TypeReference<Map<String, Double>>() {
                        });
                        Map<String, Double> domainIdfMap = JSONObject.parseObject(domainIdfJsonObject.getObject("domain_idf", JSONObject.class).toJSONString(), new TypeReference<Map<String, Double>>() {
                        });
                        Map<String, Double> joinCallIdfMap = JSONObject.parseObject(joinCallIdfJsonObject.getObject("calc_call_join_idf", JSONObject.class).toJSONString(), new TypeReference<Map<String, Double>>() {
                        });

                        Set<String> clientUniqueIdBL = UnirestUtil.getBlockUniqueId(blockUniqueIdsUrl, fraToken);
//                        log.error("clientUniqueIdBL:{}", clientUniqueIdBL.size());
                        return new FraudFeatureContext(countryIdfMap, siteIdfMap, domainIdfMap, joinCallIdfMap, clientUniqueIdBL,
                                LoadFraudFeatureContextResource.countryCodeMap,
                                LoadFraudFeatureContextResource.countryDistanceMap,
                                LoadFraudFeatureContextResource.specialCharacterRegex,
                                LoadFraudFeatureContextResource.specialSubstringList,
                                LoadFraudFeatureContextResource.letterEntropy,
                                LoadFraudFeatureContextResource.failReason);

                    } catch (Exception e) {
                        log.error("get idf from redis error", e);
                    }
                    return null;
                }
            });

    public static final LoadingCache<String, String> IDF_CACHE = CacheBuilder.newBuilder()
            .concurrencyLevel(8)
            .initialCapacity(10)
            .maximumSize(10)
            .expireAfterWrite(1, TimeUnit.HOURS)
            .build(new CacheLoader<String, String>() {
                @Override
                public String load(String redisKey) {
                    try {
                        return RedisClusterUtil.getJedis().hget("idf", redisKey);
                    } catch (Exception e) {
                        log.error("get {} from redis error:{}", redisKey, e.getMessage());
                    }

                    return "";
                }
            });


    public static Boolean filterByFeature(JSONObject jsonObject, String filter) {
        String[] filters = filter.split(",");
        for (int i = 0; i < filters.length; i++) {
            String value = filters[i];
            String filterValue = jsonObject.getString("featureName");
            if (filterValue == null || !filterValue.equalsIgnoreCase(value)) {
                return false;
            }
        }
        return true;
    }


    public static Booster getXGBModel(byte[] modelBytes) {
        Booster booster = null;
        try {
            booster = XGBoost.loadModel(modelBytes);
        } catch (Exception e) {
            log.error("load XGB model error", e);
        }
        return booster;
    }


    public static DMatrix getDMatrix(List<Float> data) {

        float[] resultArray = Floats.toArray(data);

        int row = 1;
        int col = resultArray.length;
        float missing = 0.0f;
        try {
            return new DMatrix(resultArray, row, col, missing);
        } catch (XGBoostError e) {
            log.error("get data dMatrix error:{}", e.getMessage());
        }
        return null;
    }

    private static byte[] loadLocalXGBModel() {
        try {
            InputStream inputStream = Objects.requireNonNull(CommonFlinkJobUtils.class.getClassLoader().getResourceAsStream("fraud_xgb.model"));
            return ByteStreams.toByteArray(inputStream);
        } catch (IOException e) {
            log.error("get model from local error", e);
        }
        return new byte[0];
    }
}



