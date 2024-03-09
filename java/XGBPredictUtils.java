package com.test.utils;

import java.util.*;
import java.util.stream.Collectors;

public class XGBPredictUtils {
    public static final String FEATURES_COLUMNS_STRING= "join_country_max,join_country_sum,join_country_mean,user_name_max," +
            "user_name_sum,sim_call_max_count_max,sim_call_max_count_sum,avg_interval_time_mean,avg_interval_time_max,avg_interval_time_min," +
            "interval_time_sigma_mean,interval_time_sigma_max,interval_time_sigma_min,total_call_count_mean,total_call_count_max,total_call_count_sum," +
            "average_call_minutes_mean,average_call_minutes_max,average_call_minutes_min,call_duration_sigma_mean,call_duration_sigma_max,call_duration_sigma_min," +
            "call_to_diff_bl_count_max,call_to_diff_bl_count_sum,phone_number_avg_similarity_max,phone_number_avg_similarity_sum,call_country_max,call_country_sum," +
            "call_country_mean,site_id_max,site_id_sum,site_id_mean,is_same_phone_number_sum,platform_nunique,is_guest_sum,domain_max,domain_sum,domain_mean," +
            "unique_id_feature_max,is_host_domain_sum,is_gdm_sum,username_container_special_string_sum,username_container_special_char_sum,user_name_count_max," +
            "user_name_count_mean,name_used_by_guest_sum,is_thin_client_sum,join_call_distance_max,join_call_idf_max,is_thin_client_percent";

    public static boolean getPredictResult(float[] featuresContrib) {
        float sum = 0;
        for (float value : featuresContrib) {
            sum += value;
        }
        double predictResult = 1 / (1 + Math.exp(-sum));
        return predictResult >= 0.4;
    }

    public static List<Map.Entry<String, Float>> getFeaturesContribList(float[] featuresContribValue){
        Map<String, Float> featuresContribMap = new HashMap<>();
        List<String> featuresColumnsList = Arrays.stream(FEATURES_COLUMNS_STRING.split(",")).collect(Collectors.toList());
        for (String key : featuresColumnsList) {
            int index = featuresColumnsList.indexOf(key);
            float contribValue = featuresContribValue[index];
            featuresContribMap.putIfAbsent(key, contribValue);
        }

        List<Map.Entry<String, Float>> list = new ArrayList<>(featuresContribMap.entrySet());
        list.sort((o1, o2) -> {
            if (o2.getValue() - o1.getValue() > 0) {
                return 1;
            }
            if (o2.getValue() - o1.getValue() < 0) {
                return -1;
            }
            return 0;
        });
        return list;
    }

}
