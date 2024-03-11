import re

top_category = ['oomkilled', 'flink', 'spark', 'hadoop', 'hdfs', 'etl', 'kafka', 'iceberg', 'hive', 'thrift', 's3', 'kyuubi', 'glue'
               'namenode', 'pinot', 'mysql', 'arctic', 'kubernetes', 'kube', 'jdbc', 'prometheus', 'http', 'session',
                'security', 'postgresql', 'springframework', 'json', 'outofmemoryerror', 'nullpointerexception',
                'waitingforlockexception',
                'sql', 'ciclient', 'lineage', 'configuration', 'checkpoint', 'ioexception', 'socket', 'shuffle',
                'consumer', 'producer', 'crypto', 'io', 'network', 'class', 'jar']

signature_white_list = ['org.apache.flink.metrics.MetricGroup',
                        'org.apache.hadoop.hdfs.DataStreamer',
                        'org.apache.kafka.clients.admin.AdminClientConfig',
                        'org.apache.kafka.clients.consumer.CommitFailedException',
                        'org.apache.flink.runtime.rest.handler.taskmanager.TaskManagersHandler']


def cluster_category(cluster_keywords):
    category_match = []
    cluster_kw = cluster_keywords.split(',')
    for category in top_category:
        if re.search(category, cluster_keywords, re.IGNORECASE):
            if category in cluster_kw:
                cluster_kw.remove(category)
            category_match.append(category.capitalize())
    cluster_kw_str = ','.join(cluster_kw)
    if len(category_match) > 0:
        if len(category_match) > 2:
            return ' '.join(category_match)
        else:
            return ' '.join(category_match) + ' (' + cluster_kw_str + ')'
    else:
        return cluster_kw_str


def have_white_sig(signature):
    for white in signature_white_list:
        if signature and re.search(white, signature, re.IGNORECASE):
            print('white list matched in signature:', white)
            return True
    return False
