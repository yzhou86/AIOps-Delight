import os
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

api_version = "2023-07-01-preview"
endpoint = "https://llm-proxy.intelligence.test.com/azure/v1"

#token = token_generator.get_token()
token = 'XXXX'


os.environ["AZURE_OPENAI_API_KEY"] = token
os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint

llm = AzureChatOpenAI(
    azure_deployment="gpt-4",
    openai_api_version=api_version,
    temperature=0,
    model_name="gpt-4")

prompt = PromptTemplate(
    input_variables=["log_text", "job_type"],
    template="You are expert of {job_type} jobs, you can help me analyze job logs text and give me the root "
             "cause analyze and suggest some actions. Please response with only key root cause and action items. Now, "
             "my job error logs are: {log_text}"
)

chain = LLMChain(llm=llm, prompt=prompt)

def analyze_log(log_text, job_type='spark/flink/etl'):
    try:
        return chain.run({'log_text': log_text, 'job_type': job_type})
    except Exception as e:
        print('analyze logs using LLM error:', e)
    return ""


if __name__ == '__main__':
    # log_text = "2023-10-17 01:41:42.579 [main] WARN org.apache.iceberg.util.Tasks - Retrying task after failure: Waiting for lock on table pda.wmequality_report_meeting_summary_ice org.apache.iceberg.hive.HiveTableOperations$WaitingForLockException: Waiting for lock on table pda.wmequality_report_meeting_summary_ice at org.apache.iceberg.hive.HiveTableOperations.lambda$acquireLock$9(HiveTableOperations.java:544) at org.apache.iceberg.util.Tasks$Builder.runTaskWithRetry(Tasks.java:404) at org.apache.iceberg.util.Tasks$Builder.runSingleThreaded(Tasks.java:214) at org.apache.iceberg.util.Tasks$Builder.run(Tasks.java:198)	"
    log_text = """
    2024-01-25 07:20:31,470 tn="prometheus-http-1-30462" ERROR org.apache.flink.runtime.util.ClusterUncaughtExceptionHandler [] - WARNING: Thread 'prometheus-http-1-30462' produced an uncaught exception. If you want to fail on uncaught exceptions, then configure cluster.uncaught-exception-handling accordingly
java.lang.OutOfMemoryError: Java heap space
    """
    print(analyze_log(log_text, 'flink'))


