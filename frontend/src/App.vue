<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'

function detectInitialLocale() {
  if (typeof window === 'undefined') return 'en'
  const stored = window.localStorage.getItem('aiops-delight-locale')
  if (stored === 'zh' || stored === 'en') return stored
  return window.navigator.language?.toLowerCase().startsWith('zh') ? 'zh' : 'en'
}

const COPY = {
  en: {
    welcome: 'Upload a CSV or Excel file, or choose a built-in example dataset, then ask the agent to analyze it. I will keep the full conversation history here.',
    defaultPrompt: 'Find the strongest patterns in this dataset, call out anomalies, and recommend what I should investigate next.',
    appKicker: 'DataPilot',
    headerTitle: 'Chat with your dataset',
    headerSubtitle: 'Choose tools, attach a file, and talk to the analysis agent in one running thread.',
    languageLabel: 'Language',
    english: 'English',
    chinese: '中文',
    tools: 'Tools',
    activeShort: 'active',
    dataset: 'Dataset',
    ready: 'Ready',
    waiting: 'Waiting',
    attachTitle: 'Attach CSV / XLS / XLSX',
    attachInspecting: 'Inspecting file...',
    attachCaption: 'Your file is profiled first, then the agent can analyze it.',
    examples: 'Built-in examples',
    chooseExample: 'Choose an example dataset',
    loadExample: 'Load example',
    loading: 'Loading...',
    selectedExample: 'Selected built-in example: {file}',
    attachedDataset: 'Attached dataset: {file}',
    promptIdeas: 'Prompt Ideas',
    conversation: 'Conversation',
    analysisThread: 'Analysis thread',
    noDatasetYet: 'No dataset yet',
    numeric: 'numeric',
    categorical: 'categorical',
    text: 'text',
    datetime: 'datetime',
    aiAnswer: 'AI Answer',
    analysisSummary: 'Analysis Summary',
    attachFile: 'Attach file',
    loadingExample: 'Loading example...',
    useSelectedExample: 'Use selected example',
    noDatasetAttached: 'No dataset attached',
    promptPlaceholder: 'Ask the agent to analyze the dataset, compare segments, forecast a metric, or explain anomalies.',
    analyzing: 'Analyzing...',
    send: 'Send to agent',
    analysisSettings: 'Analysis Settings',
    settingsHint: 'Upload a CSV or Excel file to unlock target, time, value, and text-column settings.',
    targetColumn: 'Target column',
    timeColumn: 'Time column',
    valueColumn: 'Value column',
    textColumns: 'Text columns',
    optional: 'Optional',
    noTextColumns: 'No text columns detected in this dataset yet.',
    textColumnsAfterUpload: 'Text-column options will appear here after upload.',
    usefulLinks: 'Useful Links',
    you: 'You',
    agent: 'Agent',
    system: 'System',
    file: 'File',
    rows: 'Rows',
    columns: 'Columns',
    textColumnsShort: 'Text columns',
    uploadFirstError: 'Upload or load a dataset before sending an analysis request.',
    inspectMessage: 'Inspecting {file} and inferring schema...',
    loadExampleMessage: 'Loading built-in example {file}...',
    inspectFailed: 'Dataset inspection failed.',
    exampleLoadFailed: 'Example dataset could not be loaded.',
    analysisFailed: 'Analysis failed.',
    runningTools: 'Running the selected tools and composing the analysis...',
    datasetHeadline: '{file} inspected with {rows} rows and {columns} columns.',
    statusOk: 'ok',
    statusSkipped: 'skipped',
    statusError: 'error',
    statusLoading: 'loading'
  },
  zh: {
    welcome: '上传 CSV 或 Excel 文件，或者直接选择内置示例数据集，然后用自然语言让智能体分析。我会在这里保留完整对话历史。',
    defaultPrompt: '请找出这个数据集里最强的模式、异常点，以及接下来最值得我调查的方向。',
    appKicker: 'DataPilot 数据智能助手',
    headerTitle: '和你的数据集对话',
    headerSubtitle: '选择工具，接入数据，然后在一个连续会话里和分析智能体交流。',
    languageLabel: '语言',
    english: 'English',
    chinese: '中文',
    tools: '工具',
    activeShort: '已启用',
    dataset: '数据集',
    ready: '就绪',
    waiting: '等待中',
    attachTitle: '上传 CSV / XLS / XLSX',
    attachInspecting: '正在解析文件...',
    attachCaption: '系统会先做数据探查，再进入智能分析。',
    examples: '内置示例',
    chooseExample: '选择一个示例数据集',
    loadExample: '载入示例',
    loading: '加载中...',
    selectedExample: '已选择内置示例：{file}',
    attachedDataset: '已附加数据集：{file}',
    promptIdeas: '提示词建议',
    conversation: '对话区',
    analysisThread: '分析会话',
    noDatasetYet: '尚未选择数据集',
    numeric: '数值列',
    categorical: '分类列',
    text: '文本列',
    datetime: '时间列',
    aiAnswer: 'AI 回答',
    analysisSummary: '分析摘要',
    attachFile: '上传文件',
    loadingExample: '正在加载示例...',
    useSelectedExample: '使用当前示例',
    noDatasetAttached: '尚未附加数据集',
    promptPlaceholder: '让智能体分析数据、比较分群、预测指标走势，或者解释异常原因。',
    analyzing: '分析中...',
    send: '发送给智能体',
    analysisSettings: '分析设置',
    settingsHint: '上传 CSV 或 Excel 文件后，就可以配置目标列、时间列、数值列和文本列。',
    targetColumn: '目标列',
    timeColumn: '时间列',
    valueColumn: '数值列',
    textColumns: '文本列',
    optional: '可选',
    noTextColumns: '当前数据集暂未识别出文本列。',
    textColumnsAfterUpload: '上传数据后，这里会显示可选文本列。',
    usefulLinks: '常用链接',
    you: '你',
    agent: '智能体',
    system: '系统',
    file: '文件',
    rows: '行数',
    columns: '列数',
    textColumnsShort: '文本列',
    uploadFirstError: '请先上传数据集或载入一个示例，再发送分析请求。',
    inspectMessage: '正在解析 {file} 并推断字段结构...',
    loadExampleMessage: '正在加载内置示例 {file}...',
    inspectFailed: '数据集解析失败。',
    exampleLoadFailed: '内置示例加载失败。',
    analysisFailed: '分析失败。',
    runningTools: '正在运行所选工具并整理分析结果...',
    datasetHeadline: '已完成对 {file} 的探查，共有 {rows} 行、{columns} 列。',
    statusOk: '成功',
    statusSkipped: '已跳过',
    statusError: '错误',
    statusLoading: '处理中'
  }
}

const TOOL_I18N = {
  data_profile: { en: { name: 'Dataset Profile', category: 'Foundation' }, zh: { name: '数据概览', category: '基础' } },
  correlation_explorer: { en: { name: 'Correlation Explorer', category: 'Signals' }, zh: { name: '相关性探索', category: '信号' } },
  anomaly_detector: { en: { name: 'Anomaly Detector', category: 'Operations' }, zh: { name: '异常检测', category: '运维' } },
  kmeans_segmentation: { en: { name: 'KMeans Segmentation', category: 'Grouping' }, zh: { name: 'KMeans 分群', category: '分群' } },
  text_clusterer: { en: { name: 'Text Clusterer', category: 'NLP' }, zh: { name: '文本聚类', category: '文本' } },
  forecast_baseline: { en: { name: 'Forecast Baseline', category: 'Time Series' }, zh: { name: '基线预测', category: '时序' } },
  classification_explorer: { en: { name: 'Classification Explorer', category: 'Supervised' }, zh: { name: '分类探索', category: '监督学习' } }
}

const EXAMPLE_I18N = {
  ops_capacity_forecast: {
    en: { label: 'Ops Capacity Forecast', description: 'Time-series operations metrics with load, latency, and queue depth for forecasting and anomaly checks.' },
    zh: { label: '运维容量预测', description: '包含负载、时延和队列深度的时序运维指标，适合做预测和异常检查。' }
  },
  incident_log_topics: {
    en: { label: 'Incident Log Topics', description: 'Free-text incident summaries and resolution hints for text clustering and segmentation.' },
    zh: { label: '事件日志主题', description: '带有事件摘要和处置建议的自由文本数据，适合文本聚类和主题分群。' }
  },
  fraud_risk_classification: {
    en: { label: 'Fraud Risk Classification', description: 'Tabular fraud-likelihood records for classification, segmentation, and signal discovery.' },
    zh: { label: '欺诈风险分类', description: '结构化欺诈风险样本，适合分类、分群和风险信号分析。' }
  },
  service_health_anomalies: {
    en: { label: 'Service Health Anomalies', description: 'Workbook with service-health metrics and ticket text, useful for anomalies and mixed-signal analysis.' },
    zh: { label: '服务健康异常', description: '包含服务健康指标和工单文本的工作簿，适合异常与混合信号分析。' }
  },
  customer_churn_signals: {
    en: { label: 'Customer Churn Signals', description: 'Subscription-health and support-behavior data for churn classification and feature ranking.' },
    zh: { label: '客户流失信号', description: '订阅健康度与支持行为数据，适合流失分类和特征重要性分析。' }
  },
  cloud_cost_guardrails: {
    en: { label: 'Cloud Cost Guardrails', description: 'Daily cloud spend, traffic, and efficiency metrics for anomaly detection and forecasting.' },
    zh: { label: '云成本护栏', description: '包含日成本、流量和资源效率指标，适合成本异常识别和趋势预测。' }
  }
}

const USEFUL_LINKS = {
  en: [
    { label: 'Tool Catalog API', href: '/api/tools', caption: 'Inspect the available tool catalog as JSON.' },
    { label: 'Health Check', href: '/api/health', caption: 'Confirm the unified Flask app is healthy.' },
    { label: 'Pandas Docs', href: 'https://pandas.pydata.org/docs/', caption: 'Useful for dataframe-oriented analysis.' },
    { label: 'Scikit-learn', href: 'https://scikit-learn.org/stable/', caption: 'Reference for clustering, anomaly detection, and classification.' },
    { label: 'Flask Docs', href: 'https://flask.palletsprojects.com/', caption: 'Backend reference for the unified app runtime.' }
  ],
  zh: [
    { label: '工具目录 API', href: '/api/tools', caption: '以 JSON 形式查看当前可用分析工具。' },
    { label: '健康检查', href: '/api/health', caption: '确认统一 Flask 应用运行正常。' },
    { label: 'Pandas 文档', href: 'https://pandas.pydata.org/docs/', caption: '适合数据表分析和清洗时参考。' },
    { label: 'Scikit-learn 文档', href: 'https://scikit-learn.org/stable/', caption: '聚类、异常检测和分类建模的参考资料。' },
    { label: 'Flask 文档', href: 'https://flask.palletsprojects.com/', caption: '统一后端运行方式的框架文档。' }
  ]
}

const SUGGESTED_PROMPTS = {
  en: [
    'Summarize the most important trends, outliers, and next actions.',
    'Focus on anomalies and tell me which rows or periods deserve attention first.',
    'Group this dataset into meaningful segments and explain each segment.',
    'Train a baseline classifier and tell me which features matter most.'
  ],
  zh: [
    '请总结最重要的趋势、异常点，以及下一步建议。',
    '请重点关注异常，并告诉我哪些行或哪些时间段最值得先排查。',
    '请把这个数据集划分成有业务意义的分群，并解释每个分群。',
    '请训练一个基线分类模型，并告诉我最重要的特征是什么。'
  ]
}

const locale = ref(detectInitialLocale())
const tools = ref([])
const examples = ref([])
const selectedTools = ref(['data_profile', 'correlation_explorer', 'anomaly_detector'])
const datasetFile = ref(null)
const datasetMeta = ref(null)
const selectedExampleId = ref('')
const prompt = ref(COPY[locale.value].defaultPrompt)
const targetColumn = ref('')
const timeColumn = ref('')
const valueColumn = ref('')
const textColumns = ref([])
const inspectLoading = ref(false)
const analyzeLoading = ref(false)
const chatViewport = ref(null)
const nextMessageId = ref(1)
const messages = ref([
  {
    id: 1,
    role: 'assistant',
    kind: 'welcome',
    text: COPY[locale.value].welcome
  }
])

const copy = computed(() => COPY[locale.value])
const usefulLinks = computed(() => USEFUL_LINKS[locale.value])
const suggestedPrompts = computed(() => SUGGESTED_PROMPTS[locale.value])
const selectedToolDetails = computed(() =>
  tools.value.filter((tool) => selectedTools.value.includes(tool.id))
)
const selectedToolNames = computed(() => selectedToolDetails.value.map((tool) => localizedToolName(tool)))
const datasetColumns = computed(() => datasetMeta.value?.columns || [])
const selectedExample = computed(() =>
  examples.value.find((example) => example.id === selectedExampleId.value) || null
)
const canAnalyze = computed(() =>
  Boolean(datasetMeta.value?.datasetId) &&
  selectedTools.value.length > 0 &&
  prompt.value.trim() &&
  !analyzeLoading.value
)

function t(key, values = {}) {
  let template = copy.value[key] ?? key
  Object.entries(values).forEach(([name, value]) => {
    template = template.replace(`{${name}}`, value)
  })
  return template
}

function localizedToolMeta(toolOrId) {
  const id = typeof toolOrId === 'string' ? toolOrId : toolOrId?.id
  return TOOL_I18N[id]?.[locale.value] || null
}

function localizedToolName(toolOrId) {
  return localizedToolMeta(toolOrId)?.name || (typeof toolOrId === 'string' ? toolOrId.replace(/_/g, ' ') : toolOrId?.name)
}

function localizedToolCategory(tool) {
  return localizedToolMeta(tool)?.category || tool.category
}

function localizedExampleMeta(exampleOrId) {
  const id = typeof exampleOrId === 'string' ? exampleOrId : exampleOrId?.id
  return EXAMPLE_I18N[id]?.[locale.value] || null
}

function localizedExampleLabel(example) {
  return localizedExampleMeta(example)?.label || example.label
}

function localizedExampleDescription(example) {
  return localizedExampleMeta(example)?.description || example.description
}

function createMessage(role, kind, payload = {}) {
  nextMessageId.value += 1
  return {
    id: nextMessageId.value,
    role,
    kind,
    ...payload
  }
}

async function scrollToBottom() {
  await nextTick()
  if (chatViewport.value) {
    chatViewport.value.scrollTop = chatViewport.value.scrollHeight
  }
}

function pushMessage(role, kind, payload = {}) {
  const message = createMessage(role, kind, payload)
  messages.value.push(message)
  scrollToBottom()
  return message.id
}

function replaceMessage(messageId, patch) {
  const index = messages.value.findIndex((message) => message.id === messageId)
  if (index >= 0) {
    messages.value[index] = {
      ...messages.value[index],
      ...patch
    }
  }
  scrollToBottom()
}

function changeLanguage(nextLocale) {
  if (nextLocale !== 'zh' && nextLocale !== 'en') return
  const previousLocale = locale.value
  locale.value = nextLocale
  if (typeof window !== 'undefined') {
    window.localStorage.setItem('aiops-delight-locale', nextLocale)
  }

  if (!prompt.value || prompt.value === COPY[previousLocale].defaultPrompt) {
    prompt.value = COPY[nextLocale].defaultPrompt
  }

  const welcomeMessage = messages.value.find((message) => message.kind === 'welcome')
  if (welcomeMessage) {
    welcomeMessage.text = COPY[nextLocale].welcome
  }
}

function formatLabel(value) {
  const key = String(value || '').toLowerCase()
  if (key === 'numeric') return t('numeric')
  if (key === 'categorical') return t('categorical')
  if (key === 'text') return t('text')
  if (key === 'datetime') return t('datetime')
  return String(value || '').replace(/_/g, ' ')
}

function formatStatusLabel(status) {
  if (status === 'ok') return t('statusOk')
  if (status === 'skipped') return t('statusSkipped')
  if (status === 'error') return t('statusError')
  if (status === 'loading') return t('statusLoading')
  return status
}

function roleLabel(role) {
  if (role === 'user') return t('you')
  if (role === 'assistant') return t('agent')
  return t('system')
}

function datasetHeadline(dataset) {
  return t('datasetHeadline', {
    file: dataset.fileName,
    rows: String(dataset.rowCount),
    columns: String(dataset.columnCount)
  })
}

function choosePrompt(text) {
  prompt.value = text
}

async function fetchTools() {
  const response = await fetch('/api/tools')
  const data = await response.json()
  tools.value = data.tools || []
}

async function fetchExamples() {
  const response = await fetch('/api/examples')
  const data = await response.json()
  examples.value = data.examples || []
}

function toggleTool(toolId) {
  if (selectedTools.value.includes(toolId)) {
    selectedTools.value = selectedTools.value.filter((id) => id !== toolId)
    return
  }
  selectedTools.value = [...selectedTools.value, toolId]
}

async function handleFileChange(event) {
  const file = event.target.files?.[0]
  datasetFile.value = file || null
  selectedExampleId.value = ''
  if (file) {
    pushMessage('user', 'upload', {
      text: t('attachedDataset', { file: file.name })
    })
    await inspectDataset()
  }
}

function applyDatasetMeta(data) {
  datasetMeta.value = data
  targetColumn.value = data.categoricalColumns?.[0] || ''
  timeColumn.value = data.datetimeColumns?.[0] || ''
  valueColumn.value = data.numericColumns?.[0] || ''
  textColumns.value = data.textColumns ? data.textColumns.slice(0, 1) : []
}

async function inspectDataset() {
  if (!datasetFile.value) return

  inspectLoading.value = true
  const loadingId = pushMessage('assistant', 'loading', {
    text: t('inspectMessage', { file: datasetFile.value.name })
  })

  try {
    const formData = new FormData()
    formData.append('file', datasetFile.value)

    const response = await fetch('/api/datasets/inspect', {
      method: 'POST',
      body: formData
    })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.error || t('inspectFailed'))
    }

    applyDatasetMeta(data)

    replaceMessage(loadingId, {
      role: 'assistant',
      kind: 'dataset',
      text: datasetHeadline(data),
      dataset: data
    })
  } catch (error) {
    replaceMessage(loadingId, {
      role: 'assistant',
      kind: 'error',
      text: error.message
    })
  } finally {
    inspectLoading.value = false
  }
}

async function loadExampleDataset() {
  if (!selectedExampleId.value) return

  datasetFile.value = null
  inspectLoading.value = true
  const example = selectedExample.value
  const loadingId = pushMessage('assistant', 'loading', {
    text: t('loadExampleMessage', { file: example?.fileName || selectedExampleId.value })
  })

  try {
    const response = await fetch('/api/examples/load', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        exampleId: selectedExampleId.value
      })
    })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.error || t('exampleLoadFailed'))
    }

    pushMessage('user', 'upload', {
      text: t('selectedExample', { file: data.fileName })
    })

    applyDatasetMeta(data)
    replaceMessage(loadingId, {
      role: 'assistant',
      kind: 'dataset',
      text: datasetHeadline(data),
      dataset: data
    })
  } catch (error) {
    replaceMessage(loadingId, {
      role: 'assistant',
      kind: 'error',
      text: error.message
    })
  } finally {
    inspectLoading.value = false
  }
}

async function analyzeDataset() {
  if (!datasetMeta.value?.datasetId) {
    pushMessage('assistant', 'error', {
      text: t('uploadFirstError')
    })
    return
  }

  const userPrompt = prompt.value.trim()
  if (!userPrompt) return

  analyzeLoading.value = true
  pushMessage('user', 'prompt', {
    text: userPrompt,
    toolNames: selectedToolNames.value,
    fileName: datasetMeta.value.fileName
  })
  const loadingId = pushMessage('assistant', 'loading', {
    text: t('runningTools')
  })

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        datasetId: datasetMeta.value.datasetId,
        selectedTools: selectedTools.value,
        prompt: userPrompt,
        targetColumn: targetColumn.value || null,
        timeColumn: timeColumn.value || null,
        valueColumn: valueColumn.value || null,
        textColumns: textColumns.value,
        language: locale.value
      })
    })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.error || t('analysisFailed'))
    }

    replaceMessage(loadingId, {
      role: 'assistant',
      kind: 'analysis',
      text: data.answer || data.summary,
      answer: data.answer || data.summary,
      summary: data.summary,
      analysis: data
    })
    prompt.value = ''
  } catch (error) {
    replaceMessage(loadingId, {
      role: 'assistant',
      kind: 'error',
      text: error.message
    })
  } finally {
    analyzeLoading.value = false
  }
}

onMounted(() => {
  fetchTools()
  fetchExamples()
  scrollToBottom()
})
</script>

<template>
  <div class="copilot-shell">
    <header class="tool-banner">
      <div class="banner-copy">
        <p class="banner-kicker">{{ t('appKicker') }}</p>
        <h1>{{ t('headerTitle') }}</h1>
        <p>{{ t('headerSubtitle') }}</p>
      </div>
      <div class="language-toggle" role="group" :aria-label="t('languageLabel')">
        <span class="language-label">{{ t('languageLabel') }}</span>
        <button type="button" class="language-chip" :class="{ active: locale === 'en' }" @click="changeLanguage('en')">
          {{ t('english') }}
        </button>
        <button type="button" class="language-chip" :class="{ active: locale === 'zh' }" @click="changeLanguage('zh')">
          {{ t('chinese') }}
        </button>
      </div>
    </header>

    <main class="workspace">
      <aside class="sidebar sidebar-left">
        <section class="sidebar-card">
          <div class="card-heading">
            <h2>{{ t('tools') }}</h2>
            <span class="card-pill">{{ selectedTools.length }} {{ t('activeShort') }}</span>
          </div>
          <div class="sidebar-tool-list">
            <button
              v-for="tool in tools"
              :key="tool.id"
              type="button"
              class="sidebar-tool"
              :class="{ active: selectedTools.includes(tool.id) }"
              @click="toggleTool(tool.id)"
            >
              <span>{{ localizedToolName(tool) }}</span>
              <small>{{ localizedToolCategory(tool) }}</small>
            </button>
          </div>
        </section>

        <section class="sidebar-card dataset-card">
          <div class="card-heading">
            <h2>{{ t('dataset') }}</h2>
            <span class="card-pill" :class="{ muted: !datasetMeta }">
              {{ datasetMeta ? t('ready') : t('waiting') }}
            </span>
          </div>

          <label class="attach-drop">
            <input type="file" accept=".csv,.xls,.xlsx" @change="handleFileChange" />
            <span class="attach-title">
              {{ inspectLoading ? t('attachInspecting') : t('attachTitle') }}
            </span>
            <span class="attach-caption">{{ t('attachCaption') }}</span>
          </label>

          <div class="example-picker">
            <label>
              <span class="fact-label">{{ t('examples') }}</span>
              <select v-model="selectedExampleId" :disabled="inspectLoading || !examples.length">
                <option value="">{{ t('chooseExample') }}</option>
                <option v-for="example in examples" :key="example.id" :value="example.id">
                  {{ localizedExampleLabel(example) }}
                </option>
              </select>
            </label>
            <button
              type="button"
              class="secondary-button"
              :disabled="inspectLoading || !selectedExampleId"
              @click="loadExampleDataset"
            >
              {{ inspectLoading ? t('loading') : t('loadExample') }}
            </button>
          </div>

          <p v-if="selectedExample" class="example-caption">
            {{ localizedExampleDescription(selectedExample) }}
          </p>

          <div v-if="datasetMeta" class="dataset-facts">
            <div class="dataset-fact dataset-fact-file">
              <span class="fact-label">{{ t('file') }}</span>
              <strong>{{ datasetMeta.fileName }}</strong>
            </div>
            <div class="dataset-fact">
              <span class="fact-label">{{ t('rows') }}</span>
              <strong>{{ datasetMeta.rowCount }}</strong>
            </div>
            <div class="dataset-fact">
              <span class="fact-label">{{ t('columns') }}</span>
              <strong>{{ datasetMeta.columnCount }}</strong>
            </div>
            <div class="dataset-fact">
              <span class="fact-label">{{ t('textColumnsShort') }}</span>
              <strong>{{ datasetMeta.textColumns.length }}</strong>
            </div>
          </div>
        </section>

        <section class="sidebar-card">
          <div class="card-heading">
            <h2>{{ t('promptIdeas') }}</h2>
          </div>
          <div class="link-stack">
            <button
              v-for="item in suggestedPrompts"
              :key="item"
              type="button"
              class="prompt-link"
              @click="choosePrompt(item)"
            >
              {{ item }}
            </button>
          </div>
        </section>

      </aside>

      <section class="chat-panel">
        <div class="chat-frame">
          <div class="chat-header">
            <div>
              <p class="chat-kicker">{{ t('conversation') }}</p>
              <h2>{{ t('analysisThread') }}</h2>
            </div>
            <div class="chat-status">
              <span class="card-pill">{{ selectedTools.length }} {{ t('activeShort') }}</span>
              <span class="card-pill muted">{{ datasetMeta ? datasetMeta.fileName : t('noDatasetYet') }}</span>
            </div>
          </div>

          <div ref="chatViewport" class="chat-history">
            <article
              v-for="message in messages"
              :key="message.id"
              class="message-row"
              :class="message.role"
            >
              <div class="message-avatar">
                {{ message.role === 'user' ? 'Y' : message.role === 'assistant' ? 'AI' : 'SYS' }}
              </div>

              <div class="message-card" :class="[message.role, message.kind]">
                <div class="message-meta">
                  <span>{{ roleLabel(message.role) }}</span>
                  <span v-if="message.toolNames?.length" class="message-tools">
                    {{ message.toolNames.join(' · ') }}
                  </span>
                </div>

                <p v-if="message.kind !== 'analysis' && message.kind !== 'dataset'" class="message-text">
                  {{ message.text }}
                </p>

                <div v-if="message.kind === 'dataset'" class="dataset-message">
                  <p class="message-text">{{ message.text }}</p>
                  <div class="message-stats">
                    <span>{{ message.dataset.numericColumns.length }} {{ t('numeric') }}</span>
                    <span>{{ message.dataset.categoricalColumns.length }} {{ t('categorical') }}</span>
                    <span>{{ message.dataset.textColumns.length }} {{ t('text') }}</span>
                    <span>{{ message.dataset.datetimeColumns.length }} {{ t('datetime') }}</span>
                  </div>
                  <div class="table-shell">
                    <table>
                      <thead>
                        <tr>
                          <th v-for="column in message.dataset.preview.columns" :key="`preview-${column}`">{{ column }}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr
                          v-for="(row, rowIndex) in message.dataset.preview.rows"
                          :key="`preview-row-${rowIndex}`"
                        >
                          <td
                            v-for="column in message.dataset.preview.columns"
                            :key="`preview-${rowIndex}-${column}`"
                          >
                            {{ row[column] }}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                <div v-if="message.kind === 'analysis'" class="analysis-message">
                  <section class="answer-panel">
                    <div class="analysis-label">{{ t('aiAnswer') }}</div>
                    <pre class="summary-block">{{ message.answer || message.text }}</pre>
                  </section>

                  <section
                    v-if="message.summary && message.summary !== (message.answer || message.text)"
                    class="summary-panel"
                  >
                    <div class="analysis-label">{{ t('analysisSummary') }}</div>
                    <pre class="summary-block secondary">{{ message.summary }}</pre>
                  </section>

                  <div class="analysis-tools">
                    <span
                      v-for="tool in message.analysis.selectedTools"
                      :key="`analysis-tool-${tool}`"
                      class="mini-chip active"
                    >
                      {{ localizedToolName(tool) }}
                    </span>
                  </div>

                  <div class="analysis-result-stack">
                    <section
                      v-for="result in message.analysis.results"
                      :key="result.toolId"
                      class="analysis-result"
                    >
                      <div class="analysis-result-head">
                        <h3>{{ localizedToolName(result.toolId) }}</h3>
                        <span class="card-pill" :class="result.status">{{ formatStatusLabel(result.status) }}</span>
                      </div>
                      <p class="result-headline">{{ result.headline }}</p>

                      <ul v-if="result.insights.length" class="insight-list">
                        <li v-for="insight in result.insights" :key="insight">{{ insight }}</li>
                      </ul>

                      <div v-if="result.warnings.length" class="result-notes">
                        <p v-for="warning in result.warnings" :key="warning">{{ warning }}</p>
                      </div>

                      <div
                        v-for="table in result.tables"
                        :key="`${result.toolId}-${table.title}`"
                        class="result-table"
                      >
                        <h4>{{ table.title }}</h4>
                        <div class="table-shell" v-if="table.columns.length">
                          <table>
                            <thead>
                              <tr>
                                <th v-for="column in table.columns" :key="`${table.title}-${column}`">{{ column }}</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr
                                v-for="(row, rowIndex) in table.rows"
                                :key="`${table.title}-${rowIndex}`"
                              >
                                <td
                                  v-for="column in table.columns"
                                  :key="`${table.title}-${rowIndex}-${column}`"
                                >
                                  {{ row[column] }}
                                </td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </section>
                  </div>
                </div>
              </div>
            </article>
          </div>

          <form class="composer" @submit.prevent="analyzeDataset">
            <div class="composer-topline">
              <label class="attach-inline">
                <input type="file" accept=".csv,.xls,.xlsx" @change="handleFileChange" />
                <span>{{ inspectLoading ? t('attachInspecting') : t('attachFile') }}</span>
              </label>
              <button
                type="button"
                class="attach-inline attach-inline-secondary"
                :disabled="inspectLoading || !selectedExampleId"
                @click="loadExampleDataset"
              >
                <span>{{ inspectLoading ? t('loadingExample') : t('useSelectedExample') }}</span>
              </button>
              <span class="composer-file">{{ datasetMeta ? datasetMeta.fileName : t('noDatasetAttached') }}</span>
            </div>

            <textarea
              v-model="prompt"
              class="composer-input"
              rows="3"
              :placeholder="t('promptPlaceholder')"
            />

            <div class="composer-footer">
              <div class="selected-tool-strip">
                <span
                  v-for="tool in selectedToolDetails"
                  :key="`selected-${tool.id}`"
                  class="mini-chip active"
                >
                  {{ localizedToolName(tool) }}
                </span>
              </div>

              <button class="send-button" :disabled="!canAnalyze">
                {{ analyzeLoading ? t('analyzing') : t('send') }}
              </button>
            </div>
          </form>
        </div>
      </section>

      <aside class="sidebar sidebar-right">
        <section class="sidebar-card">
          <div class="card-heading">
            <h2>{{ t('analysisSettings') }}</h2>
          </div>

          <p v-if="!datasetMeta" class="settings-empty">
            {{ t('settingsHint') }}
          </p>

          <div class="settings-grid" :class="{ disabled: !datasetMeta }">
            <label>
              <span>{{ t('targetColumn') }}</span>
              <select v-model="targetColumn" :disabled="!datasetMeta">
                <option value="">{{ t('optional') }}</option>
                <option v-for="column in datasetColumns" :key="`target-${column.name}`" :value="column.name">
                  {{ column.name }} ({{ formatLabel(column.kind) }})
                </option>
              </select>
            </label>

            <label>
              <span>{{ t('timeColumn') }}</span>
              <select v-model="timeColumn" :disabled="!datasetMeta">
                <option value="">{{ t('optional') }}</option>
                <option v-for="column in datasetColumns" :key="`time-${column.name}`" :value="column.name">
                  {{ column.name }} ({{ formatLabel(column.kind) }})
                </option>
              </select>
            </label>

            <label>
              <span>{{ t('valueColumn') }}</span>
              <select v-model="valueColumn" :disabled="!datasetMeta">
                <option value="">{{ t('optional') }}</option>
                <option v-for="column in (datasetMeta ? datasetMeta.numericColumns : [])" :key="`value-${column}`" :value="column">
                  {{ column }}
                </option>
              </select>
            </label>

            <div class="text-picker">
              <span>{{ t('textColumns') }}</span>
              <div class="mini-chip-wrap" v-if="datasetMeta && datasetMeta.textColumns.length">
                <button
                  v-for="column in datasetMeta.textColumns"
                  :key="`text-${column}`"
                  type="button"
                  class="mini-chip"
                  :class="{ active: textColumns.includes(column) }"
                  @click="textColumns = textColumns.includes(column) ? textColumns.filter((item) => item !== column) : [...textColumns, column]"
                >
                  {{ column }}
                </button>
              </div>
              <p v-else class="settings-empty-inline">
                {{ datasetMeta ? t('noTextColumns') : t('textColumnsAfterUpload') }}
              </p>
            </div>
          </div>
        </section>

        <section class="sidebar-card">
          <div class="card-heading">
            <h2>{{ t('usefulLinks') }}</h2>
          </div>
          <div class="link-stack">
            <a
              v-for="link in usefulLinks"
              :key="link.label"
              class="useful-link"
              :href="link.href"
              target="_blank"
              rel="noreferrer"
            >
              <strong>{{ link.label }}</strong>
              <span>{{ link.caption }}</span>
            </a>
          </div>
        </section>
      </aside>
    </main>
  </div>
</template>

<style scoped>
.copilot-shell {
  height: 100dvh;
  max-height: 100dvh;
  padding: 0.5rem;
  color: var(--ink);
  font-size: 14px;
  overflow: hidden;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 0.5rem;
}

.tool-banner,
.sidebar-card,
.chat-frame,
.message-card,
.analysis-result,
.attach-drop,
.composer {
  border: 1px solid var(--line);
  background: var(--panel);
  box-shadow: var(--shadow);
  backdrop-filter: blur(12px);
}

.tool-banner {
  width: 100%;
  max-width: none;
  margin: 0;
  padding: 0.35rem 0.8rem;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  min-height: 46px;
  background:
    radial-gradient(circle at top right, rgba(235, 132, 72, 0.14), transparent 25%),
    radial-gradient(circle at top left, rgba(52, 133, 106, 0.16), transparent 32%),
    var(--panel);
}

.banner-copy {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex: 1;
}

.language-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  flex-wrap: wrap;
  justify-content: flex-end;
  min-width: 0;
}

.language-label {
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 700;
}

.language-chip {
  border: 1px solid rgba(16, 35, 28, 0.12);
  border-radius: 999px;
  padding: 0.24rem 0.6rem;
  background: rgba(255, 255, 255, 0.65);
  color: var(--ink);
  font-size: 0.75rem;
  font-weight: 700;
}

.language-chip.active {
  background: rgba(43, 133, 103, 0.14);
  border-color: rgba(43, 133, 103, 0.26);
  color: #1c5e47;
}

.banner-copy h1,
.chat-header h2,
.card-heading h2,
.analysis-result h3,
.result-table h4 {
  margin: 0;
  line-height: 1.05;
}

.banner-copy h1 {
  font-size: 1.05rem;
  white-space: nowrap;
}

.banner-copy p {
  margin: 0;
  max-width: none;
  color: var(--muted);
  font-size: 0.76rem;
}

.banner-kicker,
.chat-kicker,
.fact-label {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.66rem;
  font-weight: 700;
  color: var(--accent-strong);
  white-space: nowrap;
}

.sidebar-tool small,
.useful-link span,
.prompt-link,
.attach-caption,
.composer-file,
.result-headline {
  color: var(--muted);
}

.workspace {
  max-width: 1488px;
  margin: 0 auto;
  width: 100%;
  display: grid;
  grid-template-columns: 224px minmax(0, 1fr) 224px;
  gap: 0.5rem;
  align-items: stretch;
  min-height: 0;
  height: 100%;
  max-height: 100%;
}

.sidebar {
  display: grid;
  gap: 0.75rem;
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  align-content: start;
  min-height: 0;
  min-width: 0;
}

.sidebar-right {
  position: sticky;
  top: 0.75rem;
}

.sidebar-left,
.sidebar-right {
  width: 100%;
}

.sidebar-tool-list {
  display: grid;
  gap: 0.45rem;
}

.sidebar-tool {
  width: 100%;
  border: 1px solid rgba(16, 35, 28, 0.1);
  border-radius: 12px;
  padding: 0.5rem 0.6rem;
  text-align: left;
  background: rgba(255, 255, 255, 0.55);
  color: var(--ink);
  display: grid;
  gap: 0.1rem;
  font-size: 0.8rem;
}

.sidebar-tool span {
  font-weight: 700;
}

.sidebar-tool.active {
  background: linear-gradient(180deg, rgba(255, 241, 232, 0.96), rgba(255, 224, 208, 0.92));
  border-color: rgba(214, 98, 55, 0.45);
}

.sidebar-card {
  border-radius: 18px;
  padding: 0.7rem;
  min-width: 0;
}

.card-heading,
.chat-header,
.analysis-result-head,
.composer-footer,
.composer-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  min-width: 0;
  flex-wrap: wrap;
}

.card-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  padding: 0.28rem 0.58rem;
  background: rgba(43, 133, 103, 0.12);
  color: #1c5e47;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: capitalize;
}

.card-pill.muted,
.card-pill.skipped {
  background: rgba(16, 35, 28, 0.08);
  color: var(--muted);
}

.card-pill.error {
  background: rgba(190, 79, 42, 0.14);
  color: #8b351a;
}

.attach-drop {
  display: grid;
  gap: 0.35rem;
  margin-top: 0.7rem;
  padding: 0.8rem;
  border-radius: 14px;
  border-style: dashed;
  cursor: pointer;
}

.attach-drop input,
.attach-inline input {
  display: none;
}

.attach-title {
  font-weight: 700;
}

.example-picker {
  display: grid;
  gap: 0.5rem;
  margin-top: 0.65rem;
}

.example-picker label {
  display: grid;
  gap: 0.35rem;
}

.example-caption {
  margin: 0.5rem 0 0;
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.45;
}

.dataset-facts,
.settings-grid,
.link-stack,
.analysis-result-stack {
  display: grid;
  gap: 0.6rem;
}

.dataset-facts {
  margin-top: 0.8rem;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.dataset-fact {
  border-radius: 12px;
  padding: 0.65rem;
  background: rgba(255, 255, 255, 0.58);
  min-width: 0;
}

.dataset-fact-file {
  grid-column: 1 / -1;
}

.dataset-fact strong {
  display: block;
  min-width: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.dataset-facts,
.settings-grid,
.link-stack {
  font-size: 0.88rem;
}

.settings-grid label,
.text-picker {
  display: grid;
  gap: 0.4rem;
}

.settings-grid.disabled {
  opacity: 0.72;
}

.settings-grid span,
.text-picker span {
  font-weight: 600;
}

.settings-empty,
.settings-empty-inline {
  margin: 0;
  color: var(--muted);
  font-size: 0.8rem;
  line-height: 1.45;
}

.settings-empty {
  margin-bottom: 0.65rem;
}

select,
.composer-input {
  border: 1px solid rgba(16, 35, 28, 0.12);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.68);
  color: var(--ink);
}

select {
  padding: 0.58rem 0.7rem;
  font-size: 0.84rem;
}

.mini-chip-wrap,
.selected-tool-strip,
.analysis-tools,
.message-stats,
.chat-status {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.mini-chip {
  border: 1px solid rgba(16, 35, 28, 0.12);
  border-radius: 999px;
  padding: 0.25rem 0.55rem;
  background: rgba(255, 255, 255, 0.62);
  color: var(--ink);
  font-size: 0.72rem;
}

.mini-chip.active {
  background: rgba(43, 133, 103, 0.12);
  border-color: rgba(43, 133, 103, 0.2);
}

.prompt-link,
.useful-link {
  border: 1px solid rgba(16, 35, 28, 0.09);
  border-radius: 12px;
  padding: 0.68rem 0.75rem;
  background: rgba(255, 255, 255, 0.58);
  text-align: left;
  text-decoration: none;
  font-size: 0.82rem;
  min-width: 0;
}

.useful-link {
  display: grid;
  gap: 0.2rem;
  color: var(--ink);
  overflow-wrap: anywhere;
}

.chat-panel {
  min-width: 0;
  height: 100%;
  min-height: 0;
}

.chat-frame {
  height: 100%;
  min-height: 0;
  border-radius: 22px;
  padding: 0.7rem;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 0.55rem;
  background:
    linear-gradient(180deg, rgba(255, 250, 241, 0.92), rgba(252, 245, 236, 0.9));
}

.chat-history {
  overflow: auto;
  padding-right: 0.2rem;
  display: grid;
  gap: 0.75rem;
}

.message-row {
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr);
  gap: 0.55rem;
  align-items: start;
}

.message-row.user {
  grid-template-columns: minmax(0, 1fr) 40px;
}

.message-row.user .message-card {
  order: 1;
}

.message-row.user .message-avatar {
  order: 2;
}

.message-avatar {
  width: 40px;
  height: 40px;
  border-radius: 13px;
  display: grid;
  place-items: center;
  font-size: 0.68rem;
  font-weight: 800;
  background: rgba(16, 35, 28, 0.08);
  color: var(--ink);
}

.message-row.assistant .message-avatar {
  background: rgba(43, 133, 103, 0.14);
  color: #1f634b;
}

.message-row.user .message-avatar {
  background: rgba(214, 98, 55, 0.14);
  color: #9b4123;
}

.message-card {
  border-radius: 18px;
  padding: 0.8rem 0.85rem;
}

.message-card.user {
  background: linear-gradient(180deg, rgba(255, 234, 224, 0.92), rgba(255, 223, 209, 0.92));
}

.message-card.assistant {
  background: rgba(255, 255, 255, 0.78);
}

.message-card.error {
  border-color: rgba(190, 79, 42, 0.2);
  background: rgba(255, 239, 234, 0.94);
}

.message-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.4rem;
  font-size: 0.68rem;
  color: var(--muted);
}

.message-tools {
  color: var(--accent-strong);
}

.message-text,
.result-headline {
  margin: 0;
  white-space: pre-wrap;
  font-size: 0.9rem;
}

.summary-block {
  margin: 0;
  white-space: pre-wrap;
  font-family: inherit;
  line-height: 1.7;
  font-size: 0.9rem;
}

.summary-block.secondary {
  color: var(--muted);
}

.answer-panel,
.summary-panel {
  display: grid;
  gap: 0.35rem;
  margin-bottom: 0.75rem;
}

.analysis-label {
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent-strong);
}

.table-shell {
  margin-top: 0.65rem;
  overflow: auto;
  border-radius: 14px;
  border: 1px solid rgba(16, 35, 28, 0.08);
  background: rgba(255, 255, 255, 0.72);
}

.insight-list {
  margin: 0.55rem 0 0;
  padding-left: 1.2rem;
}

.result-notes {
  margin-top: 0.65rem;
  padding: 0.6rem 0.75rem;
  border-radius: 12px;
  background: rgba(214, 98, 55, 0.09);
  color: #8b351a;
}

.composer {
  border-radius: 18px;
  padding: 0.65rem;
  display: grid;
  gap: 0.5rem;
}

.attach-inline {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  border-radius: 999px;
  padding: 0.32rem 0.68rem;
  background: rgba(43, 133, 103, 0.1);
  color: #1f634b;
  font-weight: 700;
  cursor: pointer;
  font-size: 0.84rem;
  border: none;
}

.attach-inline-secondary,
.secondary-button {
  background: rgba(16, 35, 28, 0.07);
  color: var(--ink);
}

.secondary-button {
  border: 1px solid rgba(16, 35, 28, 0.1);
  border-radius: 12px;
  padding: 0.5rem 0.7rem;
  font-size: 0.8rem;
  font-weight: 700;
}

.attach-inline:disabled,
.secondary-button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.composer-input {
  width: 100%;
  resize: vertical;
  min-height: 66px;
  max-height: 110px;
  padding: 0.65rem 0.75rem;
  font-size: 0.88rem;
}

.send-button {
  border: none;
  border-radius: 14px;
  padding: 0.7rem 1rem;
  background: linear-gradient(135deg, var(--accent), var(--accent-strong));
  color: white;
  font-weight: 800;
  box-shadow: 0 14px 26px rgba(190, 79, 42, 0.22);
  font-size: 0.84rem;
}

.send-button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
  box-shadow: none;
}

@media (max-width: 1100px) {
  .copilot-shell {
    height: auto;
    max-height: none;
    overflow: visible;
    display: block;
  }

  .tool-banner {
    display: block;
  }

  .banner-copy {
    display: grid;
    gap: 0.2rem;
  }

  .workspace {
    grid-template-columns: 1fr;
    height: auto;
    max-height: none;
  }

  .sidebar-right {
    position: static;
  }

  .sidebar {
    height: auto;
    overflow: visible;
  }

  .chat-frame {
    min-height: auto;
    max-height: none;
    height: auto;
  }
}

@media (max-width: 720px) {
  .copilot-shell {
    padding: 0.75rem;
  }

  .tool-banner,
  .chat-frame,
  .sidebar-card,
  .message-card,
  .composer {
    border-radius: 18px;
  }

  .dataset-facts {
    grid-template-columns: 1fr 1fr;
  }

  .message-row,
  .message-row.user {
    grid-template-columns: 1fr;
  }

  .message-row.user .message-card,
  .message-row.user .message-avatar {
    order: initial;
  }

  .message-avatar {
    display: none;
  }

  .composer-footer,
  .chat-header,
  .card-heading {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
