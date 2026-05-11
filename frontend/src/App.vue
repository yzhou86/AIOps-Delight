<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'

function detectInitialLocale() {
  if (typeof window === 'undefined') return 'en'
  const stored = window.localStorage.getItem('datapilot-locale')
  if (stored === 'zh' || stored === 'en') return stored
  return window.navigator.language?.toLowerCase().startsWith('zh') ? 'zh' : 'en'
}

const COPY = {
  en: {
    appName: 'SciPilot',
    workspace: 'Workspace',
    admin: 'Admin',
    login: 'Login',
    logout: 'Logout',
    languageLabel: 'Language',
    english: 'English',
    chinese: '中文',
    loadingApp: 'Loading SciPilot...',
    loginTitle: 'Welcome to SciPilot',
    loginSubtitle: 'AI-native data and research copilot for students and researchers.',
    loginHint: 'Preview account: guest / guest. Use your assigned account for full access.',
    audienceBadge: 'For Students & Researchers',
    phaseBadge: 'Phase 1',
    heroTitle: 'Intelligent data Q&A today. Research exploration next.',
    heroBody: 'SciPilot is designed for students, labs, and independent researchers who want to talk to datasets in plain language, inspect evidence, and move from questions to findings faster.',
    featureNowTitle: 'Smart data Q&A',
    featureNowBody: 'Upload CSV or Excel files, ask questions in natural language, and get answers, charts, and analysis tables in one thread.',
    featureResearchTitle: 'Research exploration',
    featureResearchBody: 'Upcoming workflows will help you compare signals, inspect sources, and structure exploratory research more systematically.',
    featureDeepTitle: 'Deep researcher',
    featureDeepBody: 'Future releases will expand into deeper multi-step research, synthesis, and evidence-driven investigation.',
    sponsorTitle: 'LLM Tokens Powered by Fastoken',
    sponsorBody: 'Our token access is provided by fastoken.ai. If you need stable OpenAI-compatible access for deployment or experimentation, this is our recommended provider.',
    sponsorCta: 'Visit fastoken.ai',
    username: 'Username',
    password: 'Password',
    signIn: 'Sign in',
    signingIn: 'Signing in...',
    guestSignIn: 'Preview as guest',
    loginError: 'Login failed.',
    guestBadge: 'Preview',
    guestModeNotice: 'Guest mode is read-only. Upload, tool changes, and new agent conversations are disabled.',
    guestDatasetCaption: 'Guest preview is locked to the built-in anomaly demo dataset.',
    guestPromptPlaceholder: 'Guest preview is read-only. Sign in with a full account to ask your own questions.',
    guestWelcome: 'Guest preview loaded. You can inspect the fixed anomaly demo, but you cannot upload files or start a new chat.',
    guestPasswordHint: 'Guest preview accounts cannot change password.',
    currentUser: 'Current user',
    roleAdmin: 'Admin',
    roleUser: 'User',
    changePassword: 'Change password',
    currentPassword: 'Current password',
    newPassword: 'New password',
    savePassword: 'Save password',
    saving: 'Saving...',
    passwordSaved: 'Password updated.',
    tools: 'Tools',
    autoMode: 'Auto',
    manualMode: 'Manual',
    autoModeHint: 'Let the agent choose the right tools from the data and your question.',
    autoModeBadge: 'AI auto',
    activeShort: 'active',
    dataset: 'Dataset',
    ready: 'Ready',
    waiting: 'Waiting',
    attachTitle: 'Attach CSV / XLS / XLSX',
    attachInspecting: 'Inspecting file...',
    attachCaption: 'The app profiles your file first, then the agent can analyze it.',
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
    llmStatus: 'LLM status',
    llmFallback: 'Fallback answer',
    llmErrorPrefix: 'LLM error',
    visuals: 'Visual Insights',
    attachFile: 'Attach file',
    loadingExample: 'Loading example...',
    useSelectedExample: 'Use selected example',
    noDatasetAttached: 'No dataset attached',
    promptPlaceholder: 'Ask the agent to analyze the dataset, compare segments, forecast a metric, or explain anomalies.',
    analyzing: 'Analyzing...',
    send: 'Send to agent',
    exportPdf: 'Export PDF',
    exportingPdf: 'Exporting PDF...',
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
    exportFailed: 'PDF export failed.',
    runningTools: 'Running the selected tools and composing the analysis...',
    datasetHeadline: '{file} inspected with {rows} rows and {columns} columns.',
    statusOk: 'ok',
    statusSkipped: 'skipped',
    statusError: 'error',
    statusLoading: 'loading',
    welcome: 'Upload a CSV or Excel file, or choose a built-in example dataset, then ask the agent to analyze it. I will keep the full conversation history here.',
    defaultPrompt: 'Find the strongest patterns in this dataset, call out anomalies, and recommend what I should investigate next.',
    adminTitle: 'Admin Console',
    adminSubtitle: 'Manage users, local credentials, and LLM provider settings stored in SQLite.',
    llmConfig: 'LLM Configuration',
    provider: 'Provider',
    providerAuto: 'Auto',
    providerQwen: 'Qwen / DashScope',
    providerOpenAI: 'OpenAI-compatible',
    qwenKey: 'Qwen API key',
    qwenModel: 'Qwen model',
    openaiKey: 'OpenAI-compatible API key',
    openaiBaseUrl: 'OpenAI-compatible base URL',
    openaiModel: 'OpenAI-compatible model',
    saveConfig: 'Save config',
    configSaved: 'LLM configuration saved.',
    userManagement: 'User Management',
    createUser: 'Create user',
    createUserHint: 'Admins can add normal users and set their passwords here.',
    create: 'Create',
    role: 'Role',
    userList: 'Users',
    resetPassword: 'Reset password',
    reset: 'Reset',
    createdAt: 'Created',
    updatedAt: 'Updated',
    myAccount: 'My Account',
    adminOnly: 'Admin only',
    sessionExpired: 'Your session expired. Please sign in again.',
    unauthorized: 'You do not have access to that action.',
    apiError: 'Something went wrong.',
    refreshData: 'Refresh data',
    selfPasswordHint: 'Change your own password here.',
    loginToContinue: 'Please sign in to continue.',
    userCreated: 'User {username} created.',
    userPasswordUpdated: '{username} password updated.'
  },
  zh: {
    appName: 'SciPilot',
    workspace: '工作台',
    admin: '管理台',
    login: '登录',
    logout: '退出',
    languageLabel: '语言',
    english: 'English',
    chinese: '中文',
    loadingApp: '正在加载 SciPilot...',
    loginTitle: '欢迎来到 SciPilot',
    loginSubtitle: '面向学生与科研人员的 AI 数据与研究助手。',
    loginHint: '预览账号：guest / guest。完整功能请使用分配给你的正式账号登录。',
    audienceBadge: '学生与研究者',
    phaseBadge: '第一期',
    heroTitle: '先把智能问数做到极致，再走向科研探索。',
    heroBody: 'SciPilot 面向学生、实验室和科研人员，帮助你直接和数据对话，用自然语言提问、查看证据、生成图表，并更快从问题走到结论。',
    featureNowTitle: '智能问数',
    featureNowBody: '上传 CSV 或 Excel，直接提问，获得文字回答、分析图表和结果表格，形成连续会话。',
    featureResearchTitle: '科研探索',
    featureResearchBody: '后续版本会支持更系统的科研探索流程，帮助你比较信号、梳理资料、组织研究思路。',
    featureDeepTitle: 'Deep Researcher',
    featureDeepBody: '未来会进一步扩展为更深入的多步研究与证据综合能力，支持更复杂的研究任务。',
    sponsorTitle: 'Fastoken 提供 LLM Token 支持',
    sponsorBody: '我们的 token 来源由 fastoken.ai 提供。如果你需要稳定的 OpenAI 兼容访问能力用于部署或实验，这是我们推荐的服务。',
    sponsorCta: '访问 fastoken.ai',
    username: '用户名',
    password: '密码',
    signIn: '登录',
    signingIn: '登录中...',
    guestSignIn: '一键访客预览',
    loginError: '登录失败。',
    guestBadge: '预览',
    guestModeNotice: '访客模式为只读预览，不允许上传文件、切换工具或发起新的智能问数对话。',
    guestDatasetCaption: '访客预览固定锁定在内置异常检测演示数据集。',
    guestPromptPlaceholder: '访客模式为只读预览。如需自行提问，请使用正式账号登录。',
    guestWelcome: '访客预览已加载。你可以查看固定的异常检测演示结果，但不能上传文件或发起新的对话。',
    guestPasswordHint: '访客预览账号不支持修改密码。',
    currentUser: '当前用户',
    roleAdmin: '管理员',
    roleUser: '普通用户',
    changePassword: '修改密码',
    currentPassword: '当前密码',
    newPassword: '新密码',
    savePassword: '保存密码',
    saving: '保存中...',
    passwordSaved: '密码已更新。',
    tools: '工具',
    autoMode: '自动',
    manualMode: '手动',
    autoModeHint: '由智能体根据数据结构和你的问题自动选择最合适的工具。',
    autoModeBadge: '智能自动',
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
    llmStatus: 'LLM 状态',
    llmFallback: '回退回答',
    llmErrorPrefix: 'LLM 错误',
    visuals: '可视化图表',
    attachFile: '上传文件',
    loadingExample: '正在加载示例...',
    useSelectedExample: '使用当前示例',
    noDatasetAttached: '尚未附加数据集',
    promptPlaceholder: '让智能体分析数据、比较分群、预测指标走势，或者解释异常原因。',
    analyzing: '分析中...',
    send: '发送给智能体',
    exportPdf: '导出 PDF',
    exportingPdf: '正在导出 PDF...',
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
    exportFailed: 'PDF 导出失败。',
    runningTools: '正在运行所选工具并整理分析结果...',
    datasetHeadline: '已完成对 {file} 的探查，共有 {rows} 行、{columns} 列。',
    statusOk: '成功',
    statusSkipped: '已跳过',
    statusError: '错误',
    statusLoading: '处理中',
    welcome: '上传 CSV 或 Excel 文件，或者直接选择内置示例数据集，然后用自然语言让智能体分析。我会在这里保留完整对话历史。',
    defaultPrompt: '请找出这个数据集里最强的模式、异常点，以及接下来最值得我调查的方向。',
    adminTitle: '管理后台',
    adminSubtitle: '在这里管理用户、本地账号密码，以及写入 SQLite 的 LLM 配置。',
    llmConfig: 'LLM 配置',
    provider: '提供方',
    providerAuto: '自动',
    providerQwen: '千问 / DashScope',
    providerOpenAI: 'OpenAI 兼容接口',
    qwenKey: '千问 API Key',
    qwenModel: '千问模型',
    openaiKey: 'OpenAI 兼容 API Key',
    openaiBaseUrl: 'OpenAI 兼容 Base URL',
    openaiModel: 'OpenAI 兼容模型',
    saveConfig: '保存配置',
    configSaved: 'LLM 配置已保存。',
    userManagement: '用户管理',
    createUser: '新增用户',
    createUserHint: '管理员可以在这里新增普通用户并设置密码。',
    create: '创建',
    role: '角色',
    userList: '用户列表',
    resetPassword: '重置密码',
    reset: '重置',
    createdAt: '创建时间',
    updatedAt: '更新时间',
    myAccount: '我的账号',
    adminOnly: '仅管理员可见',
    sessionExpired: '登录状态已失效，请重新登录。',
    unauthorized: '你没有权限执行这个操作。',
    apiError: '发生了一个错误。',
    refreshData: '刷新数据',
    selfPasswordHint: '在这里修改你自己的密码。',
    loginToContinue: '请先登录后继续。',
    userCreated: '用户 {username} 已创建。',
    userPasswordUpdated: '用户 {username} 的密码已更新。'
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
    { label: 'Scikit-learn', href: 'https://scikit-learn.org/stable/', caption: 'Reference for clustering, anomaly detection, and classification.' }
  ],
  zh: [
    { label: '工具目录 API', href: '/api/tools', caption: '以 JSON 形式查看当前可用分析工具。' },
    { label: '健康检查', href: '/api/health', caption: '确认统一 Flask 应用运行正常。' },
    { label: 'Pandas 文档', href: 'https://pandas.pydata.org/docs/', caption: '适合数据表分析和清洗时参考。' },
    { label: 'Scikit-learn 文档', href: 'https://scikit-learn.org/stable/', caption: '聚类、异常检测和分类建模的参考资料。' }
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

const LOGIN_FEATURES = {
  en: [
    { badge: 'Now', title: 'Smart data Q&A', body: 'Natural-language analysis over uploaded datasets, with charts and tabular evidence.' },
    { badge: 'Next', title: 'Research exploration', body: 'Structured workflows for comparing ideas, signals, sources, and early findings.' },
    { badge: 'Future', title: 'Deep researcher', body: 'Longer-horizon synthesis and deeper evidence gathering for complex research tasks.' }
  ],
  zh: [
    { badge: '现在', title: '智能问数', body: '围绕上传数据进行自然语言分析，并联动图表和表格证据。' },
    { badge: '下一步', title: '科研探索', body: '支持更结构化的问题比较、信号梳理、资料对照和初步研究分析。' },
    { badge: '未来', title: 'Deep Researcher', body: '面向更复杂研究任务的长链路综合分析与证据收集能力。' }
  ]
}

const CHART_FRAME = {
  width: 520,
  height: 230,
  left: 42,
  right: 18,
  top: 14,
  bottom: 30
}

const locale = ref(detectInitialLocale())
const sessionReady = ref(false)
const currentUser = ref(null)
const appView = ref('workspace')

const tools = ref([])
const examples = ref([])
const toolMode = ref('manual')
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
const exportLoading = ref(false)
const chatViewport = ref(null)
const nextMessageId = ref(1)
const messages = ref([])

const loginUsername = ref('')
const loginPassword = ref('')
const authLoading = ref(false)
const authError = ref('')

const accountCurrentPassword = ref('')
const accountNewPassword = ref('')
const accountLoading = ref(false)
const accountNotice = ref('')
const accountError = ref('')

const adminUsers = ref([])
const adminLoading = ref(false)
const adminNotice = ref('')
const adminError = ref('')
const newUserUsername = ref('')
const newUserPassword = ref('')
const passwordDrafts = ref({})
const llmForm = ref({
  provider: 'auto',
  qwenApiKey: '',
  qwenModel: 'qwen-turbo',
  openaiApiKey: '',
  openaiBaseUrl: 'https://api.openai.com/v1',
  openaiModel: 'gpt-4o-mini'
})

const copy = computed(() => COPY[locale.value])
const loginFeatures = computed(() => LOGIN_FEATURES[locale.value])
const usefulLinks = computed(() => USEFUL_LINKS[locale.value])
const suggestedPrompts = computed(() => SUGGESTED_PROMPTS[locale.value])
const isAdmin = computed(() => currentUser.value?.role === 'admin')
const isGuest = computed(() => currentUser.value?.username === 'guest')
const selectedToolDetails = computed(() => tools.value.filter((tool) => selectedTools.value.includes(tool.id)))
const selectedToolNames = computed(() => (toolMode.value === 'auto' ? [t('autoModeBadge')] : selectedToolDetails.value.map((tool) => localizedToolName(tool))))
const datasetColumns = computed(() => datasetMeta.value?.columns || [])
const selectedExample = computed(() => examples.value.find((example) => example.id === selectedExampleId.value) || null)
const canAnalyze = computed(() =>
  !isGuest.value &&
  Boolean(datasetMeta.value?.datasetId) &&
  (toolMode.value === 'auto' || selectedTools.value.length > 0) &&
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

function resetMessages() {
  nextMessageId.value = 1
  messages.value = [
    {
      id: 1,
      role: 'assistant',
      kind: 'welcome',
      text: t('welcome')
    }
  ]
}

function resetWorkspaceState() {
  datasetFile.value = null
  datasetMeta.value = null
  selectedExampleId.value = ''
  targetColumn.value = ''
  timeColumn.value = ''
  valueColumn.value = ''
  textColumns.value = []
  toolMode.value = 'manual'
  selectedTools.value = ['data_profile', 'correlation_explorer', 'anomaly_detector']
  prompt.value = t('defaultPrompt')
  resetMessages()
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

function setToolMode(mode) {
  if (mode !== 'auto' && mode !== 'manual') return
  toolMode.value = mode
}

function createMessage(role, kind, payload = {}) {
  nextMessageId.value += 1
  return { id: nextMessageId.value, role, kind, ...payload }
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
    messages.value[index] = { ...messages.value[index], ...patch }
  }
  scrollToBottom()
}

function changeLanguage(nextLocale) {
  if (nextLocale !== 'zh' && nextLocale !== 'en') return
  const previousLocale = locale.value
  locale.value = nextLocale
  if (typeof window !== 'undefined') {
    window.localStorage.setItem('datapilot-locale', nextLocale)
  }
  if (!prompt.value || prompt.value === COPY[previousLocale].defaultPrompt) {
    prompt.value = COPY[nextLocale].defaultPrompt
  }
  const welcomeMessage = messages.value.find((message) => message.kind === 'welcome')
  if (welcomeMessage) {
    welcomeMessage.text = COPY[nextLocale].welcome
  }
  if (currentUser.value?.username === 'guest') {
    loadGuestPreview()
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

function roleChip(role) {
  return role === 'admin' ? t('roleAdmin') : t('roleUser')
}

function llmProviderLabel(provider) {
  if (provider === 'qwen') return t('providerQwen')
  if (provider === 'openai_compatible') return t('providerOpenAI')
  return t('providerAuto')
}

function llmStatusLine(analysis) {
  const answerMeta = analysis?.llm?.answer
  const runtimeMeta = analysis?.llm?.runtime
  const source = analysis?.llm?.answerSource
  const provider = answerMeta?.provider || runtimeMeta?.provider
  const model = answerMeta?.model || runtimeMeta?.model
  const parts = [t('llmStatus') + ':']
  if (provider) parts.push(llmProviderLabel(provider))
  if (model) parts.push(model)
  if (source === 'fallback') parts.push(`(${t('llmFallback')})`)
  return parts.join(' ')
}

function llmErrorLine(analysis) {
  const error = analysis?.llm?.answer?.error || analysis?.llm?.summary?.error
  if (!error) return ''
  return `${t('llmErrorPrefix')}: ${error}`
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

function toggleTool(toolId) {
  if (toolMode.value === 'auto') return
  if (selectedTools.value.includes(toolId)) {
    selectedTools.value = selectedTools.value.filter((id) => id !== toolId)
  } else {
    selectedTools.value = [...selectedTools.value, toolId]
  }
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    credentials: 'same-origin',
    ...options
  })

  let data = {}
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    data = await response.json()
  } else {
    const text = await response.text()
    data = text ? { text } : {}
  }

  if (response.status === 401 && url !== '/api/auth/login' && url !== '/api/auth/me') {
    currentUser.value = null
    sessionReady.value = true
    authError.value = t('sessionExpired')
    resetWorkspaceState()
  }

  if (response.status === 403) {
    adminError.value = t('unauthorized')
  }

  return { response, data }
}

async function fetchSession() {
  const { response, data } = await apiFetch('/api/auth/me')
  if (!response.ok) return
  currentUser.value = data.user || null
  if (currentUser.value) {
    await hydrateAfterLogin()
  } else {
    resetWorkspaceState()
  }
}

async function hydrateAfterLogin() {
  await Promise.all([fetchTools(), fetchExamples()])
  if (isGuest.value) {
    await loadGuestPreview()
    return
  }
  if (isAdmin.value && appView.value === 'admin') {
    await loadAdminData()
  }
}

async function loginWithCredentials(username, password) {
  authLoading.value = true
  authError.value = ''
  try {
    const { response, data } = await apiFetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username,
        password
      })
    })
    if (!response.ok) {
      throw new Error(data.error || t('loginError'))
    }
    currentUser.value = data.user
    loginPassword.value = ''
    authError.value = ''
    resetWorkspaceState()
    await hydrateAfterLogin()
  } catch (error) {
    authError.value = error.message
  } finally {
    authLoading.value = false
  }
}

async function login() {
  await loginWithCredentials(loginUsername.value, loginPassword.value)
}

async function loginAsGuest() {
  loginUsername.value = 'guest'
  loginPassword.value = ''
  await loginWithCredentials('guest', 'guest')
}

async function logout() {
  await apiFetch('/api/auth/logout', { method: 'POST' })
  currentUser.value = null
  appView.value = 'workspace'
  authError.value = ''
  accountNotice.value = ''
  adminNotice.value = ''
  resetWorkspaceState()
}

async function fetchTools() {
  const { response, data } = await apiFetch('/api/tools')
  if (response.ok) {
    tools.value = data.tools || []
  }
}

async function fetchExamples() {
  const { response, data } = await apiFetch('/api/examples')
  if (response.ok) {
    examples.value = data.examples || []
    if (isGuest.value && data.examples?.length) {
      selectedExampleId.value = data.examples[0].id
    }
  }
}

async function loadGuestPreview() {
  if (!isGuest.value) return
  inspectLoading.value = true
  try {
    const { response, data } = await apiFetch(`/api/guest-demo?language=${encodeURIComponent(locale.value)}`)
    if (!response.ok) {
      throw new Error(data.error || t('apiError'))
    }
    resetMessages()
    applyDatasetMeta(data.dataset)
    selectedExampleId.value = data.exampleId || 'service_health_anomalies'
    toolMode.value = 'manual'
    selectedTools.value = data.selectedTools || ['data_profile', 'anomaly_detector']
    prompt.value = ''
    pushMessage('system', 'info', {
      text: t('guestWelcome')
    })
    pushMessage('assistant', 'dataset', {
      text: datasetHeadline(data.dataset),
      dataset: data.dataset
    })
    pushMessage('assistant', 'analysis', {
      text: data.answer || data.summary,
      answer: data.answer || data.summary,
      summary: data.summary,
      analysis: data
    })
  } catch (error) {
    pushMessage('assistant', 'error', {
      text: error.message || t('apiError')
    })
  } finally {
    inspectLoading.value = false
  }
}

function applyDatasetMeta(data) {
  datasetMeta.value = data
  targetColumn.value = data.categoricalColumns?.[0] || ''
  timeColumn.value = data.datetimeColumns?.[0] || ''
  valueColumn.value = data.numericColumns?.[0] || ''
  textColumns.value = data.textColumns ? data.textColumns.slice(0, 1) : []
}

async function handleFileChange(event) {
  if (isGuest.value) return
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

async function inspectDataset() {
  if (isGuest.value) return
  if (!datasetFile.value) return
  inspectLoading.value = true
  const loadingId = pushMessage('assistant', 'loading', {
    text: t('inspectMessage', { file: datasetFile.value.name })
  })

  try {
    const formData = new FormData()
    formData.append('file', datasetFile.value)
    const { response, data } = await apiFetch('/api/datasets/inspect', {
      method: 'POST',
      body: formData
    })
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
  if (isGuest.value) return
  if (!selectedExampleId.value) return
  datasetFile.value = null
  inspectLoading.value = true
  const example = selectedExample.value
  const loadingId = pushMessage('assistant', 'loading', {
    text: t('loadExampleMessage', { file: example?.fileName || selectedExampleId.value })
  })

  try {
    const { response, data } = await apiFetch('/api/examples/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ exampleId: selectedExampleId.value })
    })
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
  if (isGuest.value) return
  if (!datasetMeta.value?.datasetId) {
    pushMessage('assistant', 'error', { text: t('uploadFirstError') })
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
  const loadingId = pushMessage('assistant', 'loading', { text: t('runningTools') })

  try {
    const { response, data } = await apiFetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        datasetId: datasetMeta.value.datasetId,
        selectedTools: selectedTools.value,
        toolMode: toolMode.value,
        prompt: userPrompt,
        targetColumn: targetColumn.value || null,
        timeColumn: timeColumn.value || null,
        valueColumn: valueColumn.value || null,
        textColumns: textColumns.value,
        language: locale.value
      })
    })
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

    if (data.toolMode === 'auto' && data.resolvedContext) {
      targetColumn.value = data.resolvedContext.targetColumn || targetColumn.value
      timeColumn.value = data.resolvedContext.timeColumn || timeColumn.value
      valueColumn.value = data.resolvedContext.valueColumn || valueColumn.value
      textColumns.value = data.resolvedContext.textColumns?.length ? data.resolvedContext.textColumns : textColumns.value
    }
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

function exportableMessages() {
  return messages.value.map((message) => ({
    role: message.role,
    kind: message.kind,
    text: message.text || '',
    answer: message.answer || '',
    summary: message.summary || '',
    dataset: message.dataset || null,
    analysis: message.analysis || null
  }))
}

async function exportChatPdf() {
  exportLoading.value = true
  try {
    const fileStem = datasetMeta.value?.fileName
      ? datasetMeta.value.fileName.replace(/\.[^.]+$/, '')
      : 'chat'
    const title = locale.value === 'zh' ? 'SciPilot 对话导出' : 'SciPilot Chat Export'
    const response = await fetch('/api/export-chat-pdf', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
        language: locale.value,
        datasetName: datasetMeta.value?.fileName || '',
        fileName: `${fileStem}-chat-export.pdf`,
        messages: exportableMessages()
      })
    })
    if (response.status === 401) {
      currentUser.value = null
      authError.value = t('sessionExpired')
      resetWorkspaceState()
      throw new Error(t('sessionExpired'))
    }
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}))
      throw new Error(payload.error || t('exportFailed'))
    }
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${fileStem}-chat-export.pdf`
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
  } catch (error) {
    pushMessage('assistant', 'error', {
      text: error.message || t('exportFailed')
    })
  } finally {
    exportLoading.value = false
  }
}

async function changeOwnPassword() {
  if (isGuest.value) {
    accountError.value = t('guestPasswordHint')
    accountNotice.value = ''
    return
  }
  accountLoading.value = true
  accountNotice.value = ''
  accountError.value = ''
  try {
    const { response, data } = await apiFetch('/api/auth/change-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        currentPassword: accountCurrentPassword.value,
        newPassword: accountNewPassword.value
      })
    })
    if (!response.ok) {
      throw new Error(data.error || t('apiError'))
    }
    accountCurrentPassword.value = ''
    accountNewPassword.value = ''
    accountNotice.value = t('passwordSaved')
  } catch (error) {
    accountError.value = error.message
  } finally {
    accountLoading.value = false
  }
}

function setAppView(view) {
  if (view === 'admin' && !isAdmin.value) return
  appView.value = view
  if (view === 'admin' && isAdmin.value) {
    loadAdminData()
  }
}

async function loadAdminData() {
  if (!isAdmin.value) return
  adminLoading.value = true
  adminError.value = ''
  try {
    const [userResult, configResult] = await Promise.all([
      apiFetch('/api/admin/users'),
      apiFetch('/api/admin/llm-config')
    ])
    if (!userResult.response.ok) throw new Error(userResult.data.error || t('apiError'))
    if (!configResult.response.ok) throw new Error(configResult.data.error || t('apiError'))

    adminUsers.value = userResult.data.users || []
    passwordDrafts.value = Object.fromEntries(adminUsers.value.map((user) => [user.id, '']))
    llmForm.value = {
      provider: configResult.data.config?.provider || 'auto',
      qwenApiKey: configResult.data.config?.qwen_api_key || '',
      qwenModel: configResult.data.config?.qwen_model || 'qwen-turbo',
      openaiApiKey: configResult.data.config?.openai_api_key || '',
      openaiBaseUrl: configResult.data.config?.openai_base_url || 'https://api.openai.com/v1',
      openaiModel: configResult.data.config?.openai_model || 'gpt-4o-mini'
    }
  } catch (error) {
    adminError.value = error.message
  } finally {
    adminLoading.value = false
  }
}

async function createManagedUser() {
  adminNotice.value = ''
  adminError.value = ''
  try {
    const { response, data } = await apiFetch('/api/admin/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: newUserUsername.value,
        password: newUserPassword.value
      })
    })
    if (!response.ok) {
      throw new Error(data.error || t('apiError'))
    }
    newUserUsername.value = ''
    newUserPassword.value = ''
    adminNotice.value = t('userCreated', { username: data.user.username })
    await loadAdminData()
  } catch (error) {
    adminError.value = error.message
  }
}

async function resetManagedUserPassword(userId) {
  adminNotice.value = ''
  adminError.value = ''
  try {
    const { response, data } = await apiFetch(`/api/admin/users/${userId}/password`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        password: passwordDrafts.value[userId]
      })
    })
    if (!response.ok) {
      throw new Error(data.error || t('apiError'))
    }
    passwordDrafts.value[userId] = ''
    adminNotice.value = t('userPasswordUpdated', { username: data.user.username })
    await loadAdminData()
  } catch (error) {
    adminError.value = error.message
  }
}

async function saveLlmConfig() {
  adminNotice.value = ''
  adminError.value = ''
  try {
    const { response, data } = await apiFetch('/api/admin/llm-config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(llmForm.value)
    })
    if (!response.ok) {
      throw new Error(data.error || t('apiError'))
    }
    llmForm.value = {
      provider: data.config?.provider || 'auto',
      qwenApiKey: data.config?.qwen_api_key || '',
      qwenModel: data.config?.qwen_model || 'qwen-turbo',
      openaiApiKey: data.config?.openai_api_key || '',
      openaiBaseUrl: data.config?.openai_base_url || 'https://api.openai.com/v1',
      openaiModel: data.config?.openai_model || 'gpt-4o-mini'
    }
    adminNotice.value = t('configSaved')
  } catch (error) {
    adminError.value = error.message
  }
}

function formatChartValue(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return value
  if (Math.abs(num) >= 1000) return Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(num)
  if (Math.abs(num) >= 100) return num.toFixed(0)
  if (Math.abs(num) >= 10) return num.toFixed(1)
  return num.toFixed(2).replace(/\.00$/, '')
}

function shortenChartLabel(label) {
  const text = String(label ?? '')
  return text.length > 12 ? `${text.slice(0, 12)}…` : text
}

function barPoints(chart) {
  return chart?.series?.[0]?.data || []
}

function barWidth(value, chart) {
  const max = Math.max(...barPoints(chart).map((point) => Number(point.value) || 0), 1)
  return `${Math.max(6, ((Number(value) || 0) / max) * 100)}%`
}

function lineLabels(chart) {
  const labels = []
  ;(chart?.series || []).forEach((series) => {
    ;(series.data || []).forEach((point) => {
      if (!labels.includes(point.label)) labels.push(point.label)
    })
  })
  return labels
}

function lineDomain(chart) {
  const values = (chart?.series || [])
    .flatMap((series) => series.data || [])
    .map((point) => Number(point.value))
    .filter((value) => Number.isFinite(value))
  if (!values.length) return { min: 0, max: 1 }
  let min = Math.min(...values)
  let max = Math.max(...values)
  if (min === max) {
    min -= 1
    max += 1
  }
  return { min, max }
}

function lineTicks(chart) {
  const domain = lineDomain(chart)
  const step = (domain.max - domain.min) / 4
  return Array.from({ length: 5 }, (_, idx) => domain.min + step * idx)
}

function linePointX(chart, label) {
  const labels = lineLabels(chart)
  const plotWidth = CHART_FRAME.width - CHART_FRAME.left - CHART_FRAME.right
  if (labels.length <= 1) return CHART_FRAME.left + plotWidth / 2
  const index = Math.max(0, labels.indexOf(label))
  return CHART_FRAME.left + (plotWidth * index) / (labels.length - 1)
}

function linePointY(chart, value) {
  const domain = lineDomain(chart)
  const plotHeight = CHART_FRAME.height - CHART_FRAME.top - CHART_FRAME.bottom
  return CHART_FRAME.top + ((domain.max - Number(value)) / (domain.max - domain.min)) * plotHeight
}

function linePath(chart, series) {
  const points = (series?.data || []).filter((point) => Number.isFinite(Number(point.value)))
  if (!points.length) return ''
  return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${linePointX(chart, point.label)} ${linePointY(chart, point.value)}`).join(' ')
}

function lineBandPath(chart, band) {
  const points = (band?.data || []).filter((point) => Number.isFinite(Number(point.low)) && Number.isFinite(Number(point.high)))
  if (!points.length) return ''
  const upperPath = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${linePointX(chart, point.label)} ${linePointY(chart, point.high)}`).join(' ')
  const lowerPath = points.slice().reverse().map((point) => `L ${linePointX(chart, point.label)} ${linePointY(chart, point.low)}`).join(' ')
  return `${upperPath} ${lowerPath} Z`
}

function scatterPoints(chart) {
  return (chart?.series || []).flatMap((series) => series.data || [])
}

function scatterDomain(chart, key) {
  const values = scatterPoints(chart).map((point) => Number(point[key])).filter((value) => Number.isFinite(value))
  if (!values.length) return { min: 0, max: 1 }
  let min = Math.min(...values)
  let max = Math.max(...values)
  if (min === max) {
    min -= 1
    max += 1
  }
  return { min, max }
}

function scatterPointX(chart, value) {
  const domain = scatterDomain(chart, 'x')
  const plotWidth = CHART_FRAME.width - CHART_FRAME.left - CHART_FRAME.right
  return CHART_FRAME.left + ((Number(value) - domain.min) / (domain.max - domain.min)) * plotWidth
}

function scatterPointY(chart, value) {
  const domain = scatterDomain(chart, 'y')
  const plotHeight = CHART_FRAME.height - CHART_FRAME.top - CHART_FRAME.bottom
  return CHART_FRAME.top + ((domain.max - Number(value)) / (domain.max - domain.min)) * plotHeight
}

function scatterSizeDomain(chart) {
  const values = scatterPoints(chart).map((point) => Number(point.size)).filter((value) => Number.isFinite(value))
  if (!values.length) return { min: 0, max: 1 }
  let min = Math.min(...values)
  let max = Math.max(...values)
  if (min === max) min = 0
  return { min, max }
}

function scatterRadius(chart, point) {
  const raw = Number(point.size)
  if (!Number.isFinite(raw)) return chart?.variant === 'cluster3d' ? 8 : 4.8
  const domain = scatterSizeDomain(chart)
  const minRadius = chart?.variant === 'cluster3d' ? 7 : 4
  const maxRadius = chart?.variant === 'cluster3d' ? 15 : 10
  if (domain.max === domain.min) return (minRadius + maxRadius) / 2
  return minRadius + ((raw - domain.min) / (domain.max - domain.min)) * (maxRadius - minRadius)
}

function chartToken(value) {
  return String(value || 'chart').replace(/[^a-zA-Z0-9_-]+/g, '-')
}

function chartGradientId(chart, series, index) {
  return `${chartToken(chart.title)}-${chartToken(series.name)}-${index}-gradient`
}

function colorToRgb(color) {
  const hex = String(color || '#2c8f6b').replace('#', '')
  const full = hex.length === 3 ? hex.split('').map((item) => item + item).join('') : hex
  const int = Number.parseInt(full, 16)
  return { r: (int >> 16) & 255, g: (int >> 8) & 255, b: int & 255 }
}

function colorWithAlpha(color, alpha) {
  const { r, g, b } = colorToRgb(color)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

function lightenColor(color, factor = 0.35) {
  const { r, g, b } = colorToRgb(color)
  const mix = (channel) => Math.round(channel + (255 - channel) * factor)
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`
}

onMounted(async () => {
  resetMessages()
  try {
    await fetchSession()
  } finally {
    sessionReady.value = true
    scrollToBottom()
  }
})
</script>

<template>
  <div v-if="!sessionReady" class="boot-shell">
    <div class="boot-card">{{ t('loadingApp') }}</div>
  </div>

  <div v-else-if="!currentUser" class="auth-shell">
    <section class="auth-card">
      <div class="auth-showcase">
        <div class="auth-brand">
          <div class="auth-pills">
            <span class="hero-pill">{{ t('audienceBadge') }}</span>
            <span class="hero-pill hero-pill-warm">{{ t('phaseBadge') }}</span>
          </div>
          <p class="brand-kicker">{{ t('appName') }}</p>
          <h1>{{ t('heroTitle') }}</h1>
          <p>{{ t('heroBody') }}</p>
        </div>

        <div class="feature-grid">
          <article v-for="feature in loginFeatures" :key="feature.title" class="feature-card">
            <span class="feature-badge">{{ feature.badge }}</span>
            <h3>{{ feature.title }}</h3>
            <p>{{ feature.body }}</p>
          </article>
        </div>

        <a class="sponsor-card" href="https://fastoken.ai/" target="_blank" rel="noreferrer">
          <div>
            <p class="sponsor-kicker">fastoken.ai</p>
            <h3>{{ t('sponsorTitle') }}</h3>
            <p>{{ t('sponsorBody') }}</p>
          </div>
          <span class="sponsor-link">{{ t('sponsorCta') }}</span>
        </a>
      </div>

      <div class="auth-pane">
        <div class="language-toggle" role="group" :aria-label="t('languageLabel')">
          <span class="language-label">{{ t('languageLabel') }}</span>
          <button type="button" class="language-chip" :class="{ active: locale === 'en' }" @click="changeLanguage('en')">{{ t('english') }}</button>
          <button type="button" class="language-chip" :class="{ active: locale === 'zh' }" @click="changeLanguage('zh')">{{ t('chinese') }}</button>
        </div>

        <div class="auth-login-copy">
          <p class="brand-kicker">{{ t('appName') }}</p>
          <h2>{{ t('loginTitle') }}</h2>
          <p>{{ t('loginSubtitle') }}</p>
        </div>

        <form class="auth-form" @submit.prevent="login">
          <label>
            <span>{{ t('username') }}</span>
            <input v-model="loginUsername" autocomplete="username" />
          </label>
          <label>
            <span>{{ t('password') }}</span>
            <input v-model="loginPassword" type="password" autocomplete="current-password" />
          </label>
          <button class="primary-button" :disabled="authLoading">
            {{ authLoading ? t('signingIn') : t('signIn') }}
          </button>
        </form>

        <button type="button" class="secondary-button auth-guest-button" :disabled="authLoading" @click="loginAsGuest">
          {{ authLoading ? t('signingIn') : t('guestSignIn') }}
        </button>

        <p class="auth-hint">{{ t('loginHint') }}</p>
        <p v-if="authError" class="form-error">{{ authError }}</p>
      </div>
    </section>
  </div>

  <div v-else class="app-shell">
    <header class="app-bar">
      <div class="app-bar-copy">
        <p class="brand-kicker">{{ t('appName') }}</p>
        <div class="nav-strip">
          <button type="button" class="nav-chip" :class="{ active: appView === 'workspace' }" @click="setAppView('workspace')">
            {{ t('workspace') }}
          </button>
          <button
            v-if="isAdmin"
            type="button"
            class="nav-chip"
            :class="{ active: appView === 'admin' }"
            @click="setAppView('admin')"
          >
            {{ t('admin') }}
          </button>
        </div>
      </div>

      <div class="app-bar-actions">
        <div class="language-toggle" role="group" :aria-label="t('languageLabel')">
          <span class="language-label">{{ t('languageLabel') }}</span>
          <button type="button" class="language-chip" :class="{ active: locale === 'en' }" @click="changeLanguage('en')">{{ t('english') }}</button>
          <button type="button" class="language-chip" :class="{ active: locale === 'zh' }" @click="changeLanguage('zh')">{{ t('chinese') }}</button>
        </div>
        <span class="user-badge">{{ currentUser.username }} · {{ isGuest ? t('guestBadge') : roleChip(currentUser.role) }}</span>
        <button type="button" class="ghost-button" @click="logout">{{ t('logout') }}</button>
      </div>
    </header>

    <main v-if="appView === 'workspace'" class="workspace-page">
      <aside class="sidebar">
        <section class="panel-card">
          <div class="card-heading">
            <h2>{{ t('tools') }}</h2>
            <span class="card-pill">{{ isGuest ? t('guestBadge') : toolMode === 'auto' ? t('autoModeBadge') : `${selectedTools.length} ${t('activeShort')}` }}</span>
          </div>
          <div class="tool-mode-switch">
            <button type="button" class="mode-chip" :class="{ active: toolMode === 'auto' }" :disabled="isGuest" @click="setToolMode('auto')">{{ t('autoMode') }}</button>
            <button type="button" class="mode-chip" :class="{ active: toolMode === 'manual' }" :disabled="isGuest" @click="setToolMode('manual')">{{ t('manualMode') }}</button>
          </div>
          <p v-if="isGuest" class="hint-text">{{ t('guestModeNotice') }}</p>
          <p v-else-if="toolMode === 'auto'" class="hint-text">{{ t('autoModeHint') }}</p>
          <div class="sidebar-tool-list">
            <button
              v-for="tool in tools"
              :key="tool.id"
              type="button"
              class="sidebar-tool"
              :class="{ active: toolMode === 'manual' && selectedTools.includes(tool.id), preview: toolMode === 'auto' }"
              :disabled="toolMode === 'auto' || isGuest"
              @click="toggleTool(tool.id)"
            >
              <span>{{ localizedToolName(tool) }}</span>
              <small>{{ localizedToolCategory(tool) }}</small>
            </button>
          </div>
        </section>

        <section class="panel-card">
          <div class="card-heading">
            <h2>{{ t('dataset') }}</h2>
            <span class="card-pill" :class="{ muted: !datasetMeta }">{{ datasetMeta ? t('ready') : t('waiting') }}</span>
          </div>

          <label class="attach-drop">
            <input type="file" accept=".csv,.xls,.xlsx" :disabled="isGuest" @change="handleFileChange" />
            <span class="attach-title">{{ inspectLoading ? t('attachInspecting') : t('attachTitle') }}</span>
            <span class="attach-caption">{{ isGuest ? t('guestDatasetCaption') : t('attachCaption') }}</span>
          </label>

          <div class="example-picker">
            <label>
              <span class="fact-label">{{ t('examples') }}</span>
              <select v-model="selectedExampleId" :disabled="inspectLoading || !examples.length || isGuest">
                <option value="">{{ t('chooseExample') }}</option>
                <option v-for="example in examples" :key="example.id" :value="example.id">{{ localizedExampleLabel(example) }}</option>
              </select>
            </label>
            <button type="button" class="secondary-button" :disabled="inspectLoading || !selectedExampleId || isGuest" @click="loadExampleDataset">
              {{ inspectLoading ? t('loading') : t('loadExample') }}
            </button>
          </div>

          <p v-if="selectedExample" class="hint-text">{{ localizedExampleDescription(selectedExample) }}</p>

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

        <section class="panel-card">
          <div class="card-heading">
            <h2>{{ t('promptIdeas') }}</h2>
          </div>
          <div class="link-stack">
            <button v-for="item in suggestedPrompts" :key="item" type="button" class="prompt-link" :disabled="isGuest" @click="choosePrompt(item)">
              {{ item }}
            </button>
          </div>
        </section>
      </aside>

      <section class="chat-panel">
        <div class="chat-frame">
          <div class="chat-header">
            <div>
              <p class="brand-kicker">{{ t('conversation') }}</p>
              <h2>{{ t('analysisThread') }}</h2>
            </div>
            <div class="chat-status">
              <button type="button" class="secondary-button export-button" :disabled="exportLoading || !messages.length" @click="exportChatPdf">
                {{ exportLoading ? t('exportingPdf') : t('exportPdf') }}
              </button>
              <span class="card-pill">{{ isGuest ? t('guestBadge') : toolMode === 'auto' ? t('autoModeBadge') : `${selectedTools.length} ${t('activeShort')}` }}</span>
              <span class="card-pill muted">{{ datasetMeta ? datasetMeta.fileName : t('noDatasetYet') }}</span>
            </div>
          </div>

          <div ref="chatViewport" class="chat-history">
            <article v-for="message in messages" :key="message.id" class="message-row" :class="message.role">
              <div class="message-avatar">{{ message.role === 'user' ? 'Y' : message.role === 'assistant' ? 'AI' : 'SYS' }}</div>
              <div class="message-card" :class="[message.role, message.kind]">
                <div class="message-meta">
                  <span>{{ roleLabel(message.role) }}</span>
                  <span v-if="message.toolNames?.length" class="message-tools">{{ message.toolNames.join(' · ') }}</span>
                </div>

                <p v-if="message.kind !== 'analysis' && message.kind !== 'dataset'" class="message-text">{{ message.text }}</p>

                <div v-if="message.kind === 'dataset'" class="dataset-message">
                  <p class="message-text">{{ message.text }}</p>
                  <div class="message-stats">
                    <span>{{ message.dataset.numericColumns.length }} {{ t('numeric') }}</span>
                    <span>{{ message.dataset.categoricalColumns.length }} {{ t('categorical') }}</span>
                    <span>{{ message.dataset.textColumns.length }} {{ t('text') }}</span>
                    <span>{{ message.dataset.datetimeColumns.length }} {{ t('datetime') }}</span>
                  </div>
                  <div class="table-shell preview-table-shell">
                    <table class="preview-table">
                      <thead>
                        <tr>
                          <th v-for="column in message.dataset.preview.columns" :key="`preview-${column}`">{{ column }}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="(row, rowIndex) in message.dataset.preview.rows" :key="`preview-row-${rowIndex}`">
                          <td v-for="column in message.dataset.preview.columns" :key="`preview-${rowIndex}-${column}`">{{ row[column] }}</td>
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

                  <section v-if="message.analysis?.llm" class="llm-meta-panel">
                    <span class="mini-chip">{{ llmStatusLine(message.analysis) }}</span>
                    <p v-if="llmErrorLine(message.analysis)" class="llm-warning">{{ llmErrorLine(message.analysis) }}</p>
                  </section>

                  <section v-if="message.summary && message.summary !== (message.answer || message.text)" class="summary-panel">
                    <div class="analysis-label">{{ t('analysisSummary') }}</div>
                    <pre class="summary-block secondary">{{ message.summary }}</pre>
                  </section>

                  <div class="analysis-tools">
                    <span v-for="tool in message.analysis.selectedTools" :key="`analysis-tool-${tool}`" class="mini-chip active">{{ localizedToolName(tool) }}</span>
                  </div>

                  <div class="analysis-result-stack">
                    <section v-for="result in message.analysis.results" :key="result.toolId" class="analysis-result">
                      <div class="analysis-result-head">
                        <h3>{{ localizedToolName(result.toolId) }}</h3>
                        <span class="card-pill" :class="result.status">{{ formatStatusLabel(result.status) }}</span>
                      </div>
                      <p class="result-headline">{{ result.headline }}</p>

                      <ul v-if="result.insights.length" class="insight-list">
                        <li v-for="insight in result.insights" :key="insight">{{ insight }}</li>
                      </ul>

                      <div v-if="result.charts?.length" class="chart-stack">
                        <div class="analysis-label">{{ t('visuals') }}</div>
                        <section v-for="chart in result.charts" :key="`${result.toolId}-${chart.title}`" class="chart-card">
                          <div class="chart-head">
                            <h4>{{ chart.title }}</h4>
                            <div class="chart-legend" v-if="chart.series?.length">
                              <span v-for="series in chart.series" :key="`${chart.title}-${series.name}`" class="legend-item">
                                <span class="legend-dot" :style="{ backgroundColor: series.color }" />
                                {{ series.name }}
                              </span>
                            </div>
                          </div>

                          <div v-if="chart.type === 'bar'" class="chart-bars">
                            <div v-for="point in barPoints(chart)" :key="`${chart.title}-${point.label}`" class="bar-row">
                              <span class="bar-label">{{ shortenChartLabel(point.label) }}</span>
                              <div class="bar-track">
                                <span class="bar-fill" :style="{ width: barWidth(point.value, chart), backgroundColor: chart.series[0].color }" />
                              </div>
                              <span class="bar-value">{{ formatChartValue(point.value) }}</span>
                            </div>
                          </div>

                          <div v-else-if="chart.type === 'line'" class="chart-surface">
                            <svg class="chart-svg" :viewBox="`0 0 ${CHART_FRAME.width} ${CHART_FRAME.height}`" role="img">
                              <path
                                v-for="band in chart.bands || []"
                                :key="`${chart.title}-${band.name}-band`"
                                :d="lineBandPath(chart, band)"
                                :fill="colorWithAlpha(band.color, 0.18)"
                              />
                              <line x1="42" :y1="CHART_FRAME.height - CHART_FRAME.bottom" :x2="CHART_FRAME.width - CHART_FRAME.right" :y2="CHART_FRAME.height - CHART_FRAME.bottom" class="chart-axis" />
                              <line :x1="CHART_FRAME.left" :y1="CHART_FRAME.top" :x2="CHART_FRAME.left" :y2="CHART_FRAME.height - CHART_FRAME.bottom" class="chart-axis" />
                              <g v-for="tick in lineTicks(chart)" :key="`${chart.title}-${tick}`">
                                <line :x1="CHART_FRAME.left" :y1="linePointY(chart, tick)" :x2="CHART_FRAME.width - CHART_FRAME.right" :y2="linePointY(chart, tick)" class="chart-grid-line" />
                                <text x="6" :y="linePointY(chart, tick) + 4" class="chart-axis-text">{{ formatChartValue(tick) }}</text>
                              </g>
                              <path v-for="series in chart.series" :key="`${chart.title}-${series.name}-path`" :d="linePath(chart, series)" :stroke="series.color" class="chart-line" />
                              <g v-for="series in chart.series" :key="`${chart.title}-${series.name}-points`">
                                <circle v-for="point in series.data" :key="`${chart.title}-${series.name}-${point.label}`" :cx="linePointX(chart, point.label)" :cy="linePointY(chart, point.value)" r="3.5" :fill="series.color" class="chart-point" />
                              </g>
                              <text v-for="label in lineLabels(chart)" :key="`${chart.title}-${label}`" :x="linePointX(chart, label)" :y="CHART_FRAME.height - 8" text-anchor="middle" class="chart-axis-text">
                                {{ shortenChartLabel(label) }}
                              </text>
                            </svg>
                          </div>

                          <div v-else-if="chart.type === 'scatter'" class="chart-surface">
                            <svg class="chart-svg" :viewBox="`0 0 ${CHART_FRAME.width} ${CHART_FRAME.height}`" role="img">
                              <defs v-if="chart.variant === 'cluster3d'">
                                <radialGradient
                                  v-for="(series, seriesIndex) in chart.series"
                                  :key="`${chart.title}-${series.name}-gradient-def`"
                                  :id="chartGradientId(chart, series, seriesIndex)"
                                  cx="35%"
                                  cy="30%"
                                  r="70%"
                                >
                                  <stop offset="0%" :stop-color="lightenColor(series.color, 0.62)" />
                                  <stop offset="55%" :stop-color="series.color" />
                                  <stop offset="100%" :stop-color="colorWithAlpha(series.color, 0.92)" />
                                </radialGradient>
                              </defs>
                              <line :x1="CHART_FRAME.left" :y1="CHART_FRAME.height - CHART_FRAME.bottom" :x2="CHART_FRAME.width - CHART_FRAME.right" :y2="CHART_FRAME.height - CHART_FRAME.bottom" class="chart-axis" />
                              <line :x1="CHART_FRAME.left" :y1="CHART_FRAME.top" :x2="CHART_FRAME.left" :y2="CHART_FRAME.height - CHART_FRAME.bottom" class="chart-axis" />
                              <g v-for="series in chart.series" :key="`${chart.title}-${series.name}-scatter`">
                                <ellipse
                                  v-if="chart.variant === 'cluster3d'"
                                  v-for="point in series.data"
                                  :key="`${chart.title}-${series.name}-${point.label}-shadow`"
                                  :cx="scatterPointX(chart, point.x)"
                                  :cy="scatterPointY(chart, point.y) + scatterRadius(chart, point) * 0.72"
                                  :rx="scatterRadius(chart, point) * 0.92"
                                  :ry="scatterRadius(chart, point) * 0.34"
                                  :fill="colorWithAlpha(series.color, 0.18)"
                                />
                                <circle
                                  v-for="point in series.data"
                                  :key="`${chart.title}-${series.name}-${point.label}`"
                                  :cx="scatterPointX(chart, point.x)"
                                  :cy="scatterPointY(chart, point.y)"
                                  :r="scatterRadius(chart, point)"
                                  :fill="chart.variant === 'cluster3d' ? `url(#${chartGradientId(chart, series, chart.series.indexOf(series))})` : series.color"
                                  :fill-opacity="chart.variant === 'cluster3d' ? 1 : 0.82"
                                  class="chart-point"
                                />
                                <circle
                                  v-if="chart.variant === 'cluster3d'"
                                  v-for="point in series.data"
                                  :key="`${chart.title}-${series.name}-${point.label}-highlight`"
                                  :cx="scatterPointX(chart, point.x) - scatterRadius(chart, point) * 0.24"
                                  :cy="scatterPointY(chart, point.y) - scatterRadius(chart, point) * 0.28"
                                  :r="scatterRadius(chart, point) * 0.34"
                                  fill="rgba(255,255,255,0.35)"
                                />
                              </g>
                              <text :x="CHART_FRAME.width / 2" :y="CHART_FRAME.height - 6" text-anchor="middle" class="chart-axis-text">{{ chart.xLabel }}</text>
                              <text :x="16" :y="CHART_FRAME.height / 2" class="chart-axis-text" transform="rotate(-90 16 115)">{{ chart.yLabel }}</text>
                            </svg>
                          </div>
                        </section>
                      </div>

                      <div v-if="result.warnings.length" class="result-notes">
                        <p v-for="warning in result.warnings" :key="warning">{{ warning }}</p>
                      </div>

                      <div v-for="table in result.tables" :key="`${result.toolId}-${table.title}`" class="result-table">
                        <h4>{{ table.title }}</h4>
                        <div class="table-shell" v-if="table.columns.length">
                          <table>
                            <thead>
                              <tr>
                                <th v-for="column in table.columns" :key="`${table.title}-${column}`">{{ column }}</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr v-for="(row, rowIndex) in table.rows" :key="`${table.title}-${rowIndex}`">
                                <td v-for="column in table.columns" :key="`${table.title}-${rowIndex}-${column}`">{{ row[column] }}</td>
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
            <p v-if="isGuest" class="hint-text">{{ t('guestModeNotice') }}</p>
            <div class="composer-topline">
              <label class="attach-inline">
                <input type="file" accept=".csv,.xls,.xlsx" :disabled="isGuest" @change="handleFileChange" />
                <span>{{ inspectLoading ? t('attachInspecting') : t('attachFile') }}</span>
              </label>
              <button type="button" class="attach-inline attach-inline-secondary" :disabled="inspectLoading || !selectedExampleId || isGuest" @click="loadExampleDataset">
                <span>{{ inspectLoading ? t('loadingExample') : t('useSelectedExample') }}</span>
              </button>
              <span class="composer-file">{{ datasetMeta ? datasetMeta.fileName : t('noDatasetAttached') }}</span>
            </div>

            <textarea v-model="prompt" class="composer-input" rows="3" :disabled="isGuest" :placeholder="isGuest ? t('guestPromptPlaceholder') : t('promptPlaceholder')" />

            <div class="composer-footer">
              <div class="selected-tool-strip">
                <span v-if="toolMode === 'auto'" class="mini-chip active">{{ t('autoModeBadge') }}</span>
                <span v-for="tool in toolMode === 'auto' ? [] : selectedToolDetails" :key="`selected-${tool.id}`" class="mini-chip active">{{ localizedToolName(tool) }}</span>
              </div>
              <button class="send-button" :disabled="!canAnalyze">{{ analyzeLoading ? t('analyzing') : t('send') }}</button>
            </div>
          </form>
        </div>
      </section>

      <aside class="sidebar">
        <section class="panel-card">
          <div class="card-heading">
            <h2>{{ t('myAccount') }}</h2>
            <span class="card-pill">{{ isGuest ? t('guestBadge') : roleChip(currentUser.role) }}</span>
          </div>
          <p class="hint-text">{{ currentUser.username }}</p>
          <p class="hint-text">{{ isGuest ? t('guestPasswordHint') : t('selfPasswordHint') }}</p>
          <div class="form-grid compact">
            <label>
              <span>{{ t('currentPassword') }}</span>
              <input v-model="accountCurrentPassword" type="password" :disabled="isGuest" />
            </label>
            <label>
              <span>{{ t('newPassword') }}</span>
              <input v-model="accountNewPassword" type="password" :disabled="isGuest" />
            </label>
          </div>
          <button type="button" class="primary-button" :disabled="accountLoading || isGuest" @click="changeOwnPassword">
            {{ accountLoading ? t('saving') : t('savePassword') }}
          </button>
          <p v-if="accountNotice" class="form-success">{{ accountNotice }}</p>
          <p v-if="accountError" class="form-error">{{ accountError }}</p>
        </section>

        <section class="panel-card">
          <div class="card-heading">
            <h2>{{ t('analysisSettings') }}</h2>
          </div>
          <p v-if="!datasetMeta" class="hint-text">{{ t('settingsHint') }}</p>
          <div class="settings-grid" :class="{ disabled: !datasetMeta || isGuest }">
            <label>
              <span>{{ t('targetColumn') }}</span>
              <select v-model="targetColumn" :disabled="!datasetMeta || isGuest">
                <option value="">{{ t('optional') }}</option>
                <option v-for="column in datasetColumns" :key="`target-${column.name}`" :value="column.name">{{ column.name }} ({{ formatLabel(column.kind) }})</option>
              </select>
            </label>
            <label>
              <span>{{ t('timeColumn') }}</span>
              <select v-model="timeColumn" :disabled="!datasetMeta || isGuest">
                <option value="">{{ t('optional') }}</option>
                <option v-for="column in datasetColumns" :key="`time-${column.name}`" :value="column.name">{{ column.name }} ({{ formatLabel(column.kind) }})</option>
              </select>
            </label>
            <label>
              <span>{{ t('valueColumn') }}</span>
              <select v-model="valueColumn" :disabled="!datasetMeta || isGuest">
                <option value="">{{ t('optional') }}</option>
                <option v-for="column in (datasetMeta ? datasetMeta.numericColumns : [])" :key="`value-${column}`" :value="column">{{ column }}</option>
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
                  :disabled="isGuest"
                  @click="textColumns = textColumns.includes(column) ? textColumns.filter((item) => item !== column) : [...textColumns, column]"
                >
                  {{ column }}
                </button>
              </div>
              <p v-else class="hint-text">{{ datasetMeta ? t('noTextColumns') : t('textColumnsAfterUpload') }}</p>
            </div>
          </div>
        </section>

        <section class="panel-card">
          <div class="card-heading">
            <h2>{{ t('usefulLinks') }}</h2>
          </div>
          <div class="link-stack">
            <a v-for="link in usefulLinks" :key="link.label" class="useful-link" :href="link.href" target="_blank" rel="noreferrer">
              <strong>{{ link.label }}</strong>
              <span>{{ link.caption }}</span>
            </a>
          </div>
        </section>
      </aside>
    </main>

    <main v-else class="admin-page">
      <section class="admin-hero">
        <div>
          <p class="brand-kicker">{{ t('adminOnly') }}</p>
          <h1>{{ t('adminTitle') }}</h1>
          <p>{{ t('adminSubtitle') }}</p>
        </div>
        <button type="button" class="secondary-button" @click="loadAdminData">{{ t('refreshData') }}</button>
      </section>

      <p v-if="adminNotice" class="form-success">{{ adminNotice }}</p>
      <p v-if="adminError" class="form-error">{{ adminError }}</p>

      <div class="admin-grid">
        <section class="panel-card">
          <div class="card-heading">
            <h2>{{ t('llmConfig') }}</h2>
            <span class="card-pill">{{ t('adminOnly') }}</span>
          </div>
          <div class="form-grid">
            <label>
              <span>{{ t('provider') }}</span>
              <select v-model="llmForm.provider">
                <option value="auto">{{ t('providerAuto') }}</option>
                <option value="qwen">{{ t('providerQwen') }}</option>
                <option value="openai_compatible">{{ t('providerOpenAI') }}</option>
              </select>
            </label>
            <label>
              <span>{{ t('qwenModel') }}</span>
              <input v-model="llmForm.qwenModel" />
            </label>
            <label class="full-span">
              <span>{{ t('qwenKey') }}</span>
              <input v-model="llmForm.qwenApiKey" type="password" />
            </label>
            <label class="full-span">
              <span>{{ t('openaiBaseUrl') }}</span>
              <input v-model="llmForm.openaiBaseUrl" />
            </label>
            <label>
              <span>{{ t('openaiModel') }}</span>
              <input v-model="llmForm.openaiModel" />
            </label>
            <label class="full-span">
              <span>{{ t('openaiKey') }}</span>
              <input v-model="llmForm.openaiApiKey" type="password" />
            </label>
          </div>
          <button type="button" class="primary-button" :disabled="adminLoading" @click="saveLlmConfig">
            {{ adminLoading ? t('saving') : t('saveConfig') }}
          </button>
        </section>

        <section class="panel-card">
          <div class="card-heading">
            <h2>{{ t('createUser') }}</h2>
          </div>
          <p class="hint-text">{{ t('createUserHint') }}</p>
          <div class="form-grid">
            <label>
              <span>{{ t('username') }}</span>
              <input v-model="newUserUsername" />
            </label>
            <label>
              <span>{{ t('password') }}</span>
              <input v-model="newUserPassword" type="password" />
            </label>
          </div>
          <button type="button" class="primary-button" :disabled="adminLoading" @click="createManagedUser">
            {{ adminLoading ? t('saving') : t('create') }}
          </button>
        </section>

        <section class="panel-card admin-users-card">
          <div class="card-heading">
            <h2>{{ t('userList') }}</h2>
          </div>
          <div class="table-shell">
            <table>
              <thead>
                <tr>
                  <th>{{ t('username') }}</th>
                  <th>{{ t('role') }}</th>
                  <th>{{ t('createdAt') }}</th>
                  <th>{{ t('updatedAt') }}</th>
                  <th>{{ t('resetPassword') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="user in adminUsers" :key="user.id">
                  <td>{{ user.username }}</td>
                  <td>{{ roleChip(user.role) }}</td>
                  <td>{{ user.createdAt || '-' }}</td>
                  <td>{{ user.updatedAt || '-' }}</td>
                  <td>
                    <div class="inline-reset">
                      <input v-model="passwordDrafts[user.id]" type="password" :placeholder="t('newPassword')" />
                      <button type="button" class="secondary-button inline-button" @click="resetManagedUserPassword(user.id)">
                        {{ t('reset') }}
                      </button>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="panel-card">
          <div class="card-heading">
            <h2>{{ t('myAccount') }}</h2>
          </div>
          <p class="hint-text">{{ currentUser.username }}</p>
          <div class="form-grid">
            <label>
              <span>{{ t('currentPassword') }}</span>
              <input v-model="accountCurrentPassword" type="password" />
            </label>
            <label>
              <span>{{ t('newPassword') }}</span>
              <input v-model="accountNewPassword" type="password" />
            </label>
          </div>
          <button type="button" class="primary-button" :disabled="accountLoading" @click="changeOwnPassword">
            {{ accountLoading ? t('saving') : t('savePassword') }}
          </button>
        </section>
      </div>
    </main>
  </div>
</template>

<style scoped>
.boot-shell,
.auth-shell,
.app-shell {
  min-height: 100dvh;
  background:
    radial-gradient(circle at top left, rgba(44, 143, 107, 0.08), transparent 24%),
    radial-gradient(circle at top right, rgba(219, 109, 69, 0.08), transparent 22%),
    linear-gradient(180deg, #f6f3ee 0%, #efe9df 100%);
  color: var(--ink);
}

.boot-shell,
.auth-shell {
  display: grid;
  place-items: center;
  padding: 1.5rem;
}

.boot-card,
.auth-card,
.panel-card,
.chat-frame,
.message-card,
.analysis-result,
.composer,
.attach-drop,
.admin-hero {
  border: 1px solid var(--line);
  background: rgba(255, 253, 250, 0.96);
  box-shadow: var(--shadow);
  backdrop-filter: blur(12px);
  border-radius: 18px;
}

.auth-card {
  width: min(1120px, 100%);
  padding: 1.35rem;
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.85fr);
  gap: 1rem;
}

.boot-card {
  padding: 1rem 1.25rem;
  font-weight: 700;
}

.auth-brand h1,
.auth-login-copy h2,
.card-heading h2,
.chat-header h2,
.admin-hero h1,
.analysis-result h3,
.result-table h4 {
  margin: 0;
}

.auth-brand p,
.auth-login-copy p,
.admin-hero p,
.hint-text,
.attach-caption,
.composer-file,
.result-headline,
.useful-link span,
.prompt-link {
  color: var(--muted);
}

.brand-kicker,
.fact-label {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.68rem;
  font-weight: 800;
  color: var(--accent-strong);
}

.auth-showcase,
.auth-pane {
  border-radius: 16px;
  min-width: 0;
}

.auth-showcase {
  padding: 1.2rem;
  background:
    radial-gradient(circle at top right, rgba(60, 120, 255, 0.1), transparent 30%),
    radial-gradient(circle at bottom left, rgba(43, 133, 103, 0.12), transparent 34%),
    linear-gradient(180deg, rgba(247, 244, 238, 0.96), rgba(242, 236, 227, 0.94));
  display: grid;
  gap: 1rem;
}

.auth-pane {
  padding: 1rem;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(16, 35, 28, 0.08);
  display: grid;
  align-content: start;
  gap: 1rem;
}

.auth-brand h1 {
  font-size: clamp(1.72rem, 2.7vw, 2.45rem);
  line-height: 1.06;
  max-width: 12ch;
}

.auth-showcase .brand-kicker {
  font-size: clamp(1.05rem, 1.5vw, 1.4rem);
  letter-spacing: 0.14em;
  color: #435851;
}

.auth-brand p,
.auth-login-copy p {
  margin: 0;
  line-height: 1.6;
}

.auth-login-copy {
  display: grid;
  gap: 0.3rem;
}

.auth-login-copy h2 {
  font-size: 1.35rem;
}

.auth-pills,
.feature-grid {
  display: grid;
  gap: 0.65rem;
}

.auth-pills {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
}

.hero-pill,
.feature-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 0.28rem 0.7rem;
  font-size: 0.74rem;
  font-weight: 800;
}

.hero-pill {
  background: rgba(43, 133, 103, 0.12);
  color: #1d5f49;
}

.hero-pill-warm {
  background: rgba(219, 109, 69, 0.14);
  color: #9a4f32;
}

.feature-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.feature-card,
.sponsor-card {
  border-radius: 16px;
  border: 1px solid rgba(16, 35, 28, 0.08);
}

.feature-card {
  padding: 0.9rem;
  background: rgba(255, 255, 255, 0.7);
  display: grid;
  gap: 0.38rem;
}

.feature-card h3,
.sponsor-card h3 {
  margin: 0;
  font-size: 1rem;
}

.feature-card p,
.sponsor-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
}

.feature-badge {
  width: fit-content;
  background: rgba(20, 53, 43, 0.07);
  color: #36534a;
}

.sponsor-card {
  padding: 1rem;
  background: linear-gradient(135deg, rgba(23, 31, 56, 0.95), rgba(37, 84, 66, 0.95));
  color: white;
  text-decoration: none;
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1rem;
}

.sponsor-card p {
  color: rgba(255, 255, 255, 0.78);
}

.sponsor-kicker {
  margin: 0 0 0.25rem;
  font-size: 0.78rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.72);
}

.sponsor-link {
  white-space: nowrap;
  font-weight: 800;
  color: #f0d290;
}

.auth-form,
.form-grid,
.settings-grid {
  display: grid;
  gap: 0.8rem;
}

.auth-guest-button {
  width: 100%;
}

.form-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.form-grid.compact {
  grid-template-columns: 1fr;
}

.full-span {
  grid-column: 1 / -1;
}

label,
.text-picker {
  display: grid;
  gap: 0.35rem;
}

label span,
.text-picker > span,
.language-label {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--muted);
}

input,
select,
textarea,
button {
  font: inherit;
}

input,
select,
textarea {
  width: 100%;
  border: 1px solid rgba(16, 35, 28, 0.12);
  border-radius: 12px;
  padding: 0.6rem 0.72rem;
  background: rgba(255, 255, 255, 0.9);
  color: var(--ink);
}

textarea {
  resize: vertical;
  min-height: 92px;
}

button {
  cursor: pointer;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.primary-button,
.secondary-button,
.ghost-button,
.send-button,
.mode-chip,
.language-chip,
.nav-chip,
.sidebar-tool,
.attach-inline,
.prompt-link {
  border-radius: 12px;
  border: 1px solid rgba(16, 35, 28, 0.12);
  transition: 0.18s ease;
}

.primary-button,
.send-button {
  background: linear-gradient(135deg, #1d8f68, #2f6dd8);
  color: white;
  padding: 0.72rem 1rem;
  font-weight: 700;
}

.secondary-button,
.ghost-button,
.attach-inline,
.prompt-link,
.mode-chip,
.language-chip,
.nav-chip,
.sidebar-tool {
  background: rgba(255, 255, 255, 0.88);
  color: var(--ink);
}

.ghost-button,
.language-chip,
.nav-chip,
.mode-chip {
  padding: 0.42rem 0.75rem;
}

.secondary-button,
.attach-inline,
.prompt-link,
.sidebar-tool {
  padding: 0.58rem 0.72rem;
}

.export-button {
  white-space: nowrap;
}

.language-toggle {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.language-chip.active,
.nav-chip.active,
.mode-chip.active,
.sidebar-tool.active,
.mini-chip.active {
  background: rgba(43, 133, 103, 0.12);
  border-color: rgba(43, 133, 103, 0.26);
  color: #1c5e47;
}

.auth-hint,
.form-error,
.form-success {
  margin: 0;
  font-size: 0.9rem;
}

.form-error {
  color: #b7492c;
}

.form-success {
  color: #1d7b5b;
}

.app-shell {
  padding: 0.6rem;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  gap: 0.6rem;
}

.app-bar {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255, 253, 250, 0.94);
  box-shadow: var(--shadow);
  padding: 0.45rem 0.8rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
}

.app-bar-copy,
.app-bar-actions,
.nav-strip {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.user-badge,
.card-pill,
.mini-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  border: 1px solid rgba(16, 35, 28, 0.12);
  border-radius: 999px;
  padding: 0.22rem 0.62rem;
  font-size: 0.72rem;
  font-weight: 700;
  background: rgba(255, 255, 255, 0.72);
}

.card-pill.muted {
  color: var(--muted);
}

.workspace-page {
  max-width: 1600px;
  margin: 0 auto;
  width: 100%;
  min-height: 0;
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr) 248px;
  gap: 0.6rem;
}

.sidebar {
  min-height: 0;
  display: grid;
  gap: 0.6rem;
  align-content: start;
  overflow-y: auto;
}

.panel-card {
  padding: 0.85rem;
  display: grid;
  gap: 0.7rem;
}

.card-heading,
.chat-header,
.analysis-result-head,
.chart-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.6rem;
}

.sidebar-tool-list,
.link-stack,
.dataset-facts,
.message-stats,
.analysis-result-stack,
.chart-stack {
  display: grid;
  gap: 0.55rem;
}

.sidebar-tool {
  display: grid;
  gap: 0.12rem;
  text-align: left;
}

.sidebar-tool small {
  color: var(--muted);
}

.sidebar-tool.preview {
  opacity: 0.7;
}

.attach-drop input,
.attach-inline input {
  display: none;
}

.attach-drop {
  padding: 0.85rem;
  display: grid;
  gap: 0.35rem;
  cursor: pointer;
}

.attach-title {
  font-weight: 700;
}

.example-picker {
  display: grid;
  gap: 0.55rem;
}

.dataset-facts {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.dataset-fact-file {
  grid-column: 1 / -1;
}

.dataset-fact {
  min-width: 0;
  padding: 0.55rem 0.65rem;
  border-radius: 12px;
  background: rgba(243, 239, 232, 0.88);
  display: grid;
  gap: 0.18rem;
}

.chat-panel {
  min-width: 0;
}

.chat-frame {
  height: 100%;
  min-height: 0;
  padding: 0.85rem;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 0.7rem;
}

.chat-history {
  min-height: 0;
  overflow-y: auto;
  padding-right: 0.15rem;
  display: grid;
  gap: 0.7rem;
  align-content: start;
  grid-auto-rows: max-content;
}

.message-row {
  display: grid;
  grid-template-columns: 42px minmax(0, 1fr);
  gap: 0.6rem;
  align-items: start;
}

.message-avatar {
  width: 42px;
  height: 42px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, rgba(44, 143, 107, 0.16), rgba(74, 108, 247, 0.16));
  color: #245745;
  font-weight: 800;
}

.message-card {
  padding: 0.8rem;
  display: grid;
  gap: 0.6rem;
}

.message-meta {
  display: flex;
  justify-content: space-between;
  gap: 0.6rem;
  color: var(--muted);
  font-size: 0.8rem;
}

.message-text,
.summary-block {
  margin: 0;
  white-space: pre-wrap;
  line-height: 1.55;
}

.summary-block {
  font-family: inherit;
  background: rgba(244, 240, 233, 0.72);
  padding: 0.8rem;
  border-radius: 12px;
}

.summary-block.secondary {
  background: rgba(240, 245, 244, 0.88);
}

.table-shell {
  overflow: auto;
  border: 1px solid rgba(16, 35, 28, 0.08);
  border-radius: 12px;
}

.preview-table-shell {
  max-height: 240px;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
}

th,
td {
  padding: 0.55rem 0.65rem;
  border-bottom: 1px solid rgba(16, 35, 28, 0.08);
  text-align: left;
  vertical-align: top;
}

th {
  background: rgba(246, 243, 238, 0.92);
}

.preview-table {
  table-layout: fixed;
}

.preview-table th,
.preview-table td {
  padding: 0.38rem 0.5rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.2;
  max-width: 180px;
}

.analysis-tools,
.selected-tool-strip,
.mini-chip-wrap,
.chart-legend,
.tool-mode-switch,
.composer-topline,
.composer-footer,
.chat-status,
.llm-meta-panel {
  display: flex;
  gap: 0.45rem;
  flex-wrap: wrap;
  align-items: center;
}

.llm-warning {
  margin: 0;
  font-size: 0.78rem;
  color: #9f5537;
}

.analysis-label {
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent-strong);
}

.insight-list {
  margin: 0;
  padding-left: 1.2rem;
}

.chart-card,
.result-table {
  display: grid;
  gap: 0.5rem;
  padding: 0.7rem;
  border-radius: 14px;
  background: rgba(248, 246, 241, 0.82);
}

.chart-legend {
  justify-content: flex-end;
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.74rem;
  color: var(--muted);
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
}

.chart-bars {
  display: grid;
  gap: 0.38rem;
}

.bar-row {
  display: grid;
  grid-template-columns: 88px minmax(0, 1fr) 54px;
  gap: 0.5rem;
  align-items: center;
}

.bar-label,
.bar-value {
  font-size: 0.78rem;
}

.bar-track {
  height: 10px;
  border-radius: 999px;
  background: rgba(16, 35, 28, 0.08);
  overflow: hidden;
}

.bar-fill {
  display: block;
  height: 100%;
  border-radius: 999px;
}

.chart-surface {
  overflow-x: auto;
}

.chart-svg {
  width: 100%;
  min-width: 460px;
  height: auto;
}

.chart-axis,
.chart-grid-line {
  stroke: rgba(16, 35, 28, 0.12);
  stroke-width: 1;
}

.chart-grid-line {
  stroke-dasharray: 3 4;
}

.chart-axis-text {
  fill: rgba(16, 35, 28, 0.58);
  font-size: 11px;
}

.chart-line {
  fill: none;
  stroke-width: 2.5;
}

.chart-point {
  stroke: rgba(255, 255, 255, 0.72);
  stroke-width: 1.4;
}

.result-notes {
  display: grid;
  gap: 0.32rem;
  color: #9f5537;
  font-size: 0.84rem;
}

.composer {
  padding: 0.75rem;
  display: grid;
  gap: 0.65rem;
}

.attach-inline {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

.attach-inline-secondary {
  background: rgba(244, 240, 233, 0.92);
}

.composer-file {
  font-size: 0.82rem;
}

.send-button {
  min-width: 160px;
}

.settings-grid.disabled {
  opacity: 0.65;
}

.text-picker .mini-chip {
  background: rgba(255, 255, 255, 0.92);
}

.useful-link {
  display: grid;
  gap: 0.18rem;
  text-decoration: none;
  color: inherit;
  padding: 0.65rem 0.72rem;
  border-radius: 12px;
  background: rgba(247, 243, 236, 0.85);
}

.admin-page {
  max-width: 1440px;
  margin: 0 auto;
  width: 100%;
  display: grid;
  gap: 0.7rem;
  align-content: start;
  overflow: auto;
}

.admin-hero {
  padding: 0.95rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
}

.admin-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.7rem;
}

.admin-users-card {
  grid-column: 1 / -1;
}

.inline-reset {
  display: flex;
  gap: 0.45rem;
  align-items: center;
}

.inline-button {
  white-space: nowrap;
}

@media (max-width: 1180px) {
  .auth-card {
    grid-template-columns: 1fr;
  }

  .feature-grid {
    grid-template-columns: 1fr;
  }

  .workspace-page,
  .admin-grid {
    grid-template-columns: 1fr;
  }

  .sidebar {
    overflow: visible;
  }
}

@media (max-width: 760px) {
  .app-shell {
    padding: 0.45rem;
  }

  .app-bar,
  .admin-hero,
  .chat-header,
  .analysis-result-head,
  .chart-head {
    align-items: flex-start;
    flex-direction: column;
  }

  .message-row {
    grid-template-columns: 1fr;
  }

  .form-grid,
  .dataset-facts {
    grid-template-columns: 1fr;
  }

  .sponsor-card {
    align-items: flex-start;
    flex-direction: column;
  }
}
</style>
