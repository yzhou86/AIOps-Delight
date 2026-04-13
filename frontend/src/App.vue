<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'

const tools = ref([])
const selectedTools = ref(['data_profile', 'correlation_explorer', 'anomaly_detector'])
const datasetFile = ref(null)
const datasetMeta = ref(null)
const prompt = ref('Find the strongest patterns in this dataset, call out anomalies, and recommend what I should investigate next.')
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
    text: 'Upload a CSV or Excel file, choose your data science tools from the banner, and ask the agent to analyze it. I will keep the full conversation history here.'
  }
])

const usefulLinks = [
  { label: 'Tool Catalog API', href: '/api/tools', caption: 'Inspect the available tool catalog as JSON.' },
  { label: 'Health Check', href: '/api/health', caption: 'Confirm the unified Flask app is healthy.' },
  { label: 'Pandas Docs', href: 'https://pandas.pydata.org/docs/', caption: 'Useful for dataframe-oriented analysis.' },
  { label: 'Scikit-learn', href: 'https://scikit-learn.org/stable/', caption: 'Reference for clustering, anomaly detection, and classification.' },
  { label: 'Flask Docs', href: 'https://flask.palletsprojects.com/', caption: 'Backend reference for the unified app runtime.' }
]

const suggestedPrompts = [
  'Summarize the most important trends, outliers, and next actions.',
  'Focus on anomalies and tell me which rows or periods deserve attention first.',
  'Group this dataset into meaningful segments and explain each segment.',
  'Train a baseline classifier and tell me which features matter most.'
]

const selectedToolDetails = computed(() =>
  tools.value.filter((tool) => selectedTools.value.includes(tool.id))
)

const selectedToolNames = computed(() => selectedToolDetails.value.map((tool) => tool.name))

const datasetColumns = computed(() => datasetMeta.value?.columns || [])

const canAnalyze = computed(() =>
  Boolean(datasetMeta.value?.datasetId) &&
  selectedTools.value.length > 0 &&
  prompt.value.trim() &&
  !analyzeLoading.value
)

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

function formatLabel(value) {
  return value.replace(/_/g, ' ')
}

function roleLabel(role) {
  if (role === 'user') return 'You'
  if (role === 'assistant') return 'Agent'
  return 'System'
}

function datasetHeadline(dataset) {
  return `${dataset.fileName} inspected with ${dataset.rowCount} rows and ${dataset.columnCount} columns.`
}

function choosePrompt(text) {
  prompt.value = text
}

async function fetchTools() {
  const response = await fetch('/api/tools')
  const data = await response.json()
  tools.value = data.tools || []
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
  if (file) {
    pushMessage('user', 'upload', {
      text: `Attached dataset: ${file.name}`
    })
    await inspectDataset()
  }
}

async function inspectDataset() {
  if (!datasetFile.value) return

  inspectLoading.value = true
  const loadingId = pushMessage('assistant', 'loading', {
    text: `Inspecting ${datasetFile.value.name} and inferring schema...`
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
      throw new Error(data.error || 'Dataset inspection failed.')
    }

    datasetMeta.value = data
    targetColumn.value = data.categoricalColumns?.[0] || ''
    timeColumn.value = data.datetimeColumns?.[0] || ''
    valueColumn.value = data.numericColumns?.[0] || ''
    textColumns.value = data.textColumns ? data.textColumns.slice(0, 1) : []

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
      text: 'Upload and inspect a dataset before sending an analysis request.'
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
    text: 'Running the selected tools and composing the analysis...'
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
        textColumns: textColumns.value
      })
    })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.error || 'Analysis failed.')
    }

    replaceMessage(loadingId, {
      role: 'assistant',
      kind: 'analysis',
      text: data.answer || data.summary,
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
  scrollToBottom()
})
</script>

<template>
  <div class="copilot-shell">
    <header class="tool-banner">
      <div class="banner-copy">
        <p class="banner-kicker">AIOps Delight Copilot</p>
        <h1>Chat with your dataset</h1>
        <p>
          Choose tools, attach a file, and talk to the analysis agent in one running thread.
        </p>
      </div>
    </header>

    <main class="workspace">
      <aside class="sidebar sidebar-left">
        <section class="sidebar-card">
          <div class="card-heading">
            <h2>Tools</h2>
            <span class="card-pill">{{ selectedTools.length }} active</span>
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
              <span>{{ tool.name }}</span>
              <small>{{ tool.category }}</small>
            </button>
          </div>
        </section>

        <section class="sidebar-card dataset-card">
          <div class="card-heading">
            <h2>Dataset</h2>
            <span class="card-pill" :class="{ muted: !datasetMeta }">
              {{ datasetMeta ? 'Ready' : 'Waiting' }}
            </span>
          </div>

          <label class="attach-drop">
            <input type="file" accept=".csv,.xls,.xlsx" @change="handleFileChange" />
            <span class="attach-title">
              {{ inspectLoading ? 'Inspecting file...' : 'Attach CSV / XLS / XLSX' }}
            </span>
            <span class="attach-caption">Your file is profiled first, then the agent can analyze it.</span>
          </label>

          <div v-if="datasetMeta" class="dataset-facts">
            <div>
              <span class="fact-label">File</span>
              <strong>{{ datasetMeta.fileName }}</strong>
            </div>
            <div>
              <span class="fact-label">Rows</span>
              <strong>{{ datasetMeta.rowCount }}</strong>
            </div>
            <div>
              <span class="fact-label">Columns</span>
              <strong>{{ datasetMeta.columnCount }}</strong>
            </div>
            <div>
              <span class="fact-label">Text columns</span>
              <strong>{{ datasetMeta.textColumns.length }}</strong>
            </div>
          </div>
        </section>

        <section class="sidebar-card" v-if="datasetMeta">
          <div class="card-heading">
            <h2>Analysis Settings</h2>
          </div>

          <div class="settings-grid">
            <label>
              <span>Target column</span>
              <select v-model="targetColumn">
                <option value="">Optional</option>
                <option v-for="column in datasetColumns" :key="`target-${column.name}`" :value="column.name">
                  {{ column.name }} ({{ formatLabel(column.kind) }})
                </option>
              </select>
            </label>

            <label>
              <span>Time column</span>
              <select v-model="timeColumn">
                <option value="">Optional</option>
                <option v-for="column in datasetColumns" :key="`time-${column.name}`" :value="column.name">
                  {{ column.name }} ({{ formatLabel(column.kind) }})
                </option>
              </select>
            </label>

            <label>
              <span>Value column</span>
              <select v-model="valueColumn">
                <option value="">Optional</option>
                <option v-for="column in datasetMeta.numericColumns" :key="`value-${column}`" :value="column">
                  {{ column }}
                </option>
              </select>
            </label>

            <div class="text-picker">
              <span>Text columns</span>
              <div class="mini-chip-wrap">
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
            </div>
          </div>
        </section>

        <section class="sidebar-card">
          <div class="card-heading">
            <h2>Prompt Ideas</h2>
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
              <p class="chat-kicker">Conversation</p>
              <h2>Analysis thread</h2>
            </div>
            <div class="chat-status">
              <span class="card-pill">{{ selectedTools.length }} tools active</span>
              <span class="card-pill muted">{{ datasetMeta ? datasetMeta.fileName : 'No dataset yet' }}</span>
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
                    <span>{{ message.dataset.numericColumns.length }} numeric</span>
                    <span>{{ message.dataset.categoricalColumns.length }} categorical</span>
                    <span>{{ message.dataset.textColumns.length }} text</span>
                    <span>{{ message.dataset.datetimeColumns.length }} datetime</span>
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
                  <pre class="summary-block">{{ message.text }}</pre>
                  <div class="analysis-tools">
                    <span
                      v-for="tool in message.analysis.selectedTools"
                      :key="`analysis-tool-${tool}`"
                      class="mini-chip active"
                    >
                      {{ formatLabel(tool) }}
                    </span>
                  </div>

                  <div class="analysis-result-stack">
                    <section
                      v-for="result in message.analysis.results"
                      :key="result.toolId"
                      class="analysis-result"
                    >
                      <div class="analysis-result-head">
                        <h3>{{ result.toolName }}</h3>
                        <span class="card-pill" :class="result.status">{{ result.status }}</span>
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
                <span>{{ inspectLoading ? 'Inspecting...' : 'Attach file' }}</span>
              </label>
              <span class="composer-file">{{ datasetMeta ? datasetMeta.fileName : 'No dataset attached' }}</span>
            </div>

            <textarea
              v-model="prompt"
              class="composer-input"
              rows="3"
              placeholder="Ask the agent to analyze the dataset, compare segments, forecast a metric, or explain anomalies."
            />

            <div class="composer-footer">
              <div class="selected-tool-strip">
                <span
                  v-for="tool in selectedToolDetails"
                  :key="`selected-${tool.id}`"
                  class="mini-chip active"
                >
                  {{ tool.name }}
                </span>
              </div>

              <button class="send-button" :disabled="!canAnalyze">
                {{ analyzeLoading ? 'Analyzing...' : 'Send to agent' }}
              </button>
            </div>
          </form>
        </div>
      </section>

      <aside class="sidebar sidebar-right">
        <section class="sidebar-card">
          <div class="card-heading">
            <h2>Useful Links</h2>
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
  width: 100%;
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
  max-width: 1440px;
  margin: 0 auto;
  width: 100%;
  display: grid;
  grid-template-columns: 248px minmax(0, 1fr) 216px;
  gap: 0.75rem;
  align-items: stretch;
  min-height: 0;
  height: 100%;
  max-height: 100%;
}

.sidebar {
  display: grid;
  gap: 0.75rem;
  height: 100%;
  overflow: auto;
  align-content: start;
  min-height: 0;
}

.sidebar-right {
  position: sticky;
  top: 0.75rem;
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

.dataset-facts div {
  border-radius: 12px;
  padding: 0.65rem;
  background: rgba(255, 255, 255, 0.58);
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

.settings-grid span,
.text-picker span {
  font-weight: 600;
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
}

.useful-link {
  display: grid;
  gap: 0.2rem;
  color: var(--ink);
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
