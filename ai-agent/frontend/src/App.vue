<script setup>
import { ref } from 'vue'

const messages = ref([
  { role: 'assistant', content: '你好！我是AI助手，有什么可以帮助你的吗？' }
])
const inputMessage = ref('')
const isLoading = ref(false)

async function sendMessage() {
  if (!inputMessage.value.trim() || isLoading.value) return

  const userMessage = inputMessage.value.trim()
  messages.value.push({ role: 'user', content: userMessage })
  inputMessage.value = ''
  isLoading.value = true

  try {
    const response = await fetch('http://localhost:5000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message: userMessage })
    })

    if (response.ok) {
      const data = await response.json()
      messages.value.push({ role: 'assistant', content: data.response })
    } else {
      const errorData = await response.json()
      messages.value.push({ role: 'assistant', content: `错误：${errorData.error}` })
    }
  } catch (error) {
    messages.value.push({ role: 'assistant', content: `请求失败：${error.message}` })
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <div class="chat-container">
    <div class="chat-header">
      <h1>AI助手</h1>
    </div>
    <div class="chat-messages">
      <div 
        v-for="(msg, index) in messages" 
        :key="index" 
        :class="['message', msg.role]"
      >
        <div class="message-content">{{ msg.content }}</div>
      </div>
      <div v-if="isLoading" class="message assistant">
        <div class="message-content">思考中...</div>
      </div>
    </div>
    <div class="chat-input-area">
      <input
        v-model="inputMessage"
        @keyup.enter="sendMessage"
        placeholder="输入您的问题..."
        class="chat-input"
      />
      <button @click="sendMessage" :disabled="isLoading" class="send-button">
        发送
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-container {
  width: 100%;
  max-width: 800px;
  height: 100vh;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  background-color: #f5f5f5;
}

.chat-header {
  background-color: #42b883;
  color: white;
  padding: 1rem;
  text-align: center;
}

.chat-header h1 {
  margin: 0;
  font-size: 1.5rem;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  display: flex;
  max-width: 70%;
}

.message.user {
  margin-left: auto;
  justify-content: flex-end;
}

.message.assistant {
  margin-right: auto;
  justify-content: flex-start;
}

.message-content {
  padding: 0.8rem 1.2rem;
  border-radius: 1rem;
  line-height: 1.4;
}

.message.user .message-content {
  background-color: #42b883;
  color: white;
  border-bottom-right-radius: 0.2rem;
}

.message.assistant .message-content {
  background-color: white;
  color: #333;
  border-bottom-left-radius: 0.2rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.chat-input-area {
  padding: 1rem;
  background-color: white;
  display: flex;
  gap: 0.5rem;
  border-top: 1px solid #e0e0e0;
}

.chat-input {
  flex: 1;
  padding: 0.8rem 1rem;
  border: 1px solid #ddd;
  border-radius: 2rem;
  font-size: 1rem;
  outline: none;
}

.chat-input:focus {
  border-color: #42b883;
}

.send-button {
  padding: 0.8rem 1.5rem;
  background-color: #42b883;
  color: white;
  border: none;
  border-radius: 2rem;
  cursor: pointer;
  font-size: 1rem;
  transition: background-color 0.2s;
}

.send-button:hover:not(:disabled) {
  background-color: #369e6f;
}

.send-button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}
</style>
