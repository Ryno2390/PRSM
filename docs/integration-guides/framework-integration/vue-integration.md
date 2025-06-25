# Vue.js Integration Guide

Integrate PRSM into Vue.js applications for intelligent, reactive AI-powered user interfaces.

## 🎯 Overview

This guide covers integrating PRSM into Vue.js applications using the Composition API, reactive patterns, and modern Vue 3 features for seamless AI interactions.

## 📋 Prerequisites

- Vue 3.0+
- PRSM instance running
- Basic knowledge of Vue Composition API

## 🚀 Quick Start

### 1. Installation

```bash
npm install @prsm/vue-sdk axios
# or
yarn add @prsm/vue-sdk axios
```

### 2. Setup PRSM Plugin

```javascript
// main.js
import { createApp } from 'vue'
import App from './App.vue'
import { PRSMPlugin } from '@prsm/vue-sdk'

const app = createApp(App)

app.use(PRSMPlugin, {
  baseURL: import.meta.env.VITE_PRSM_URL || 'http://localhost:8000',
  apiKey: import.meta.env.VITE_PRSM_API_KEY,
  timeout: 30000
})

app.mount('#app')
```

### 3. Basic Chat Component

```vue
<!-- components/ChatInterface.vue -->
<template>
  <div class="chat-interface">
    <div class="conversation">
      <div 
        v-for="message in conversation" 
        :key="message.id"
        :class="['message', message.role]"
      >
        <div class="message-content">
          <strong>{{ message.role === 'user' ? 'You' : 'AI' }}:</strong>
          {{ message.content }}
        </div>
        <div class="message-meta" v-if="message.confidence">
          Confidence: {{ (message.confidence * 100).toFixed(1) }}%
        </div>
      </div>
      
      <div v-if="loading" class="message loading">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
        AI is thinking...
      </div>
      
      <div v-if="error" class="message error">
        Error: {{ error }}
      </div>
    </div>
    
    <form @submit.prevent="handleSubmit" class="message-form">
      <input
        v-model="message"
        type="text"
        placeholder="Type your message..."
        :disabled="loading"
        class="message-input"
      />
      <button 
        type="submit" 
        :disabled="loading || !message.trim()"
        class="send-button"
      >
        {{ loading ? '...' : 'Send' }}
      </button>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { usePRSM } from '@prsm/vue-sdk'

const { query, loading, error } = usePRSM()

const message = ref('')
const conversation = ref([
  {
    id: 1,
    role: 'assistant',
    content: 'Hello! How can I help you today?'
  }
])

const handleSubmit = async () => {
  if (!message.value.trim() || loading.value) return

  const userMessage = {
    id: Date.now(),
    role: 'user',
    content: message.value
  }

  conversation.value.push(userMessage)
  const currentMessage = message.value
  message.value = ''

  try {
    const response = await query({
      prompt: currentMessage,
      userId: 'demo-user',
      context: { conversationId: 'demo-conversation' }
    })

    const aiMessage = {
      id: Date.now() + 1,
      role: 'assistant',
      content: response.answer,
      confidence: response.confidence
    }

    conversation.value.push(aiMessage)
  } catch (err) {
    console.error('Chat error:', err)
    conversation.value.push({
      id: Date.now() + 1,
      role: 'error',
      content: 'Sorry, something went wrong.'
    })
  }
}
</script>

<style scoped>
.chat-interface {
  display: flex;
  flex-direction: column;
  height: 500px;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  overflow: hidden;
}

.conversation {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: #f8f9fa;
}

.message {
  margin: 0.5rem 0;
  padding: 0.75rem 1rem;
  border-radius: 1rem;
  max-width: 80%;
}

.message.user {
  background: #007bff;
  color: white;
  margin-left: auto;
}

.message.assistant {
  background: white;
  border: 1px solid #dee2e6;
}

.message.error {
  background: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}

.message-form {
  display: flex;
  padding: 1rem;
  border-top: 1px solid #dee2e6;
  background: white;
}

.message-input {
  flex: 1;
  padding: 0.5rem 1rem;
  border: 1px solid #ced4da;
  border-radius: 1.5rem;
  outline: none;
}

.send-button {
  margin-left: 0.5rem;
  padding: 0.5rem 1rem;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 1.5rem;
  cursor: pointer;
}

.typing-indicator {
  display: inline-block;
}

.typing-indicator span {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #007bff;
  animation: typing 1.4s infinite ease-in-out;
  margin: 0 2px;
}

.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
  0%, 80%, 100% {
    transform: scale(0);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}
</style>
```

## 🧩 Composables

### Advanced PRSM Composable

```javascript
// composables/usePRSMAdvanced.js
import { ref, computed, onUnmounted } from 'vue'
import { usePRSMClient } from '@prsm/vue-sdk'

export function usePRSMAdvanced() {
  const client = usePRSMClient()
  const loading = ref(false)
  const error = ref(null)
  const history = ref([])
  const abortController = ref(null)

  const totalQueries = computed(() => history.value.length)
  const averageConfidence = computed(() => {
    const confidenceValues = history.value
      .map(item => item.confidence)
      .filter(Boolean)
    
    if (confidenceValues.length === 0) return 0
    
    return confidenceValues.reduce((sum, conf) => sum + conf, 0) / confidenceValues.length
  })

  const query = async (prompt, options = {}) => {
    if (abortController.value) {
      abortController.value.abort()
    }

    abortController.value = new AbortController()
    loading.value = true
    error.value = null

    try {
      const response = await client.query({
        prompt,
        signal: abortController.value.signal,
        ...options
      })

      const historyEntry = {
        id: Date.now(),
        prompt,
        response: response.answer,
        confidence: response.confidence,
        timestamp: new Date(),
        usage: response.usage
      }

      history.value.unshift(historyEntry)
      return response

    } catch (err) {
      if (err.name !== 'AbortError') {
        error.value = err.message
        throw err
      }
    } finally {
      loading.value = false
      abortController.value = null
    }
  }

  const cancelQuery = () => {
    if (abortController.value) {
      abortController.value.abort()
    }
  }

  const clearHistory = () => {
    history.value = []
  }

  const retryLastQuery = async () => {
    const lastEntry = history.value[0]
    if (lastEntry) {
      return await query(lastEntry.prompt)
    }
  }

  onUnmounted(() => {
    if (abortController.value) {
      abortController.value.abort()
    }
  })

  return {
    query,
    cancelQuery,
    clearHistory,
    retryLastQuery,
    loading: computed(() => loading.value),
    error: computed(() => error.value),
    history: computed(() => history.value),
    totalQueries,
    averageConfidence
  }
}
```

### Streaming Composable

```javascript
// composables/usePRSMStream.js
import { ref, computed, onUnmounted } from 'vue'
import { usePRSMClient } from '@prsm/vue-sdk'

export function usePRSMStream() {
  const client = usePRSMClient()
  const streaming = ref(false)
  const streamContent = ref('')
  const error = ref(null)
  const currentStream = ref(null)

  const hasContent = computed(() => streamContent.value.length > 0)

  const startStream = async (prompt, options = {}) => {
    streaming.value = true
    streamContent.value = ''
    error.value = null

    try {
      currentStream.value = await client.streamQuery({
        prompt,
        ...options
      })

      currentStream.value.on('data', (chunk) => {
        streamContent.value += chunk.content
      })

      currentStream.value.on('complete', (finalData) => {
        streaming.value = false
      })

      currentStream.value.on('error', (err) => {
        error.value = err.message
        streaming.value = false
      })

    } catch (err) {
      error.value = err.message
      streaming.value = false
    }
  }

  const stopStream = () => {
    if (currentStream.value) {
      currentStream.value.abort()
      streaming.value = false
    }
  }

  const clearContent = () => {
    streamContent.value = ''
  }

  onUnmounted(() => {
    stopStream()
  })

  return {
    startStream,
    stopStream,
    clearContent,
    streaming: computed(() => streaming.value),
    streamContent: computed(() => streamContent.value),
    error: computed(() => error.value),
    hasContent
  }
}
```

## 🔄 State Management with Pinia

### PRSM Store

```javascript
// stores/prsm.js
import { defineStore } from 'pinia'
import { PRSMClient } from '@prsm/sdk'

const client = new PRSMClient({
  baseURL: import.meta.env.VITE_PRSM_URL,
  apiKey: import.meta.env.VITE_PRSM_API_KEY
})

export const usePRSMStore = defineStore('prsm', {
  state: () => ({
    conversations: {},
    currentConversationId: null,
    loading: false,
    error: null,
    preferences: {
      autoSave: true,
      maxHistory: 50,
      theme: 'light'
    },
    usage: {
      totalQueries: 0,
      totalTokens: 0,
      totalCost: 0
    }
  }),

  getters: {
    currentConversation: (state) => {
      return state.currentConversationId 
        ? state.conversations[state.currentConversationId] 
        : null
    },
    
    conversationList: (state) => {
      return Object.values(state.conversations)
        .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))
    },

    isLoading: (state) => state.loading,
    
    hasError: (state) => !!state.error
  },

  actions: {
    async submitQuery(prompt, options = {}) {
      this.loading = true
      this.error = null

      try {
        const response = await client.query({
          prompt,
          ...options
        })

        // Update usage statistics
        this.usage.totalQueries++
        if (response.usage?.tokens) {
          this.usage.totalTokens += response.usage.tokens
        }
        if (response.usage?.cost) {
          this.usage.totalCost += response.usage.cost
        }

        // Add to conversation
        const conversationId = options.conversationId || this.currentConversationId
        if (conversationId) {
          this.addMessage(conversationId, {
            role: 'assistant',
            content: response.answer,
            confidence: response.confidence,
            timestamp: new Date().toISOString()
          })
        }

        return response

      } catch (error) {
        this.error = error.message
        throw error
      } finally {
        this.loading = false
      }
    },

    createConversation(title = 'New Conversation') {
      const id = Date.now().toString()
      const conversation = {
        id,
        title,
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }

      this.conversations[id] = conversation
      this.currentConversationId = id
      
      return id
    },

    addMessage(conversationId, message) {
      if (!this.conversations[conversationId]) {
        this.createConversation()
        conversationId = this.currentConversationId
      }

      const messageWithId = {
        id: Date.now(),
        ...message,
        timestamp: message.timestamp || new Date().toISOString()
      }

      this.conversations[conversationId].messages.push(messageWithId)
      this.conversations[conversationId].updatedAt = new Date().toISOString()

      // Auto-save if enabled
      if (this.preferences.autoSave) {
        this.saveToLocalStorage()
      }
    },

    deleteConversation(conversationId) {
      delete this.conversations[conversationId]
      if (this.currentConversationId === conversationId) {
        this.currentConversationId = null
      }
    },

    setCurrentConversation(conversationId) {
      this.currentConversationId = conversationId
    },

    updatePreferences(newPreferences) {
      this.preferences = { ...this.preferences, ...newPreferences }
      this.saveToLocalStorage()
    },

    clearError() {
      this.error = null
    },

    saveToLocalStorage() {
      localStorage.setItem('prsm-store', JSON.stringify({
        conversations: this.conversations,
        preferences: this.preferences,
        usage: this.usage
      }))
    },

    loadFromLocalStorage() {
      const saved = localStorage.getItem('prsm-store')
      if (saved) {
        try {
          const data = JSON.parse(saved)
          this.conversations = data.conversations || {}
          this.preferences = { ...this.preferences, ...data.preferences }
          this.usage = { ...this.usage, ...data.usage }
        } catch (error) {
          console.error('Failed to load from localStorage:', error)
        }
      }
    }
  }
})
```

### Using Store in Components

```vue
<!-- components/ConversationManager.vue -->
<template>
  <div class="conversation-manager">
    <div class="sidebar">
      <button @click="createNewConversation" class="new-conversation-btn">
        New Conversation
      </button>
      
      <div class="conversation-list">
        <div
          v-for="conversation in conversationList"
          :key="conversation.id"
          :class="['conversation-item', { active: currentConversationId === conversation.id }]"
          @click="setCurrentConversation(conversation.id)"
        >
          <h4>{{ conversation.title }}</h4>
          <small>{{ formatDate(conversation.updatedAt) }}</small>
          <button 
            @click.stop="deleteConversation(conversation.id)"
            class="delete-btn"
          >
            ×
          </button>
        </div>
      </div>
    </div>
    
    <div class="chat-area">
      <ChatInterface 
        v-if="currentConversationId"
        :conversation-id="currentConversationId"
      />
      <div v-else class="no-conversation">
        Select or create a conversation to start chatting
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { usePRSMStore } from '@/stores/prsm'
import ChatInterface from './ChatInterface.vue'

const store = usePRSMStore()

const conversationList = computed(() => store.conversationList)
const currentConversationId = computed(() => store.currentConversationId)

const createNewConversation = () => {
  store.createConversation()
}

const setCurrentConversation = (id) => {
  store.setCurrentConversation(id)
}

const deleteConversation = (id) => {
  if (confirm('Are you sure you want to delete this conversation?')) {
    store.deleteConversation(id)
  }
}

const formatDate = (dateStr) => {
  return new Date(dateStr).toLocaleDateString()
}

// Load saved data on mount
store.loadFromLocalStorage()
</script>
```

## 📱 Mobile-First Components

### Responsive Chat Interface

```vue
<!-- components/ResponsiveChatInterface.vue -->
<template>
  <div class="responsive-chat" :class="{ mobile: isMobile, minimized: isMinimized }">
    <div class="chat-header" v-if="isMobile">
      <h3>AI Assistant</h3>
      <button @click="toggleMinimized" class="minimize-btn">
        {{ isMinimized ? '▲' : '▼' }}
      </button>
    </div>
    
    <Transition name="slide-fade">
      <div v-if="!isMinimized" class="chat-content">
        <div class="messages" ref="messagesContainer">
          <TransitionGroup name="message" tag="div">
            <div
              v-for="message in messages"
              :key="message.id"
              :class="['message', message.role]"
            >
              {{ message.content }}
            </div>
          </TransitionGroup>
        </div>
        
        <div class="input-area">
          <div class="suggestions" v-if="suggestions.length > 0">
            <button
              v-for="suggestion in suggestions"
              :key="suggestion"
              @click="sendMessage(suggestion)"
              class="suggestion-chip"
            >
              {{ suggestion }}
            </button>
          </div>
          
          <form @submit.prevent="handleSubmit" class="input-form">
            <input
              v-model="currentMessage"
              type="text"
              placeholder="Type a message..."
              :disabled="loading"
              class="message-input"
              @focus="onInputFocus"
              @blur="onInputBlur"
            />
            <button 
              type="submit" 
              :disabled="loading || !currentMessage.trim()"
              class="send-btn"
            >
              <svg width="20" height="20" viewBox="0 0 24 24">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
              </svg>
            </button>
          </form>
        </div>
      </div>
    </Transition>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { usePRSMAdvanced } from '@/composables/usePRSMAdvanced'

const props = defineProps({
  userId: {
    type: String,
    default: 'default-user'
  },
  suggestions: {
    type: Array,
    default: () => []
  }
})

const { query, loading } = usePRSMAdvanced()

const isMobile = ref(false)
const isMinimized = ref(false)
const currentMessage = ref('')
const messages = ref([])
const messagesContainer = ref(null)

const handleResize = () => {
  isMobile.value = window.innerWidth <= 768
}

const toggleMinimized = () => {
  isMinimized.value = !isMinimized.value
}

const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

const sendMessage = async (content) => {
  if (!content.trim()) return

  const userMessage = {
    id: Date.now(),
    role: 'user',
    content,
    timestamp: new Date()
  }

  messages.value.push(userMessage)
  scrollToBottom()

  try {
    const response = await query(content, {
      userId: props.userId,
      context: { mobile: isMobile.value }
    })

    const aiMessage = {
      id: Date.now() + 1,
      role: 'assistant',
      content: response.answer,
      timestamp: new Date()
    }

    messages.value.push(aiMessage)
    scrollToBottom()

  } catch (error) {
    console.error('Failed to send message:', error)
    messages.value.push({
      id: Date.now() + 1,
      role: 'error',
      content: 'Failed to send message. Please try again.',
      timestamp: new Date()
    })
  }
}

const handleSubmit = () => {
  sendMessage(currentMessage.value)
  currentMessage.value = ''
}

const onInputFocus = () => {
  if (isMobile.value) {
    setTimeout(scrollToBottom, 300) // Wait for keyboard animation
  }
}

const onInputBlur = () => {
  // Handle input blur if needed
}

watch(messages, scrollToBottom, { deep: true })

onMounted(() => {
  handleResize()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.responsive-chat {
  display: flex;
  flex-direction: column;
  height: 100%;
  max-height: 600px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  background: white;
}

.responsive-chat.mobile {
  position: fixed;
  bottom: 0;
  right: 1rem;
  width: calc(100vw - 2rem);
  max-width: 400px;
  z-index: 1000;
}

.responsive-chat.minimized {
  height: 60px;
}

.chat-header {
  padding: 1rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.minimize-btn {
  background: none;
  border: none;
  color: white;
  font-size: 1.2rem;
  cursor: pointer;
}

.chat-content {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: linear-gradient(to bottom, #f8f9fa, #ffffff);
}

.message {
  margin: 0.5rem 0;
  padding: 0.75rem 1rem;
  border-radius: 18px;
  max-width: 80%;
  animation: slideIn 0.3s ease-out;
}

.message.user {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  margin-left: auto;
  border-bottom-right-radius: 4px;
}

.message.assistant {
  background: #f1f3f4;
  border-bottom-left-radius: 4px;
}

.message.error {
  background: #fee;
  color: #c53030;
  border: 1px solid #fed7d7;
}

.suggestions {
  padding: 0.5rem 1rem;
  display: flex;
  gap: 0.5rem;
  overflow-x: auto;
}

.suggestion-chip {
  background: #e2e8f0;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 1rem;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.2s;
}

.suggestion-chip:hover {
  background: #cbd5e0;
}

.input-form {
  display: flex;
  padding: 1rem;
  border-top: 1px solid #e2e8f0;
  background: white;
}

.message-input {
  flex: 1;
  padding: 0.75rem 1rem;
  border: 1px solid #e2e8f0;
  border-radius: 24px;
  outline: none;
  font-size: 1rem;
}

.send-btn {
  margin-left: 0.5rem;
  width: 48px;
  height: 48px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Transitions */
.slide-fade-enter-active,
.slide-fade-leave-active {
  transition: all 0.3s ease;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  transform: translateY(20px);
  opacity: 0;
}

.message-enter-active {
  transition: all 0.3s ease;
}

.message-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 768px) {
  .responsive-chat:not(.mobile) {
    margin: 0;
    border-radius: 0;
    height: 100vh;
  }
  
  .message {
    max-width: 90%;
  }
  
  .input-form {
    padding: 1rem;
    padding-bottom: calc(1rem + env(safe-area-inset-bottom));
  }
}
</style>
```

## 🔧 Directives and Plugins

### Auto-scroll Directive

```javascript
// directives/autoScroll.js
export const autoScroll = {
  mounted(el) {
    const observer = new MutationObserver(() => {
      el.scrollTop = el.scrollHeight
    })
    
    observer.observe(el, {
      childList: true,
      subtree: true
    })
    
    el._autoScrollObserver = observer
  },
  
  unmounted(el) {
    if (el._autoScrollObserver) {
      el._autoScrollObserver.disconnect()
    }
  }
}

// Usage in component
// <div v-auto-scroll class="messages">
```

### Voice Input Directive

```javascript
// directives/voiceInput.js
export const voiceInput = {
  mounted(el, binding) {
    if (!('webkitSpeechRecognition' in window)) {
      console.warn('Speech recognition not supported')
      return
    }

    const recognition = new webkitSpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = false
    recognition.lang = 'en-US'

    const button = document.createElement('button')
    button.innerHTML = '🎤'
    button.className = 'voice-input-btn'
    button.type = 'button'
    
    button.addEventListener('click', () => {
      recognition.start()
    })

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      el.value = transcript
      el.dispatchEvent(new Event('input'))
      
      if (binding.value) {
        binding.value(transcript)
      }
    }

    el.parentNode.appendChild(button)
    el._voiceButton = button
    el._recognition = recognition
  },

  unmounted(el) {
    if (el._voiceButton) {
      el._voiceButton.remove()
    }
    if (el._recognition) {
      el._recognition.abort()
    }
  }
}
```

## 🧪 Testing

### Component Testing with Vue Test Utils

```javascript
// tests/components/ChatInterface.spec.js
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import ChatInterface from '@/components/ChatInterface.vue'
import { usePRSMStore } from '@/stores/prsm'

// Mock the PRSM SDK
vi.mock('@prsm/vue-sdk', () => ({
  usePRSM: () => ({
    query: vi.fn(),
    loading: ref(false),
    error: ref(null)
  })
}))

describe('ChatInterface', () => {
  let wrapper
  let store

  beforeEach(() => {
    setActivePinia(createPinia())
    store = usePRSMStore()
    
    wrapper = mount(ChatInterface, {
      global: {
        plugins: [createPinia()]
      }
    })
  })

  afterEach(() => {
    wrapper.unmount()
  })

  test('renders chat interface', () => {
    expect(wrapper.find('.chat-interface').exists()).toBe(true)
    expect(wrapper.find('.message-input').exists()).toBe(true)
    expect(wrapper.find('.send-button').exists()).toBe(true)
  })

  test('sends message on form submit', async () => {
    const input = wrapper.find('.message-input')
    const form = wrapper.find('.message-form')

    await input.setValue('Test message')
    await form.trigger('submit')

    expect(wrapper.vm.conversation).toHaveLength(2) // Initial + user message
  })

  test('disables input when loading', async () => {
    wrapper.vm.loading = true
    await wrapper.vm.$nextTick()

    const input = wrapper.find('.message-input')
    const button = wrapper.find('.send-button')

    expect(input.attributes('disabled')).toBeDefined()
    expect(button.attributes('disabled')).toBeDefined()
  })

  test('displays error message', async () => {
    wrapper.vm.error = 'Test error'
    await wrapper.vm.$nextTick()

    expect(wrapper.find('.message.error').exists()).toBe(true)
    expect(wrapper.text()).toContain('Test error')
  })
})
```

### Composable Testing

```javascript
// tests/composables/usePRSMAdvanced.spec.js
import { usePRSMAdvanced } from '@/composables/usePRSMAdvanced'

// Mock the client
const mockClient = {
  query: vi.fn()
}

vi.mock('@prsm/vue-sdk', () => ({
  usePRSMClient: () => mockClient
}))

describe('usePRSMAdvanced', () => {
  test('executes query successfully', async () => {
    const mockResponse = {
      answer: 'Test response',
      confidence: 0.9,
      usage: { tokens: 50 }
    }
    mockClient.query.mockResolvedValue(mockResponse)

    const { query, history } = usePRSMAdvanced()
    
    await query('Test prompt', { userId: 'test' })

    expect(history.value).toHaveLength(1)
    expect(history.value[0].prompt).toBe('Test prompt')
    expect(history.value[0].response).toBe('Test response')
  })

  test('handles query errors', async () => {
    mockClient.query.mockRejectedValue(new Error('API Error'))

    const { query, error } = usePRSMAdvanced()

    await expect(query('Test prompt')).rejects.toThrow('API Error')
    expect(error.value).toBe('API Error')
  })

  test('cancels ongoing queries', async () => {
    const { query, cancelQuery } = usePRSMAdvanced()

    const queryPromise = query('Test prompt')
    cancelQuery()

    // Query should be cancelled
    await expect(queryPromise).rejects.toThrow()
  })
})
```

## 🚀 Performance Optimization

### Lazy Loading Components

```vue
<!-- components/LazyAIFeatures.vue -->
<template>
  <div class="ai-features">
    <Suspense>
      <template #default>
        <component 
          :is="currentComponent"
          v-bind="componentProps"
        />
      </template>
      <template #fallback>
        <div class="loading-skeleton">
          <div class="skeleton-line"></div>
          <div class="skeleton-line"></div>
          <div class="skeleton-line"></div>
        </div>
      </template>
    </Suspense>
  </div>
</template>

<script setup>
import { ref, computed, defineAsyncComponent } from 'vue'

const props = defineProps({
  feature: {
    type: String,
    required: true
  }
})

// Lazy load components
const ChatInterface = defineAsyncComponent(() => import('./ChatInterface.vue'))
const VoiceChat = defineAsyncComponent(() => import('./VoiceChat.vue'))
const DocumentAnalyzer = defineAsyncComponent(() => import('./DocumentAnalyzer.vue'))

const components = {
  chat: ChatInterface,
  voice: VoiceChat,
  document: DocumentAnalyzer
}

const currentComponent = computed(() => components[props.feature])
const componentProps = computed(() => ({
  // Pass common props to all components
  userId: 'current-user'
}))
</script>
```

### Virtual Scrolling for Large Conversations

```vue
<!-- components/VirtualConversation.vue -->
<template>
  <div class="virtual-conversation" ref="container">
    <RecycleScroller
      class="scroller"
      :items="messages"
      :item-size="estimatedItemSize"
      key-field="id"
      v-slot="{ item, index }"
    >
      <div class="message-wrapper">
        <ChatMessage
          :message="item"
          :index="index"
          @resize="onMessageResize"
        />
      </div>
    </RecycleScroller>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { RecycleScroller } from 'vue-virtual-scroller'
import ChatMessage from './ChatMessage.vue'

const props = defineProps({
  messages: {
    type: Array,
    required: true
  }
})

const container = ref(null)
const estimatedItemSize = ref(80)
const itemSizes = ref(new Map())

const onMessageResize = (messageId, height) => {
  itemSizes.value.set(messageId, height)
  
  // Update estimated size based on average
  const sizes = Array.from(itemSizes.value.values())
  estimatedItemSize.value = sizes.reduce((sum, size) => sum + size, 0) / sizes.length
}
</script>
```

## 📦 Production Build

### Vite Configuration

```javascript
// vite.config.js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'prsm-sdk': ['@prsm/vue-sdk'],
          'vue-vendor': ['vue', 'pinia'],
          'ui-components': ['./src/components']
        }
      }
    }
  },
  define: {
    'process.env': process.env
  }
})
```

### Environment Configuration

```javascript
// .env.production
VITE_PRSM_URL=https://api.prsm.ai
VITE_PRSM_API_KEY=your_production_api_key
VITE_APP_TITLE=PRSM AI Assistant
VITE_ENABLE_ANALYTICS=true
```

## 📋 Best Practices

### Error Handling

```vue
<!-- components/ErrorBoundary.vue -->
<template>
  <div class="error-boundary">
    <slot v-if="!hasError" />
    <div v-else class="error-fallback">
      <h2>Something went wrong</h2>
      <p>{{ error?.message || 'An unexpected error occurred' }}</p>
      <button @click="retry" class="retry-btn">
        Try Again
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, onErrorCaptured } from 'vue'

const hasError = ref(false)
const error = ref(null)

onErrorCaptured((err, instance, info) => {
  console.error('Error captured:', err, info)
  hasError.value = true
  error.value = err
  
  // Send to error reporting service
  if (import.meta.env.PROD) {
    reportError(err, { instance, info })
  }
  
  return false // Stop propagation
})

const retry = () => {
  hasError.value = false
  error.value = null
}

const reportError = (error, context) => {
  // Implementation for error reporting
  console.log('Reporting error:', error, context)
}
</script>
```

### Input Validation

```javascript
// utils/validation.js
export const validateMessage = (message) => {
  if (!message || typeof message !== 'string') {
    throw new Error('Message must be a non-empty string')
  }
  
  if (message.length > 10000) {
    throw new Error('Message is too long (max 10,000 characters)')
  }
  
  // Remove potential XSS vectors
  return message
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .trim()
}

export const sanitizeUserInput = (input) => {
  return input
    .replace(/[<>]/g, '')
    .trim()
}
```

---

**Need help with Vue.js integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).