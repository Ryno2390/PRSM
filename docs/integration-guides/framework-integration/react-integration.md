# React Integration Guide

Integrate PRSM into React applications for intelligent, AI-powered user interfaces and experiences.

## 🎯 Overview

This guide covers integrating PRSM into React applications, including hooks, components, state management, and real-time features for modern web applications.

## 📋 Prerequisites

- React 16.8+ (for hooks support)
- PRSM instance running
- Basic knowledge of React hooks and modern patterns

## 🚀 Quick Start

### 1. Installation

```bash
npm install @prsm/react-sdk axios
# or
yarn add @prsm/react-sdk axios
```

### 2. Setup PRSM Provider

```jsx
// App.js
import React from 'react';
import { PRSMProvider } from '@prsm/react-sdk';
import ChatInterface from './components/ChatInterface';

const App = () => {
  return (
    <PRSMProvider
      config={{
        baseURL: process.env.REACT_APP_PRSM_URL || 'http://localhost:8000',
        apiKey: process.env.REACT_APP_PRSM_API_KEY,
        timeout: 30000
      }}
    >
      <div className="App">
        <ChatInterface />
      </div>
    </PRSMProvider>
  );
};

export default App;
```

### 3. Basic Chat Component

```jsx
// components/ChatInterface.js
import React, { useState } from 'react';
import { usePRSM } from '@prsm/react-sdk';

const ChatInterface = () => {
  const [message, setMessage] = useState('');
  const [conversation, setConversation] = useState([]);
  const { query, loading, error } = usePRSM();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    const userMessage = { role: 'user', content: message };
    setConversation(prev => [...prev, userMessage]);
    setMessage('');

    try {
      const response = await query({
        prompt: message,
        userId: 'demo-user',
        context: { conversationId: 'demo-conversation' }
      });

      const aiMessage = { role: 'assistant', content: response.answer };
      setConversation(prev => [...prev, aiMessage]);
    } catch (err) {
      console.error('Chat error:', err);
      const errorMessage = { role: 'error', content: 'Sorry, something went wrong.' };
      setConversation(prev => [...prev, errorMessage]);
    }
  };

  return (
    <div className="chat-interface">
      <div className="conversation">
        {conversation.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'You' : 'AI'}:</strong> {msg.content}
          </div>
        ))}
        {loading && <div className="message loading">AI is thinking...</div>}
        {error && <div className="message error">Error: {error}</div>}
      </div>
      
      <form onSubmit={handleSubmit} className="message-form">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !message.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
```

## 🎣 Custom Hooks

### Advanced PRSM Hook

```jsx
// hooks/usePRSMAdvanced.js
import { useState, useCallback, useRef, useEffect } from 'react';
import { usePRSMContext } from '@prsm/react-sdk';

export const usePRSMAdvanced = () => {
  const { client } = usePRSMContext();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const abortControllerRef = useRef(null);

  const query = useCallback(async (prompt, options = {}) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setLoading(true);
    setError(null);

    try {
      const response = await client.query({
        prompt,
        signal: abortControllerRef.current.signal,
        ...options
      });

      const historyEntry = {
        id: Date.now(),
        prompt,
        response: response.answer,
        confidence: response.confidence,
        timestamp: new Date(),
        usage: response.usage
      };

      setHistory(prev => [historyEntry, ...prev]);
      return response;

    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
        throw err;
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  }, [client]);

  const cancelQuery = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    query,
    cancelQuery,
    clearHistory,
    loading,
    error,
    history
  };
};
```

### Streaming Hook

```jsx
// hooks/usePRSMStream.js
import { useState, useCallback, useRef, useEffect } from 'react';
import { usePRSMContext } from '@prsm/react-sdk';

export const usePRSMStream = () => {
  const { client } = usePRSMContext();
  const [streaming, setStreaming] = useState(false);
  const [streamContent, setStreamContent] = useState('');
  const [error, setError] = useState(null);
  const streamRef = useRef(null);

  const startStream = useCallback(async (prompt, options = {}) => {
    setStreaming(true);
    setStreamContent('');
    setError(null);

    try {
      streamRef.current = await client.streamQuery({
        prompt,
        ...options
      });

      streamRef.current.on('data', (chunk) => {
        setStreamContent(prev => prev + chunk.content);
      });

      streamRef.current.on('complete', (finalData) => {
        setStreaming(false);
      });

      streamRef.current.on('error', (err) => {
        setError(err.message);
        setStreaming(false);
      });

    } catch (err) {
      setError(err.message);
      setStreaming(false);
    }
  }, [client]);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.abort();
      setStreaming(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.abort();
      }
    };
  }, []);

  return {
    startStream,
    stopStream,
    streaming,
    streamContent,
    error
  };
};
```

## 🧩 Reusable Components

### AI Chat Component

```jsx
// components/AIChat.js
import React, { useState, useRef, useEffect } from 'react';
import { usePRSMAdvanced } from '../hooks/usePRSMAdvanced';
import './AIChat.css';

const AIChat = ({ 
  userId = 'default-user',
  placeholder = 'Ask me anything...',
  welcomeMessage = 'Hello! How can I help you today?',
  onNewMessage,
  className = ''
}) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { role: 'assistant', content: welcomeMessage, timestamp: new Date() }
  ]);
  const messagesEndRef = useRef(null);
  const { query, loading, error } = usePRSMAdvanced();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');

    try {
      const response = await query(currentInput, {
        userId,
        context: {
          conversationHistory: messages.slice(-5) // Last 5 messages for context
        }
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.answer,
        confidence: response.confidence,
        timestamp: new Date(),
        usage: response.usage
      };

      setMessages(prev => [...prev, assistantMessage]);
      
      if (onNewMessage) {
        onNewMessage(userMessage, assistantMessage);
      }

    } catch (err) {
      const errorMessage = {
        role: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  return (
    <div className={`ai-chat ${className}`}>
      <div className="messages-container">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-content">
              {msg.content}
            </div>
            <div className="message-meta">
              {msg.confidence && (
                <span className="confidence">
                  Confidence: {(msg.confidence * 100).toFixed(1)}%
                </span>
              )}
              <span className="timestamp">
                {msg.timestamp.toLocaleTimeString()}
              </span>
            </div>
          </div>
        ))}
        {loading && (
          <div className="message assistant loading">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={placeholder}
          disabled={loading}
          className="message-input"
        />
        <button 
          type="submit" 
          disabled={loading || !input.trim()}
          className="send-button"
        >
          {loading ? '...' : 'Send'}
        </button>
      </form>

      {error && (
        <div className="error-banner">
          Error: {error}
        </div>
      )}
    </div>
  );
};

export default AIChat;
```

### Smart Suggestions Component

```jsx
// components/SmartSuggestions.js
import React, { useState, useEffect } from 'react';
import { usePRSM } from '@prsm/react-sdk';

const SmartSuggestions = ({ 
  context = '',
  onSuggestionSelect,
  maxSuggestions = 3 
}) => {
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const { query } = usePRSM();

  useEffect(() => {
    if (context.length > 10) {
      generateSuggestions(context);
    }
  }, [context]);

  const generateSuggestions = async (inputContext) => {
    setLoading(true);
    try {
      const response = await query({
        prompt: `Based on this context: "${inputContext}", suggest ${maxSuggestions} relevant follow-up questions or topics. Return only the suggestions, one per line.`,
        userId: 'suggestions-user',
        context: { type: 'suggestion_generation' }
      });

      const suggestionList = response.answer
        .split('\n')
        .filter(s => s.trim())
        .slice(0, maxSuggestions);

      setSuggestions(suggestionList);
    } catch (err) {
      console.error('Failed to generate suggestions:', err);
      setSuggestions([]);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="suggestions-loading">Generating suggestions...</div>;
  }

  if (suggestions.length === 0) {
    return null;
  }

  return (
    <div className="smart-suggestions">
      <h4>Suggestions:</h4>
      <div className="suggestions-list">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            className="suggestion-item"
            onClick={() => onSuggestionSelect(suggestion)}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SmartSuggestions;
```

## 🔄 State Management Integration

### Redux Integration

```jsx
// store/prsmSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { PRSMClient } from '@prsm/sdk';

const client = new PRSMClient({
  baseURL: process.env.REACT_APP_PRSM_URL,
  apiKey: process.env.REACT_APP_PRSM_API_KEY
});

export const submitQuery = createAsyncThunk(
  'prsm/submitQuery',
  async ({ prompt, userId, context }, { rejectWithValue }) => {
    try {
      const response = await client.query({ prompt, userId, context });
      return response;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

const prsmSlice = createSlice({
  name: 'prsm',
  initialState: {
    conversations: {},
    loading: false,
    error: null,
    currentConversation: null
  },
  reducers: {
    setCurrentConversation: (state, action) => {
      state.currentConversation = action.payload;
    },
    addMessage: (state, action) => {
      const { conversationId, message } = action.payload;
      if (!state.conversations[conversationId]) {
        state.conversations[conversationId] = [];
      }
      state.conversations[conversationId].push(message);
    },
    clearError: (state) => {
      state.error = null;
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(submitQuery.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(submitQuery.fulfilled, (state, action) => {
        state.loading = false;
        const { conversationId } = action.meta.arg.context || {};
        if (conversationId) {
          const assistantMessage = {
            role: 'assistant',
            content: action.payload.answer,
            timestamp: new Date().toISOString(),
            confidence: action.payload.confidence
          };
          if (!state.conversations[conversationId]) {
            state.conversations[conversationId] = [];
          }
          state.conversations[conversationId].push(assistantMessage);
        }
      })
      .addCase(submitQuery.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  }
});

export const { setCurrentConversation, addMessage, clearError } = prsmSlice.actions;
export default prsmSlice.reducer;
```

### Context API Pattern

```jsx
// contexts/PRSMContext.js
import React, { createContext, useContext, useReducer } from 'react';

const PRSMContext = createContext();

const initialState = {
  sessions: {},
  activeSession: null,
  preferences: {
    autoSave: true,
    maxHistory: 50,
    theme: 'light'
  }
};

function prsmReducer(state, action) {
  switch (action.type) {
    case 'CREATE_SESSION':
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [action.payload.id]: {
            id: action.payload.id,
            messages: [],
            createdAt: new Date(),
            updatedAt: new Date()
          }
        },
        activeSession: action.payload.id
      };
    
    case 'ADD_MESSAGE':
      const { sessionId, message } = action.payload;
      return {
        ...state,
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...state.sessions[sessionId],
            messages: [...(state.sessions[sessionId]?.messages || []), message],
            updatedAt: new Date()
          }
        }
      };
    
    case 'SET_PREFERENCES':
      return {
        ...state,
        preferences: { ...state.preferences, ...action.payload }
      };
    
    default:
      return state;
  }
}

export const PRSMProvider = ({ children }) => {
  const [state, dispatch] = useReducer(prsmReducer, initialState);

  const createSession = (id = Date.now().toString()) => {
    dispatch({ type: 'CREATE_SESSION', payload: { id } });
    return id;
  };

  const addMessage = (sessionId, message) => {
    dispatch({ type: 'ADD_MESSAGE', payload: { sessionId, message } });
  };

  const setPreferences = (preferences) => {
    dispatch({ type: 'SET_PREFERENCES', payload: preferences });
  };

  const value = {
    ...state,
    createSession,
    addMessage,
    setPreferences
  };

  return (
    <PRSMContext.Provider value={value}>
      {children}
    </PRSMContext.Provider>
  );
};

export const usePRSMContext = () => {
  const context = useContext(PRSMContext);
  if (!context) {
    throw new Error('usePRSMContext must be used within a PRSMProvider');
  }
  return context;
};
```

## 🔄 Real-time Features

### WebSocket Integration

```jsx
// hooks/usePRSMWebSocket.js
import { useState, useEffect, useRef, useCallback } from 'react';

export const usePRSMWebSocket = (url, options = {}) => {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);
  const ws = useRef(null);
  const reconnectAttempts = useRef(0);

  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        setConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
        console.log('PRSM WebSocket connected');
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMessages(prev => [...prev, data]);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.current.onclose = () => {
        setConnected(false);
        
        // Auto-reconnect logic
        if (reconnectAttempts.current < 5) {
          reconnectAttempts.current++;
          setTimeout(() => {
            console.log(`Reconnecting... Attempt ${reconnectAttempts.current}`);
            connect();
          }, 1000 * reconnectAttempts.current);
        }
      };

      ws.current.onerror = (error) => {
        setError('WebSocket connection error');
        console.error('WebSocket error:', error);
      };

    } catch (err) {
      setError('Failed to create WebSocket connection');
    }
  }, [url]);

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close();
    }
  }, []);

  const sendMessage = useCallback((message) => {
    if (ws.current && connected) {
      ws.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, [connected]);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    connected,
    messages,
    error,
    sendMessage,
    connect,
    disconnect
  };
};
```

### Server-Sent Events

```jsx
// hooks/usePRSMEvents.js
import { useState, useEffect, useRef } from 'react';

export const usePRSMEvents = (endpoint) => {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const eventSourceRef = useRef(null);

  useEffect(() => {
    const connectEventSource = () => {
      try {
        eventSourceRef.current = new EventSource(endpoint);

        eventSourceRef.current.onopen = () => {
          setConnected(true);
          setError(null);
        };

        eventSourceRef.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            setEvents(prev => [...prev, data]);
          } catch (err) {
            console.error('Failed to parse SSE data:', err);
          }
        };

        eventSourceRef.current.onerror = () => {
          setConnected(false);
          setError('Event source connection error');
        };

      } catch (err) {
        setError('Failed to create event source');
      }
    };

    connectEventSource();

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [endpoint]);

  const clearEvents = () => setEvents([]);

  return { events, connected, error, clearEvents };
};
```

## 📱 Mobile-Responsive Design

### Responsive Chat Component

```jsx
// components/ResponsiveChat.js
import React, { useState, useEffect } from 'react';
import { usePRSM } from '@prsm/react-sdk';
import './ResponsiveChat.css';

const ResponsiveChat = () => {
  const [isMobile, setIsMobile] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const { query, loading } = usePRSM();

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return (
    <div className={`responsive-chat ${isMobile ? 'mobile' : 'desktop'} ${isMinimized ? 'minimized' : ''}`}>
      {isMobile && (
        <div className="chat-header">
          <h3>AI Assistant</h3>
          <button 
            className="minimize-btn"
            onClick={() => setIsMinimized(!isMinimized)}
          >
            {isMinimized ? '↑' : '↓'}
          </button>
        </div>
      )}
      
      {!isMinimized && (
        <div className="chat-content">
          {/* Chat content here */}
        </div>
      )}
    </div>
  );
};

export default ResponsiveChat;
```

## 🎨 Styling with CSS-in-JS

### Styled Components

```jsx
// components/StyledChat.js
import styled from 'styled-components';
import { usePRSM } from '@prsm/react-sdk';

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 500px;
  border: 1px solid #e1e5e9;
  border-radius: 8px;
  overflow: hidden;
  background: white;
  
  @media (max-width: 768px) {
    height: 100vh;
    border-radius: 0;
  }
`;

const MessagesArea = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background: linear-gradient(to bottom, #f8f9fa, #ffffff);
`;

const Message = styled.div`
  margin: 8px 0;
  padding: 12px 16px;
  border-radius: 18px;
  max-width: 80%;
  
  ${props => props.role === 'user' ? `
    background: #007bff;
    color: white;
    margin-left: auto;
    border-bottom-right-radius: 4px;
  ` : `
    background: #f1f3f4;
    color: #333;
    border-bottom-left-radius: 4px;
  `}
  
  ${props => props.loading && `
    background: #e9ecef;
    animation: pulse 1.5s ease-in-out infinite;
  `}
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;

const InputForm = styled.form`
  display: flex;
  padding: 16px;
  border-top: 1px solid #e1e5e9;
  background: white;
`;

const MessageInput = styled.input`
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #ced4da;
  border-radius: 24px;
  outline: none;
  
  &:focus {
    border-color: #007bff;
    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
  }
  
  &:disabled {
    background-color: #f8f9fa;
    opacity: 0.6;
  }
`;

const SendButton = styled.button`
  margin-left: 8px;
  padding: 12px 24px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 24px;
  cursor: pointer;
  
  &:hover:not(:disabled) {
    background: #0056b3;
  }
  
  &:disabled {
    background: #6c757d;
    cursor: not-allowed;
  }
`;

const StyledChat = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const { query, loading } = usePRSM();

  return (
    <ChatContainer>
      <MessagesArea>
        {messages.map((msg, index) => (
          <Message key={index} role={msg.role} loading={msg.loading}>
            {msg.content}
          </Message>
        ))}
      </MessagesArea>
      
      <InputForm onSubmit={handleSubmit}>
        <MessageInput
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={loading}
        />
        <SendButton type="submit" disabled={loading || !input.trim()}>
          {loading ? '...' : 'Send'}
        </SendButton>
      </InputForm>
    </ChatContainer>
  );
};

export default StyledChat;
```

## 🧪 Testing

### Component Testing with React Testing Library

```jsx
// __tests__/AIChat.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { PRSMProvider } from '@prsm/react-sdk';
import AIChat from '../components/AIChat';

// Mock the PRSM SDK
jest.mock('@prsm/react-sdk', () => ({
  ...jest.requireActual('@prsm/react-sdk'),
  usePRSM: () => ({
    query: jest.fn(),
    loading: false,
    error: null
  })
}));

const renderWithProvider = (component) => {
  return render(
    <PRSMProvider config={{ baseURL: 'http://test', apiKey: 'test' }}>
      {component}
    </PRSMProvider>
  );
};

describe('AIChat Component', () => {
  test('renders chat interface', () => {
    renderWithProvider(<AIChat />);
    
    expect(screen.getByPlaceholderText('Ask me anything...')).toBeInTheDocument();
    expect(screen.getByText('Send')).toBeInTheDocument();
  });

  test('sends message on form submit', async () => {
    const mockQuery = jest.fn().mockResolvedValue({
      answer: 'Test response',
      confidence: 0.95
    });

    jest.doMock('@prsm/react-sdk', () => ({
      usePRSM: () => ({
        query: mockQuery,
        loading: false,
        error: null
      })
    }));

    renderWithProvider(<AIChat />);
    
    const input = screen.getByPlaceholderText('Ask me anything...');
    const sendButton = screen.getByText('Send');

    fireEvent.change(input, { target: { value: 'Test message' } });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(mockQuery).toHaveBeenCalledWith('Test message', expect.any(Object));
    });
  });

  test('displays loading state', () => {
    jest.doMock('@prsm/react-sdk', () => ({
      usePRSM: () => ({
        query: jest.fn(),
        loading: true,
        error: null
      })
    }));

    renderWithProvider(<AIChat />);
    
    expect(screen.getByText('AI is thinking...')).toBeInTheDocument();
  });
});
```

### Custom Hook Testing

```jsx
// __tests__/usePRSMAdvanced.test.js
import { renderHook, act } from '@testing-library/react';
import { usePRSMAdvanced } from '../hooks/usePRSMAdvanced';

// Mock the context
const mockClient = {
  query: jest.fn()
};

jest.mock('@prsm/react-sdk', () => ({
  usePRSMContext: () => ({ client: mockClient })
}));

describe('usePRSMAdvanced Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('executes query and updates history', async () => {
    const mockResponse = {
      answer: 'Test response',
      confidence: 0.9,
      usage: { tokens: 50 }
    };
    mockClient.query.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => usePRSMAdvanced());

    await act(async () => {
      await result.current.query('Test prompt', { userId: 'test' });
    });

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].prompt).toBe('Test prompt');
    expect(result.current.history[0].response).toBe('Test response');
  });

  test('handles errors correctly', async () => {
    mockClient.query.mockRejectedValue(new Error('API Error'));

    const { result } = renderHook(() => usePRSMAdvanced());

    await act(async () => {
      try {
        await result.current.query('Test prompt');
      } catch (error) {
        // Expected to throw
      }
    });

    expect(result.current.error).toBe('API Error');
  });
});
```

## 🚀 Performance Optimization

### Memoization and Optimization

```jsx
// components/OptimizedChat.js
import React, { memo, useMemo, useCallback } from 'react';
import { usePRSM } from '@prsm/react-sdk';

const Message = memo(({ message, onEdit, onDelete }) => (
  <div className={`message ${message.role}`}>
    {message.content}
    {message.role === 'user' && (
      <div className="message-actions">
        <button onClick={() => onEdit(message.id)}>Edit</button>
        <button onClick={() => onDelete(message.id)}>Delete</button>
      </div>
    )}
  </div>
));

const OptimizedChat = memo(({ userId, maxMessages = 50 }) => {
  const [messages, setMessages] = useState([]);
  const { query, loading } = usePRSM();

  const displayMessages = useMemo(() => 
    messages.slice(-maxMessages),
    [messages, maxMessages]
  );

  const handleSendMessage = useCallback(async (content) => {
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await query(content, { userId });
      const aiMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.answer,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  }, [query, userId]);

  const handleEditMessage = useCallback((messageId) => {
    // Edit logic here
  }, []);

  const handleDeleteMessage = useCallback((messageId) => {
    setMessages(prev => prev.filter(msg => msg.id !== messageId));
  }, []);

  return (
    <div className="optimized-chat">
      <div className="messages">
        {displayMessages.map(message => (
          <Message
            key={message.id}
            message={message}
            onEdit={handleEditMessage}
            onDelete={handleDeleteMessage}
          />
        ))}
      </div>
      {/* Input form */}
    </div>
  );
});

export default OptimizedChat;
```

## 📋 Best Practices

### Error Boundaries

```jsx
// components/PRSMErrorBoundary.js
import React from 'react';

class PRSMErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('PRSM Error Boundary:', error, errorInfo);
    
    // Log to your error reporting service
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-fallback">
          <h2>Something went wrong with the AI chat.</h2>
          <p>Please refresh the page and try again.</p>
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default PRSMErrorBoundary;
```

### Security Considerations

```jsx
// utils/security.js
export const sanitizeInput = (input) => {
  // Remove potential XSS vectors
  return input
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+="/gi, '');
};

export const validatePrompt = (prompt) => {
  if (!prompt || typeof prompt !== 'string') {
    throw new Error('Invalid prompt');
  }
  
  if (prompt.length > 10000) {
    throw new Error('Prompt too long');
  }
  
  return sanitizeInput(prompt);
};

// Usage in components
const handleSubmit = async (rawPrompt) => {
  try {
    const cleanPrompt = validatePrompt(rawPrompt);
    const response = await query(cleanPrompt, { userId });
    // Process response
  } catch (error) {
    setError(error.message);
  }
};
```

## 📦 Production Build

### Environment Configuration

```javascript
// config/production.js
const config = {
  prsm: {
    baseURL: process.env.REACT_APP_PRSM_URL,
    apiKey: process.env.REACT_APP_PRSM_API_KEY,
    timeout: 30000,
    retryAttempts: 3
  },
  features: {
    analytics: true,
    errorReporting: true,
    performance: true
  }
};

export default config;
```

### Build Optimization

```json
{
  "scripts": {
    "build": "react-scripts build",
    "build:analyze": "npm run build && npx bundle-analyzer build/static/js/*.js",
    "build:production": "REACT_APP_ENV=production npm run build"
  },
  "dependencies": {
    "@prsm/react-sdk": "^1.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  }
}
```

---

**Need help with React integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).