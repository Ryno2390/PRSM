// PRSM JavaScript SDK - Next.js Chat Interface Component
// This component demonstrates how to build a modern chat interface
// using PRSM SDK with React hooks and real-time streaming.

import React, { useState, useRef, useEffect } from 'react';
import { PRSMClient } from '@prsm/sdk';

// Initialize PRSM client for client-side usage
const prsm = new PRSMClient({
  // Note: For client-side usage, implement proper authentication
  // This example uses API routes for secure server-side calls
  baseURL: '/api' // Use Next.js API routes as proxy
});

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Hello! I\'m powered by PRSM. How can I help you today?',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [settings, setSettings] = useState({
    temperature: 0.7,
    maxTokens: 500,
    stream: true
  });
  const [error, setError] = useState(null);
  const [usageStats, setUsageStats] = useState({
    totalTokens: 0,
    totalCost: 0
  });

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setError(null);
    setIsLoading(true);

    // Create abort controller for cancellation
    abortControllerRef.current = new AbortController();

    try {
      if (settings.stream) {
        await handleStreamingResponse(userMessage.content);
      } else {
        await handleRegularResponse(userMessage.content);
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('Chat error:', error);
        setError(error.message || 'An error occurred while processing your message');
      }
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      setStreamingContent('');
    }
  };

  // Handle streaming responses
  const handleStreamingResponse = async (message) => {
    setIsStreaming(true);
    setStreamingContent('');

    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        model: selectedModel,
        stream: true,
        options: {
          temperature: settings.temperature,
          maxTokens: settings.maxTokens
        }
      }),
      signal: abortControllerRef.current.signal
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let assistantMessage = {
      id: Date.now() + 1,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      model: selectedModel,
      streaming: true
    };

    // Add placeholder message
    setMessages(prev => [...prev, assistantMessage]);

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            
            if (data === '[DONE]') {
              // Finalize the message
              setMessages(prev => prev.map(msg => 
                msg.id === assistantMessage.id 
                  ? { ...msg, streaming: false, content: streamingContent }
                  : msg
              ));
              return;
            }

            try {
              const parsed = JSON.parse(data);
              
              if (parsed.error) {
                throw new Error(parsed.message || 'Streaming error');
              }

              if (parsed.done) {
                // Update final message and usage stats
                if (parsed.usage) {
                  setUsageStats(prev => ({
                    totalTokens: prev.totalTokens + parsed.usage.totalTokens,
                    totalCost: prev.totalCost + parsed.cost
                  }));
                }
                
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessage.id 
                    ? { ...msg, streaming: false, content: streamingContent }
                    : msg
                ));
                return;
              }

              if (parsed.content) {
                setStreamingContent(prev => prev + parsed.content);
                assistantMessage.content += parsed.content;
              }

            } catch (parseError) {
              console.warn('Failed to parse streaming data:', data);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  };

  // Handle regular (non-streaming) responses
  const handleRegularResponse = async (message) => {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        model: selectedModel,
        stream: false,
        options: {
          temperature: settings.temperature,
          maxTokens: settings.maxTokens
        }
      }),
      signal: abortControllerRef.current.signal
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || `HTTP ${response.status}`);
    }

    const result = await response.json();

    const assistantMessage = {
      id: Date.now() + 1,
      role: 'assistant',
      content: result.content,
      timestamp: new Date(),
      model: result.model,
      usage: result.usage,
      cost: result.cost
    };

    setMessages(prev => [...prev, assistantMessage]);

    // Update usage stats
    setUsageStats(prev => ({
      totalTokens: prev.totalTokens + result.usage.totalTokens,
      totalCost: prev.totalCost + result.cost
    }));
  };

  // Cancel ongoing request
  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  // Clear chat history
  const handleClear = () => {
    setMessages([{
      id: 1,
      role: 'assistant',
      content: 'Chat cleared. How can I help you?',
      timestamp: new Date()
    }]);
    setUsageStats({ totalTokens: 0, totalCost: 0 });
    setError(null);
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto bg-white">
      {/* Header */}
      <div className="bg-blue-600 text-white p-4 shadow-lg">
        <div className="flex justify-between items-center">
          <h1 className="text-xl font-bold">PRSM Chat Interface</h1>
          <div className="flex items-center space-x-4 text-sm">
            <span>Tokens: {usageStats.totalTokens}</span>
            <span>Cost: ${usageStats.totalCost.toFixed(4)}</span>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      <div className="bg-gray-100 p-4 border-b">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="border rounded px-3 py-1 text-sm"
              disabled={isLoading}
            >
              <option value="gpt-4">GPT-4 (Best Quality)</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Fast)</option>
              <option value="claude-3">Claude 3 (Reasoning)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Temperature: {settings.temperature}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={settings.temperature}
              onChange={(e) => setSettings(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
              className="w-20"
              disabled={isLoading}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Tokens
            </label>
            <input
              type="number"
              min="50"
              max="2000"
              value={settings.maxTokens}
              onChange={(e) => setSettings(prev => ({ ...prev, maxTokens: parseInt(e.target.value) }))}
              className="border rounded px-2 py-1 text-sm w-20"
              disabled={isLoading}
            />
          </div>

          <div>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={settings.stream}
                onChange={(e) => setSettings(prev => ({ ...prev, stream: e.target.checked }))}
                disabled={isLoading}
              />
              <span className="text-sm">Stream responses</span>
            </label>
          </div>

          <button
            onClick={handleClear}
            className="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600"
            disabled={isLoading}
          >
            Clear Chat
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 mx-4 mt-4 rounded">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 text-gray-900'
              }`}
            >
              <div className="whitespace-pre-wrap">{message.content}</div>
              {message.streaming && (
                <div className="text-xs opacity-75 mt-1">
                  {streamingContent && <div>{streamingContent}</div>}
                  <div className="animate-pulse">Typing...</div>
                </div>
              )}
              <div className="text-xs opacity-75 mt-1">
                {message.timestamp.toLocaleTimeString()}
                {message.model && ` • ${message.model}`}
                {message.cost && ` • $${message.cost.toFixed(4)}`}
              </div>
            </div>
          </div>
        ))}
        
        {/* Streaming content display */}
        {isStreaming && streamingContent && (
          <div className="flex justify-start">
            <div className="max-w-xs lg:max-w-md px-4 py-2 rounded-lg bg-gray-200 text-gray-900">
              <div className="whitespace-pre-wrap">{streamingContent}</div>
              <div className="text-xs opacity-75 mt-1 animate-pulse">
                Streaming...
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="border-t p-4">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
            maxLength={10000}
          />
          
          {isLoading ? (
            <button
              type="button"
              onClick={handleCancel}
              className="bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-600"
            >
              Cancel
            </button>
          ) : (
            <button
              type="submit"
              disabled={!inputMessage.trim()}
              className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          )}
        </form>
        
        <div className="text-xs text-gray-500 mt-2">
          {inputMessage.length}/10000 characters
          {settings.stream && " • Streaming enabled"}
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;