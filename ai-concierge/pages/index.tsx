import React, { useState, useRef, useEffect, createContext, useContext } from 'react';
import Head from 'next/head';
import Image from 'next/image';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    sources?: string[];
    confidence?: 'high' | 'medium' | 'low';
    escalationSuggested?: boolean;
    responseTime?: number;
  };
}

interface ChatResponse {
  content: string;
  sourceReferences: string[];
  confidence: 'high' | 'medium' | 'low';
  escalationSuggested: boolean;
  responseMetadata: {
    provider: string;
    model: string;
    responseTime: number;
    tokensUsed: number;
  };
}

// Theme Context
const ThemeContext = createContext<{
  isDark: boolean;
  toggleTheme: () => void;
}>({ isDark: false, toggleTheme: () => {} });

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDark, setIsDark] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const toggleTheme = () => {
    setIsDark(!isDark);
    localStorage.setItem('theme', !isDark ? 'dark' : 'light');
  };

  // Load theme preference on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      setIsDark(true);
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Add welcome message
    setMessages([{
      id: 'welcome',
      role: 'assistant',
      content: `Welcome! I'm PRSM's AI Investor Relations Concierge. I have comprehensive knowledge of PRSM's investment opportunity, technical architecture, team capabilities, and strategic vision.

I can help you with:
• **Investment details** - funding strategy, valuation, financial projections
• **Technical architecture** - security, scalability, performance metrics  
• **Business model** - revenue streams, tokenomics, market positioning
• **Team & execution** - development progress, validation evidence
• **Strategic partnerships** - Apple collaboration, growth opportunities

For deeper technical due diligence, I can guide you to PRSM's complete open-source repository at **https://github.com/Ryno2390/PRSM** which contains 400+ files, comprehensive documentation, and live demos.

How can I assist with your PRSM evaluation today?`,
      timestamp: new Date(),
      metadata: { confidence: 'high' }
    }]);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          conversationHistory: messages.slice(-6) // Last 3 exchanges
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: ChatResponse = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.content,
        timestamp: new Date(),
        metadata: {
          sources: data.sourceReferences,
          confidence: data.confidence,
          escalationSuggested: data.escalationSuggested,
          responseTime: data.responseMetadata.responseTime
        }
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'An error occurred');
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getConfidenceColor = (confidence?: string) => {
    const baseColors = {
      high: isDark ? 'text-green-400' : 'text-green-600',
      medium: isDark ? 'text-yellow-400' : 'text-yellow-600',
      low: isDark ? 'text-prsm-error' : 'text-prsm-light-error',
      default: isDark ? 'text-prsm-text-secondary' : 'text-prsm-light-text-secondary'
    };
    
    switch (confidence) {
      case 'high': return baseColors.high;
      case 'medium': return baseColors.medium;
      case 'low': return baseColors.low;
      default: return baseColors.default;
    }
  };

  const formatSources = (sources?: string[]) => {
    if (!sources || sources.length === 0) return null;
    
    return (
      <div className={`mt-3 p-3 rounded-lg border-l-4 border-blue-500 ${
        isDark ? 'bg-prsm-bg-tertiary' : 'bg-prsm-light-bg-tertiary'
      }`}>
        <div className={`text-sm font-medium mb-1 ${
          isDark ? 'text-prsm-text-primary' : 'text-prsm-light-text-primary'
        }`}>Sources:</div>
        <ul className={`text-xs space-y-1 ${
          isDark ? 'text-prsm-text-secondary' : 'text-prsm-light-text-secondary'
        }`}>
          {sources.map((source, index) => (
            <li key={index} className="flex items-start">
              <span className="text-blue-500 mr-1">•</span>
              <span>{source}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      <Head>
        <title>PRSM AI Investor Concierge</title>
        <meta name="description" content="24/7 Intelligent Investor Relations Assistant" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className={`min-h-screen transition-colors duration-300 ${
        isDark 
          ? 'bg-prsm-bg-primary' 
          : 'bg-prsm-light-bg-primary light-theme'
      }`}>
        {/* Header */}
        <header className={`shadow-sm border-b transition-colors duration-300 ${
          isDark ? 'bg-prsm-bg-secondary border-prsm-border' : 'bg-prsm-light-bg-primary border-prsm-light-border'
        }`}>
          <div className="max-w-4xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                {/* PRSM Logo */}
                <div className="relative w-12 h-12">
                  <Image
                    src={isDark ? '/PRSM_Logo_Dark.png' : '/PRSM_Logo_Light.png'}
                    alt="PRSM Logo"
                    width={48}
                    height={48}
                    className="object-contain"
                  />
                </div>
                <div>
                  <h1 className={`text-xl font-bold transition-colors duration-300 ${
                    isDark ? 'text-prsm-text-primary' : 'text-prsm-light-text-primary'
                  }`}>PRSM AI Investor Concierge</h1>
                  <p className={`text-sm transition-colors duration-300 ${
                    isDark ? 'text-prsm-text-secondary' : 'text-prsm-light-text-secondary'
                  }`}>24/7 Intelligent Investor Relations Assistant</p>
                </div>
              </div>
              
              {/* Theme Toggle */}
              <button
                onClick={toggleTheme}
                className={`p-2 rounded-lg transition-colors duration-300 hover:bg-opacity-20 ${
                  isDark 
                    ? 'text-prsm-text-secondary hover:bg-prsm-bg-tertiary hover:text-prsm-text-primary' 
                    : 'text-prsm-light-text-secondary hover:bg-prsm-light-bg-secondary hover:text-prsm-light-text-primary'
                }`}
                aria-label="Toggle theme"
              >
                {isDark ? (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </header>

        {/* Chat Container */}
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className={`rounded-lg shadow-lg h-[calc(100vh-200px)] flex flex-col transition-colors duration-300 ${
            isDark ? 'bg-prsm-bg-secondary' : 'bg-prsm-light-bg-primary'
          }`}>
            
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-3 transition-colors duration-300 ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : isDark ? 'bg-prsm-bg-tertiary text-prsm-text-primary' : 'bg-prsm-light-bg-secondary text-prsm-light-text-primary'
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    
                    {/* Metadata for assistant messages */}
                    {message.role === 'assistant' && message.metadata && (
                      <div className={`mt-3 pt-3 border-t transition-colors duration-300 ${
                        isDark ? 'border-prsm-border' : 'border-prsm-light-border'
                      }`}>
                        <div className={`flex items-center justify-between text-xs transition-colors duration-300 ${
                          isDark ? 'text-prsm-text-secondary' : 'text-prsm-light-text-secondary'
                        }`}>
                          <span className={`font-medium ${getConfidenceColor(message.metadata.confidence)}`}>
                            {message.metadata.confidence?.toUpperCase()} CONFIDENCE
                          </span>
                          {message.metadata.responseTime && (
                            <span>{message.metadata.responseTime}ms</span>
                          )}
                        </div>
                        
                        {message.metadata.escalationSuggested && (
                          <div className={`mt-2 p-2 border rounded text-xs transition-colors duration-300 ${
                            isDark 
                              ? 'bg-prsm-bg-tertiary border-yellow-600 text-yellow-400' 
                              : 'bg-yellow-50 border-yellow-200 text-yellow-800'
                          }`}>
                            <strong>Escalation Suggested:</strong> This query may benefit from direct consultation with our team.
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Sources */}
                    {message.role === 'assistant' && formatSources(message.metadata?.sources)}
                  </div>
                </div>
              ))}
              
              {/* Loading indicator */}
              {isLoading && (
                <div className="flex justify-start">
                  <div className={`rounded-lg px-4 py-3 max-w-[80%] transition-colors duration-300 ${
                    isDark ? 'bg-prsm-bg-tertiary' : 'bg-prsm-light-bg-secondary'
                  }`}>
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className={`w-2 h-2 rounded-full animate-bounce ${
                          isDark ? 'bg-prsm-text-secondary' : 'bg-prsm-light-text-secondary'
                        }`}></div>
                        <div className={`w-2 h-2 rounded-full animate-bounce ${
                          isDark ? 'bg-prsm-text-secondary' : 'bg-prsm-light-text-secondary'
                        }`} style={{ animationDelay: '0.1s' }}></div>
                        <div className={`w-2 h-2 rounded-full animate-bounce ${
                          isDark ? 'bg-prsm-text-secondary' : 'bg-prsm-light-text-secondary'
                        }`} style={{ animationDelay: '0.2s' }}></div>
                      </div>
                      <span className={`text-sm transition-colors duration-300 ${
                        isDark ? 'text-prsm-text-primary' : 'text-prsm-light-text-primary'
                      }`}>Analyzing your question...</span>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Error display */}
              {error && (
                <div className="flex justify-center">
                  <div className={`border rounded-lg px-4 py-3 max-w-[80%] transition-colors duration-300 ${
                    isDark 
                      ? 'bg-prsm-bg-tertiary border-prsm-error text-prsm-error' 
                      : 'bg-red-50 border-prsm-light-error text-prsm-light-error'
                  }`}>
                    <div className="text-sm">
                      <strong>Error:</strong> {error}
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>

            {/* Input Form */}
            <div className={`border-t p-4 transition-colors duration-300 ${
              isDark 
                ? 'bg-prsm-bg-tertiary border-prsm-border' 
                : 'bg-prsm-light-bg-secondary border-prsm-light-border'
            }`}>
              <form onSubmit={handleSubmit} className="flex space-x-4">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask me about PRSM's investment opportunity, technical architecture, or business strategy..."
                  className={`flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors duration-300 ${
                    isDark 
                      ? 'bg-prsm-bg-secondary border-prsm-border text-prsm-text-primary placeholder-prsm-text-secondary' 
                      : 'bg-prsm-light-bg-primary border-prsm-light-border text-prsm-light-text-primary placeholder-prsm-light-text-secondary'
                  }`}
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={isLoading || !inputValue.trim()}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Send
                </button>
              </form>
              
              <div className={`mt-2 text-xs text-center transition-colors duration-300 ${
                isDark ? 'text-prsm-text-secondary' : 'text-prsm-light-text-secondary'
              }`}>
                Powered by PRSM's AI coordination technology •
                <span className="ml-1 font-medium">Advanced Prototype</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </ThemeContext.Provider>
  );
}