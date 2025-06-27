// PRSM React Integration Example
// Demonstrates using PRSM SDK in React applications

import React, { useState, useEffect } from 'react';
import PRSM from '../index.js';

const PRSMChat = () => {
    const [client, setClient] = useState(null);
    const [message, setMessage] = useState('');
    const [conversation, setConversation] = useState([]);
    const [loading, setLoading] = useState(false);
    const [balance, setBalance] = useState(0);

    useEffect(() => {
        // Initialize PRSM client
        const prsmClient = new PRSM.Client({
            apiKey: process.env.REACT_APP_PRSM_API_KEY || 'demo-key',
            endpoint: process.env.REACT_APP_PRSM_ENDPOINT || 'http://localhost:8000'
        });
        
        setClient(prsmClient);
        
        // Load initial balance
        prsmClient.marketplace.getBalance()
            .then(bal => setBalance(bal.ftns))
            .catch(console.error);
    }, []);

    const sendMessage = async () => {
        if (!client || !message.trim()) return;
        
        setLoading(true);
        const userMessage = { role: 'user', content: message };
        setConversation(prev => [...prev, userMessage]);
        
        try {
            const response = await client.query({
                prompt: message,
                maxTokens: 150,
                temperature: 0.7
            });
            
            const assistantMessage = {
                role: 'assistant',
                content: response.text,
                cost: response.cost
            };
            
            setConversation(prev => [...prev, assistantMessage]);
            setBalance(prev => prev - (response.ftnsSpent || 0));
            
        } catch (error) {
            console.error('PRSM query error:', error);
            const errorMessage = {
                role: 'assistant',
                content: `Error: ${error.message}`,
                isError: true
            };
            setConversation(prev => [...prev, errorMessage]);
        }
        
        setMessage('');
        setLoading(false);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="prsm-chat">
            <div className="chat-header">
                <h3>🤖 PRSM AI Chat</h3>
                <div className="balance">💰 {balance.toFixed(2)} FTNS</div>
            </div>
            
            <div className="chat-conversation">
                {conversation.map((msg, index) => (
                    <div 
                        key={index} 
                        className={`message ${msg.role} ${msg.isError ? 'error' : ''}`}
                    >
                        <div className="message-content">{msg.content}</div>
                        {msg.cost && (
                            <div className="message-cost">Cost: {msg.cost} FTNS</div>
                        )}
                    </div>
                ))}
                {loading && (
                    <div className="message assistant loading">
                        <div className="typing-indicator">🤖 Thinking...</div>
                    </div>
                )}
            </div>
            
            <div className="chat-input">
                <textarea
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask PRSM anything about distributed AI..."
                    disabled={loading}
                />
                <button 
                    onClick={sendMessage}
                    disabled={loading || !message.trim()}
                >
                    Send
                </button>
            </div>
            
            <style jsx>{`
                .prsm-chat {
                    max-width: 600px;
                    margin: 0 auto;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    overflow: hidden;
                }
                
                .chat-header {
                    background: #f5f5f5;
                    padding: 1rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                
                .chat-conversation {
                    height: 400px;
                    overflow-y: auto;
                    padding: 1rem;
                }
                
                .message {
                    margin-bottom: 1rem;
                    padding: 0.5rem;
                    border-radius: 4px;
                }
                
                .message.user {
                    background: #e3f2fd;
                    margin-left: 2rem;
                }
                
                .message.assistant {
                    background: #f5f5f5;
                    margin-right: 2rem;
                }
                
                .message.error {
                    background: #ffebee;
                    color: #c62828;
                }
                
                .message-cost {
                    font-size: 0.8em;
                    color: #666;
                    margin-top: 0.25rem;
                }
                
                .chat-input {
                    padding: 1rem;
                    display: flex;
                    gap: 0.5rem;
                }
                
                .chat-input textarea {
                    flex: 1;
                    padding: 0.5rem;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    resize: vertical;
                    min-height: 60px;
                }
                
                .chat-input button {
                    padding: 0.5rem 1rem;
                    background: #1976d2;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                
                .chat-input button:disabled {
                    background: #ccc;
                    cursor: not-allowed;
                }
                
                .typing-indicator {
                    font-style: italic;
                    color: #666;
                }
            `}</style>
        </div>
    );
};

export default PRSMChat;