# PRSM WebSocket API Documentation

## Overview

The PRSM WebSocket API provides real-time communication capabilities for live AI conversations, instant notifications, and dynamic data updates. This enables token-by-token streaming responses, live system updates, and collaborative features.

## Connection Endpoints

### Global WebSocket Connection
```
ws://localhost:8000/ws/{user_id}
```

**Purpose**: General real-time communication, notifications, and system updates.

**Parameters**:
- `user_id` (string): Unique identifier for the user session

### Conversation WebSocket Connection
```
ws://localhost:8000/ws/conversation/{user_id}/{conversation_id}
```

**Purpose**: Real-time conversation streaming with token-by-token AI responses.

**Parameters**:
- `user_id` (string): Unique identifier for the user session
- `conversation_id` (string): Unique identifier for the conversation

## Message Format

All WebSocket messages use JSON format:

```json
{
  "type": "message_type",
  "data": {
    // Message-specific data
  }
}
```

## Client-to-Server Messages

### Connection Messages

#### Initial Connection
```json
{
  "type": "connection",
  "data": {
    "user_id": "ui_user_1234567890",
    "client_type": "web_ui"
  }
}
```

#### Join Conversation
```json
{
  "type": "join_conversation",
  "data": {
    "conversation_id": "conv_abc123",
    "user_id": "ui_user_1234567890"
  }
}
```

### Message Sending

#### Send Message with Streaming
```json
{
  "type": "send_message",
  "data": {
    "content": "Hello PRSM! How can you help with my research?",
    "conversation_id": "conv_abc123",
    "streaming": true
  }
}
```

### Heartbeat

#### Keep Connection Alive
```json
{
  "type": "heartbeat",
  "data": {
    "timestamp": 1640995200000,
    "user_id": "ui_user_1234567890"
  }
}
```

## Server-to-Client Messages

### AI Response Streaming

#### Response Chunk
Sent for each token/chunk during AI response generation.

```json
{
  "type": "ai_response_chunk",
  "data": {
    "conversation_id": "conv_abc123",
    "message_id": "msg_def456",
    "chunk": "Hello! I'm here to help with your research. ",
    "is_complete": false
  }
}
```

#### Response Complete
Sent when AI response is fully generated.

```json
{
  "type": "ai_response_complete",
  "data": {
    "conversation_id": "conv_abc123",
    "message_id": "msg_def456",
    "complete_content": "Hello! I'm here to help with your research. What specific area would you like to explore?",
    "metadata": {
      "model_used": "nwtn-v1",
      "tokens_used": 23,
      "context_used": 1523,
      "context_limit": 4096,
      "total_cost": 0.0045
    }
  }
}
```

### Notifications

#### System Notification
```json
{
  "type": "notification",
  "data": {
    "title": "FTNS Balance Updated",
    "message": "Your balance has increased by 50 FTNS from staking rewards",
    "type": "info",
    "priority": "normal"
  }
}
```

### Typing Indicators

#### Typing Status
```json
{
  "type": "typing_indicator",
  "data": {
    "conversation_id": "conv_abc123",
    "is_typing": true,
    "user_id": "system"
  }
}
```

### Live Data Updates

#### Tokenomics Update
Real-time updates for FTNS balance, staking, and earnings.

```json
{
  "type": "tokenomics_update",
  "data": {
    "balance": {
      "total": 1550,
      "available": 1050,
      "locked": 500,
      "currency": "FTNS"
    },
    "staking": {
      "staked_amount": 500,
      "apy": 8.5,
      "rewards_earned": 42.5,
      "staking_period": "30_days"
    },
    "earnings": {
      "total_earned": 206.8,
      "sources": {
        "ipfs_hosting": 119.2,
        "model_hosting": 65.6,
        "compute_contribution": 22.0
      },
      "current_status": "active"
    },
    "metadata": {
      "significant_change": true
    }
  }
}
```

#### Task Update
Real-time updates for task management.

```json
{
  "type": "task_update",
  "data": {
    "task": {
      "task_id": "task_ghi789",
      "title": "Analyze protein folding simulation",
      "description": "Review and analyze the latest protein folding results",
      "status": "completed",
      "priority": "high"
    },
    "action": "completed",
    "metadata": {
      "completion_time": "2024-01-15T14:30:00Z"
    }
  }
}
```

#### File Update
Real-time updates for file operations.

```json
{
  "type": "file_update",
  "data": {
    "file": {
      "file_id": "file_jkl012",
      "filename": "research_data.csv",
      "content_type": "text/csv",
      "size": 1048576
    },
    "action": "uploaded",
    "metadata": {
      "processing_status": "complete"
    }
  }
}
```

### Error Handling

#### WebSocket Error
```json
{
  "type": "error",
  "data": {
    "error_message": "Failed to process message",
    "error_code": "PROCESSING_ERROR",
    "details": {
      "original_message": "send_message",
      "timestamp": "2024-01-15T14:30:00Z"
    }
  }
}
```

## Connection Management

### Connection States

WebSocket connections can be in the following states:
- `CONNECTING` (0): Connection is being established
- `OPEN` (1): Connection is active and ready for communication
- `CLOSING` (2): Connection is being closed
- `CLOSED` (3): Connection is closed

### Reconnection Logic

The client implements automatic reconnection with exponential backoff:

1. **Initial Delay**: 1 second
2. **Maximum Attempts**: 5
3. **Backoff Strategy**: Exponential (1s, 2s, 4s, 8s, 16s)
4. **Maximum Delay**: 30 seconds

### Health Monitoring

The system includes several health monitoring features:

#### Heartbeat
- Sent every 60 seconds to maintain connection
- Detects network issues and connection drops

#### Health Checks
- Performed every 30 seconds
- Automatically reconnects if connection is lost

#### Connection Status
The client provides detailed connection status:

```javascript
const status = prsmAPI.getConnectionStatus();
console.log(status);
// {
//   api_connected: true,
//   global_ws_status: 1, // WebSocket.OPEN
//   conversation_ws_status: 1,
//   current_conversation: "conv_abc123",
//   reconnect_attempts: 0,
//   user_id: "ui_user_1234567890"
// }
```

## Client Implementation

### JavaScript API Client

The PRSM JavaScript API client provides easy WebSocket integration:

```javascript
// Initialize the API client (automatically connects WebSockets)
const prsmAPI = new PRSMAPIClient('http://localhost:8000');

// Register event handlers
prsmAPI.onWebSocketMessage('ai_response_chunk', (data) => {
  console.log('Received chunk:', data.chunk);
  // Update UI with streaming content
});

prsmAPI.onWebSocketMessage('notification', (data) => {
  console.log('Notification:', data.title, data.message);
  // Show notification to user
});

// Send a message with streaming response
const response = await prsmAPI.sendMessageWithStreaming(
  "How can machine learning help with drug discovery?"
);

if (response.streaming) {
  console.log('Message sent via WebSocket - expecting streaming response');
}
```

### Event Handler Registration

Register custom handlers for specific message types:

```javascript
// Custom handler for tokenomics updates
prsmAPI.onWebSocketMessage('tokenomics_update', (data) => {
  updateBalanceDisplay(data.balance.total);
  updateStakingInfo(data.staking);
  updateEarnings(data.earnings);
});

// Custom handler for task updates
prsmAPI.onWebSocketMessage('task_update', (data) => {
  if (data.action === 'completed') {
    showSuccessNotification(`Task completed: ${data.task.title}`);
  }
  refreshTaskList();
});
```

## Security Considerations

### Authentication
- WebSocket connections inherit authentication from the HTTP session
- User ID validation ensures connections are properly authorized

### Message Validation
- All incoming messages are validated against expected schemas
- Invalid messages are logged and ignored

### Rate Limiting
- Connections are subject to rate limiting to prevent abuse
- Excessive message sending results in temporary connection throttling

### Content Security
- All message content is sanitized before processing
- No executable code is accepted via WebSocket messages

## Testing

### Test Interface

A comprehensive test interface is available at `/test_websocket.html`:

```bash
# Start the UI server
cd PRSM_ui_mockup
python -m http.server 8080

# Open the WebSocket test interface
open http://localhost:8080/test_websocket.html
```

### Test Scenarios

The test interface includes:
- **Connection Testing**: Verify WebSocket establishment
- **Message Streaming**: Test real-time AI response streaming
- **Notification Testing**: Validate notification delivery
- **Connection Recovery**: Test automatic reconnection
- **Event Logging**: Monitor all WebSocket events

### Manual Testing

```javascript
// Test connection status
console.log(prsmAPI.getConnectionStatus());

// Test heartbeat
prsmAPI.sendHeartbeat();

// Test notification system
prsmAPI.showNotification('Test', 'This is a test notification', 'info');

// Test WebSocket health check
prsmAPI.checkWebSocketHealth();
```

## Troubleshooting

### Common Issues

#### Connection Failures
- **Symptom**: WebSocket connections fail to establish
- **Causes**: API server not running, firewall blocking WebSocket connections
- **Solution**: Verify API server is running on port 8000, check firewall settings

#### Reconnection Loops
- **Symptom**: Constant reconnection attempts
- **Causes**: Network instability, server overload
- **Solution**: Check network connection, verify server health

#### Missing Messages
- **Symptom**: Some WebSocket messages not received
- **Causes**: Connection drops, message handler errors
- **Solution**: Check connection status, verify event handlers

### Debug Logging

Enable detailed WebSocket logging:

```javascript
// Enable debug mode
localStorage.setItem('prsm_debug', 'true');

// View WebSocket events in console
prsmAPI.onWebSocketMessage('*', (type, data) => {
  console.log(`WebSocket ${type}:`, data);
});
```

### Error Recovery

The system includes automatic error recovery:
- **Connection Drops**: Automatic reconnection with exponential backoff
- **Message Failures**: Fallback to REST API for critical operations
- **Server Errors**: Graceful degradation to mock mode

## Performance Considerations

### Message Throughput
- The WebSocket system can handle hundreds of messages per second
- Token streaming is optimized for real-time display with minimal latency

### Memory Management
- Event handlers are automatically cleaned up on connection close
- Message history is limited to prevent memory leaks

### Network Efficiency
- Messages are compressed using built-in WebSocket compression
- Heartbeat messages are minimal to reduce bandwidth usage

## Future Enhancements

Planned improvements for the WebSocket API:

1. **Message Persistence**: Store and replay missed messages
2. **Multi-User Conversations**: Support for collaborative chat sessions
3. **Voice Integration**: Real-time audio streaming for voice conversations
4. **File Streaming**: Progressive file upload with real-time progress
5. **Advanced Notifications**: Rich media notifications with actions

---

For more information, see the [PRSM API Documentation](api/) and [Web Interface Guide](../PRSM_ui_mockup/README.md).