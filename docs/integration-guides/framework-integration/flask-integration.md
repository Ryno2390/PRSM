# Flask Integration Guide

Integrate PRSM into Flask applications for AI-powered web development with Python.

## 🎯 Overview

This guide covers integrating PRSM into Flask applications, including routing, templates, error handling, and production deployment patterns.

## 📋 Prerequisites

- Python 3.8+
- Flask 2.0+
- PRSM instance running
- Basic Flask knowledge

## 🚀 Quick Start

### 1. Installation

```bash
pip install flask prsm-sdk
```

### 2. Basic Flask App

```python
# app.py
from flask import Flask, request, jsonify, render_template
from prsm_sdk import PRSMClient
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret')

# Initialize PRSM client
prsm = PRSMClient(
    base_url=os.environ.get('PRSM_URL', 'http://localhost:8000'),
    api_key=os.environ.get('PRSM_API_KEY'),
    timeout=30
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id', 'anonymous')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = prsm.query(
            prompt=user_message,
            user_id=user_id,
            context={'endpoint': '/api/chat'}
        )
        
        return jsonify({
            'response': response.answer,
            'confidence': response.confidence,
            'usage': response.usage
        })
        
    except Exception as e:
        app.logger.error(f'Chat error: {str(e)}')
        return jsonify({'error': 'AI service temporarily unavailable'}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### 3. HTML Template

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>PRSM Flask Demo</title>
    <style>
        .chat-container { max-width: 600px; margin: 50px auto; }
        .messages { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; }
        .message { margin: 10px 0; }
        .user { text-align: right; color: blue; }
        .ai { text-align: left; color: green; }
        .input-container { margin-top: 10px; }
        input[type="text"] { width: 80%; padding: 8px; }
        button { padding: 8px 16px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>PRSM AI Chat</h1>
        <div id="messages" class="messages"></div>
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type your message...">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        function addMessage(message, isUser = false) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'ai');
            messageDiv.textContent = (isUser ? 'You: ' : 'AI: ') + message;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, true);
            input.value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        user_id: 'demo-user'
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    addMessage(data.response);
                } else {
                    addMessage('Error: ' + data.error);
                }
            } catch (error) {
                addMessage('Error: Failed to send message');
            }
        }

        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
```

## 🏗️ Advanced Flask Integration

### Application Factory Pattern

```python
# app/__init__.py
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from logging.handlers import RotatingFileHandler
import os

def create_app(config_name='development'):
    app = Flask(__name__)
    
    # Configuration
    app.config.from_object(f'config.{config_name.title()}Config')
    
    # Extensions
    CORS(app)
    
    # Rate limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    
    # Register blueprints
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    # Error handlers
    from app.errors import bp as errors_bp
    app.register_blueprint(errors_bp)
    
    # Logging
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        file_handler = RotatingFileHandler('logs/prsm_app.log',
                                         maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('PRSM Flask app startup')
    
    return app
```

### Configuration Management

```python
# config.py
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    
    # PRSM Configuration
    PRSM_URL = os.environ.get('PRSM_URL') or 'http://localhost:8000'
    PRSM_API_KEY = os.environ.get('PRSM_API_KEY')
    PRSM_TIMEOUT = int(os.environ.get('PRSM_TIMEOUT', '30'))
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite://'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
```

### PRSM Service Layer

```python
# app/services/prsm_service.py
from prsm_sdk import PRSMClient, PRSMError
from flask import current_app, g
import time
import logging

class PRSMService:
    def __init__(self):
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize PRSM client with current app config"""
        self.client = PRSMClient(
            base_url=current_app.config['PRSM_URL'],
            api_key=current_app.config['PRSM_API_KEY'],
            timeout=current_app.config['PRSM_TIMEOUT']
        )
    
    def query(self, prompt, user_id, context=None, **kwargs):
        """Execute PRSM query with error handling and logging"""
        start_time = time.time()
        
        try:
            current_app.logger.info(f'PRSM query started for user {user_id}')
            
            response = self.client.query(
                prompt=prompt,
                user_id=user_id,
                context=context or {},
                **kwargs
            )
            
            duration = time.time() - start_time
            current_app.logger.info(
                f'PRSM query completed in {duration:.2f}s for user {user_id}'
            )
            
            return response
            
        except PRSMError as e:
            duration = time.time() - start_time
            current_app.logger.error(
                f'PRSM query failed after {duration:.2f}s for user {user_id}: {str(e)}'
            )
            raise
        except Exception as e:
            duration = time.time() - start_time
            current_app.logger.error(
                f'Unexpected error after {duration:.2f}s for user {user_id}: {str(e)}'
            )
            raise PRSMError(f'Unexpected error: {str(e)}')
    
    def health_check(self):
        """Check PRSM service health"""
        try:
            return self.client.health()
        except Exception as e:
            current_app.logger.error(f'PRSM health check failed: {str(e)}')
            return {'status': 'unhealthy', 'error': str(e)}

def get_prsm_service():
    """Get PRSM service instance"""
    if 'prsm_service' not in g:
        g.prsm_service = PRSMService()
    return g.prsm_service
```

### API Blueprint

```python
# app/api/__init__.py
from flask import Blueprint

bp = Blueprint('api', __name__)

from app.api import routes, errors
```

```python
# app/api/routes.py
from flask import request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from app.api import bp
from app.services.prsm_service import get_prsm_service
from prsm_sdk import PRSMError
import uuid

# Rate limiting for API endpoints
limiter = Limiter(key_func=get_remote_address)

@bp.route('/chat', methods=['POST'])
@limiter.limit("30 per minute")
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        
        # Validation
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        if len(message) > 1000:
            return jsonify({'error': 'Message too long (max 1000 characters)'}), 400
        
        user_id = data.get('user_id', str(uuid.uuid4()))
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        # Get PRSM service
        prsm_service = get_prsm_service()
        
        # Execute query
        response = prsm_service.query(
            prompt=message,
            user_id=user_id,
            context={
                'session_id': session_id,
                'endpoint': '/api/chat',
                'user_agent': request.headers.get('User-Agent', ''),
                'ip_address': request.remote_addr
            }
        )
        
        return jsonify({
            'response': response.answer,
            'confidence': response.confidence,
            'session_id': session_id,
            'usage': {
                'tokens': response.usage.tokens if response.usage else 0,
                'cost': response.usage.cost if response.usage else 0
            }
        })
        
    except PRSMError as e:
        current_app.logger.error(f'PRSM error in chat: {str(e)}')
        return jsonify({
            'error': 'AI service temporarily unavailable',
            'code': 'PRSM_ERROR'
        }), 502
        
    except Exception as e:
        current_app.logger.error(f'Unexpected error in chat: {str(e)}')
        return jsonify({
            'error': 'Internal server error',
            'code': 'INTERNAL_ERROR'
        }), 500

@bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        prsm_service = get_prsm_service()
        prsm_health = prsm_service.health_check()
        
        return jsonify({
            'status': 'healthy',
            'prsm': prsm_health,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 503

@bp.route('/analytics', methods=['POST'])
@limiter.limit("100 per hour")
def analytics():
    """Submit analytics data"""
    try:
        data = request.get_json()
        
        # Log analytics data
        current_app.logger.info(f'Analytics: {data}')
        
        return jsonify({'status': 'received'})
        
    except Exception as e:
        current_app.logger.error(f'Analytics error: {str(e)}')
        return jsonify({'error': 'Failed to process analytics'}), 500
```

### Error Handling

```python
# app/errors/__init__.py
from flask import Blueprint

bp = Blueprint('errors', __name__)

from app.errors import handlers
```

```python
# app/errors/handlers.py
from flask import jsonify, request, current_app
from app.errors import bp
from prsm_sdk import PRSMError

@bp.app_errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('errors/404.html'), 404

@bp.app_errorhandler(500)
def internal_error(error):
    current_app.logger.error(f'Server Error: {error}')
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('errors/500.html'), 500

@bp.app_errorhandler(PRSMError)
def handle_prsm_error(error):
    current_app.logger.error(f'PRSM Error: {error}')
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'AI service error',
            'message': str(error)
        }), 502
    return render_template('errors/ai_error.html', error=error), 502

@bp.app_errorhandler(429)
def ratelimit_handler(e):
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': str(e.description)
        }), 429
    return render_template('errors/ratelimit.html'), 429
```

## 🔄 Streaming Integration

### Server-Sent Events

```python
# app/api/streaming.py
from flask import Response, request, stream_template
import json
import time

@bp.route('/stream', methods=['POST'])
@limiter.limit("10 per minute")
def stream_chat():
    """Stream AI responses using Server-Sent Events"""
    
    def generate():
        try:
            data = request.get_json()
            message = data.get('message', '')
            user_id = data.get('user_id', 'anonymous')
            
            prsm_service = get_prsm_service()
            
            # Start streaming
            yield f"data: {json.dumps({'type': 'start', 'message': 'Processing your request...'})}\n\n"
            
            # Simulate streaming response (replace with actual PRSM streaming)
            response = prsm_service.query(
                prompt=message,
                user_id=user_id,
                context={'streaming': True}
            )
            
            # Stream the response in chunks
            answer = response.answer
            for i in range(0, len(answer), 10):
                chunk = answer[i:i+10]
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                time.sleep(0.1)  # Simulate streaming delay
            
            yield f"data: {json.dumps({'type': 'complete', 'usage': response.usage.__dict__ if response.usage else {}})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )
```

### WebSocket Integration

```python
# app/websocket.py
from flask_socketio import SocketIO, emit, disconnect
from flask import request
from app.services.prsm_service import get_prsm_service
import logging

socketio = SocketIO(cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    logging.info(f'Client connected: {request.sid}')
    emit('status', {'message': 'Connected to PRSM AI'})

@socketio.on('disconnect')
def handle_disconnect():
    logging.info(f'Client disconnected: {request.sid}')

@socketio.on('chat_message')
def handle_chat_message(data):
    try:
        message = data.get('message', '')
        user_id = data.get('user_id', request.sid)
        
        if not message:
            emit('error', {'message': 'Message is required'})
            return
        
        # Emit thinking status
        emit('thinking', {'message': 'AI is thinking...'})
        
        # Get response from PRSM
        prsm_service = get_prsm_service()
        response = prsm_service.query(
            prompt=message,
            user_id=user_id,
            context={
                'websocket': True,
                'session_id': request.sid
            }
        )
        
        # Emit response
        emit('ai_response', {
            'message': response.answer,
            'confidence': response.confidence,
            'usage': response.usage.__dict__ if response.usage else {}
        })
        
    except Exception as e:
        logging.error(f'WebSocket error: {str(e)}')
        emit('error', {'message': 'AI service temporarily unavailable'})

@socketio.on_error_default
def default_error_handler(e):
    logging.error(f'WebSocket error: {str(e)}')
    emit('error', {'message': 'Connection error occurred'})
```

## 🗄️ Database Integration

### SQLAlchemy Models

```python
# app/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }

class Conversation(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'message_count': len(self.messages)
        }

class Message(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversation.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_from_user = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float)
    tokens_used = db.Column(db.Integer)
    cost = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'content': self.content,
            'is_from_user': self.is_from_user,
            'confidence': self.confidence,
            'tokens_used': self.tokens_used,
            'cost': self.cost,
            'created_at': self.created_at.isoformat()
        }
```

### Database Service

```python
# app/services/database_service.py
from app.models import db, User, Conversation, Message
from flask import current_app
import uuid

class DatabaseService:
    
    @staticmethod
    def create_user(username, email):
        """Create a new user"""
        user = User(username=username, email=email)
        db.session.add(user)
        db.session.commit()
        return user
    
    @staticmethod
    def get_or_create_user(user_id, username=None, email=None):
        """Get existing user or create new one"""
        user = User.query.get(user_id)
        if not user and username and email:
            user = DatabaseService.create_user(username, email)
        return user
    
    @staticmethod
    def create_conversation(user_id, title=None):
        """Create a new conversation"""
        conversation = Conversation(
            user_id=user_id,
            title=title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        )
        db.session.add(conversation)
        db.session.commit()
        return conversation
    
    @staticmethod
    def add_message(conversation_id, content, is_from_user, confidence=None, tokens_used=None, cost=None):
        """Add a message to a conversation"""
        message = Message(
            conversation_id=conversation_id,
            content=content,
            is_from_user=is_from_user,
            confidence=confidence,
            tokens_used=tokens_used,
            cost=cost
        )
        db.session.add(message)
        db.session.commit()
        return message
    
    @staticmethod
    def get_conversation_history(conversation_id, limit=50):
        """Get conversation history"""
        return Message.query.filter_by(conversation_id=conversation_id)\
                          .order_by(Message.created_at.desc())\
                          .limit(limit).all()
    
    @staticmethod
    def get_user_conversations(user_id, limit=20):
        """Get user's conversations"""
        return Conversation.query.filter_by(user_id=user_id)\
                                .order_by(Conversation.updated_at.desc())\
                                .limit(limit).all()
```

## 📊 Monitoring & Analytics

### Application Metrics

```python
# app/monitoring.py
from prometheus_flask_exporter import PrometheusMetrics
from flask import request, g
import time

def init_metrics(app):
    """Initialize Prometheus metrics"""
    metrics = PrometheusMetrics(app)
    
    # Custom metrics
    prsm_request_duration = metrics.histogram(
        'prsm_request_duration_seconds',
        'Time spent on PRSM requests',
        labels={'endpoint': lambda: request.endpoint, 'method': lambda: request.method}
    )
    
    prsm_request_count = metrics.counter(
        'prsm_requests_total',
        'Total PRSM requests',
        labels={'endpoint': lambda: request.endpoint, 'status': lambda: g.get('prsm_status', 'unknown')}
    )
    
    @app.before_request
    def before_request():
        g.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            prsm_request_duration.observe(duration)
        return response
    
    return metrics
```

### Custom Analytics

```python
# app/analytics.py
from collections import defaultdict
from datetime import datetime, timedelta
import json
import redis

class AnalyticsService:
    def __init__(self, redis_url=None):
        self.redis_client = redis.from_url(redis_url) if redis_url else None
    
    def track_query(self, user_id, prompt_length, response_time, tokens_used, confidence):
        """Track query analytics"""
        data = {
            'user_id': user_id,
            'prompt_length': prompt_length,
            'response_time': response_time,
            'tokens_used': tokens_used,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if self.redis_client:
            # Store in Redis for real-time analytics
            key = f"analytics:query:{datetime.utcnow().strftime('%Y-%m-%d')}"
            self.redis_client.lpush(key, json.dumps(data))
            self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
    
    def get_daily_stats(self, date=None):
        """Get daily statistics"""
        if not self.redis_client:
            return {}
        
        date_str = (date or datetime.utcnow()).strftime('%Y-%m-%d')
        key = f"analytics:query:{date_str}"
        
        queries = self.redis_client.lrange(key, 0, -1)
        if not queries:
            return {}
        
        stats = {
            'total_queries': len(queries),
            'avg_response_time': 0,
            'avg_confidence': 0,
            'total_tokens': 0,
            'unique_users': set()
        }
        
        for query_json in queries:
            query = json.loads(query_json)
            stats['avg_response_time'] += query['response_time']
            stats['avg_confidence'] += query['confidence']
            stats['total_tokens'] += query['tokens_used']
            stats['unique_users'].add(query['user_id'])
        
        if stats['total_queries'] > 0:
            stats['avg_response_time'] /= stats['total_queries']
            stats['avg_confidence'] /= stats['total_queries']
        
        stats['unique_users'] = len(stats['unique_users'])
        return stats
```

## 🧪 Testing

### Unit Tests

```python
# tests/test_api.py
import unittest
from unittest.mock import patch, MagicMock
from app import create_app
from app.models import db
import json

class APITestCase(unittest.TestCase):
    
    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()
        db.create_all()
    
    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
    
    @patch('app.services.prsm_service.PRSMClient')
    def test_chat_endpoint(self, mock_prsm_client):
        """Test chat endpoint"""
        # Mock PRSM response
        mock_response = MagicMock()
        mock_response.answer = "Test AI response"
        mock_response.confidence = 0.95
        mock_response.usage.tokens = 50
        mock_response.usage.cost = 0.001
        
        mock_client_instance = MagicMock()
        mock_client_instance.query.return_value = mock_response
        mock_prsm_client.return_value = mock_client_instance
        
        # Test request
        response = self.client.post('/api/chat',
            data=json.dumps({
                'message': 'Hello AI',
                'user_id': 'test-user'
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['response'], 'Test AI response')
        self.assertEqual(data['confidence'], 0.95)
    
    def test_chat_validation(self):
        """Test chat input validation"""
        # Empty message
        response = self.client.post('/api/chat',
            data=json.dumps({'message': ''}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        
        # No message field
        response = self.client.post('/api/chat',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
    
    @patch('app.services.prsm_service.PRSMClient')
    def test_health_endpoint(self, mock_prsm_client):
        """Test health check endpoint"""
        mock_client_instance = MagicMock()
        mock_client_instance.health.return_value = {'status': 'healthy'}
        mock_prsm_client.return_value = mock_client_instance
        
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')

if __name__ == '__main__':
    unittest.main()
```

### Integration Tests

```python
# tests/test_integration.py
import unittest
import os
import tempfile
from app import create_app
from app.models import db

class IntegrationTestCase(unittest.TestCase):
    
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        self.app = create_app('testing')
        self.app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{self.db_path}'
        self.app.config['TESTING'] = True
        
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()
        
        db.create_all()
    
    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_full_conversation_flow(self):
        """Test complete conversation flow"""
        # This would test with a real PRSM instance
        # Skip if PRSM_URL not set
        if not os.environ.get('TEST_PRSM_URL'):
            self.skipTest('TEST_PRSM_URL not set')
        
        # Test conversation creation and message exchange
        pass
```

## 🚀 Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "wsgi:app"]
```

### Production Configuration

```python
# wsgi.py
import os
from app import create_app

app = create_app(os.getenv('FLASK_ENV', 'production'))

if __name__ == "__main__":
    app.run()
```

```bash
# requirements.txt
Flask==2.3.3
prsm-sdk==1.0.0
Flask-CORS==4.0.0
Flask-Limiter==3.5.0
Flask-SQLAlchemy==3.0.5
Flask-SocketIO==5.3.6
gunicorn==21.2.0
prometheus-flask-exporter==0.23.0
redis==5.0.1
python-dotenv==1.0.0
psycopg2-binary==2.9.7
```

## 📋 Best Practices

### Security

```python
# app/security.py
from flask import request, abort
from functools import wraps
import jwt
import os

def require_api_key(f):
    """Require API key for access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_KEY'):
            abort(401)
        return f(*args, **kwargs)
    return decorated_function

def require_jwt(f):
    """Require valid JWT token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            abort(401)
        
        try:
            payload = jwt.decode(token, os.environ.get('JWT_SECRET'), algorithms=['HS256'])
            request.user = payload
        except jwt.InvalidTokenError:
            abort(401)
        
        return f(*args, **kwargs)
    return decorated_function
```

### Performance Optimization

```python
# app/cache.py
from flask_caching import Cache
import hashlib

cache = Cache()

def cache_prsm_response(timeout=300):
    """Cache PRSM responses"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Create cache key from prompt and user
            prompt = kwargs.get('prompt', '')
            user_id = kwargs.get('user_id', '')
            cache_key = f"prsm:{hashlib.md5(f'{prompt}:{user_id}'.encode()).hexdigest()}"
            
            # Try to get from cache
            cached_response = cache.get(cache_key)
            if cached_response:
                return cached_response
            
            # Get fresh response
            response = f(*args, **kwargs)
            
            # Cache if high confidence
            if hasattr(response, 'confidence') and response.confidence > 0.8:
                cache.set(cache_key, response, timeout=timeout)
            
            return response
        return decorated_function
    return decorator
```

---

**Need help with Flask integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).