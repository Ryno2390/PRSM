# Django Integration Guide

Complete guide for integrating PRSM with Django applications to add AI capabilities to your web applications.

## Overview

Django is a high-level Python web framework that encourages rapid development and clean design. This guide shows how to integrate PRSM into Django projects to create AI-powered web applications.

## Benefits of Django + PRSM

- **Mature Framework**: Robust, battle-tested web framework
- **ORM Integration**: Built-in database models and migrations
- **Admin Interface**: Automatic admin panel for managing AI interactions
- **Authentication**: Built-in user management and permissions
- **Scalability**: Support for large-scale applications

## Installation

```bash
# Install Django and PRSM SDK
pip install django prsm-sdk

# Optional: Async support and caching
pip install django[async] redis django-redis celery
```

## Project Setup

### 1. Create Django Project with PRSM

```bash
# Create new Django project
django-admin startproject ai_project
cd ai_project

# Create AI app
python manage.py startapp ai_assistant
```

### 2. Django Settings Configuration

```python
# settings.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# PRSM Configuration
PRSM_CONFIG = {
    'BASE_URL': os.getenv('PRSM_API_URL', 'http://localhost:8000'),
    'API_KEY': os.getenv('PRSM_API_KEY', 'your-api-key'),
    'DEFAULT_TIMEOUT': 30,
    'MAX_RETRIES': 3,
    'CACHE_TTL': 300,
}

# Django Apps
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'ai_assistant',  # Your AI app
]

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Cache configuration (Redis)
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Async support
ASGI_APPLICATION = 'ai_project.asgi.application'

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'prsm_integration.log',
        },
    },
    'loggers': {
        'ai_assistant': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

### 3. ASGI Configuration for Async Support

```python
# asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_project.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    # Add WebSocket routing if needed
})
```

## Models and Database Integration

### AI Query Models

```python
# ai_assistant/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from decimal import Decimal

class AISession(models.Model):
    """AI conversation session"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    context = models.JSONField(default=dict)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Session {self.session_id} - {self.user.username}"

class AIQuery(models.Model):
    """Individual AI query record"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session = models.ForeignKey(AISession, on_delete=models.CASCADE, null=True, blank=True)
    query_id = models.CharField(max_length=100, unique=True)
    prompt = models.TextField()
    response = models.TextField(blank=True)
    ftns_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    processing_time = models.FloatField(default=0)
    quality_score = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Query {self.query_id} - {self.user.username}"
    
    def mark_completed(self, response: str, ftns_cost: float, processing_time: float, quality_score: int):
        """Mark query as completed"""
        self.response = response
        self.ftns_cost = Decimal(str(ftns_cost))
        self.processing_time = processing_time
        self.quality_score = quality_score
        self.status = 'completed'
        self.completed_at = timezone.now()
        self.save()
    
    def mark_failed(self, error_message: str):
        """Mark query as failed"""
        self.error_message = error_message
        self.status = 'failed'
        self.completed_at = timezone.now()
        self.save()

class UserAIQuota(models.Model):
    """User AI usage quota"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    monthly_quota = models.DecimalField(max_digits=10, decimal_places=2, default=1000)
    used_quota = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    last_reset = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.used_quota}/{self.monthly_quota}"
    
    def has_quota(self, required_ftns: float) -> bool:
        """Check if user has sufficient quota"""
        return self.used_quota + Decimal(str(required_ftns)) <= self.monthly_quota
    
    def use_quota(self, ftns_used: float):
        """Deduct used FTNS from quota"""
        self.used_quota += Decimal(str(ftns_used))
        self.save()
```

### Run Migrations

```bash
python manage.py makemigrations ai_assistant
python manage.py migrate
```

## PRSM Service Integration

### PRSM Service Class

```python
# ai_assistant/services.py
import asyncio
import logging
from typing import Optional, Dict, Any
from django.conf import settings
from django.core.cache import cache
from prsm_sdk import PRSMClient, PRSMError
from .models import AIQuery, AISession, UserAIQuota

logger = logging.getLogger(__name__)

class PRSMService:
    """Django service for PRSM integration"""
    
    def __init__(self):
        self.client: Optional[PRSMClient] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize PRSM client"""
        try:
            self.client = PRSMClient(
                base_url=settings.PRSM_CONFIG['BASE_URL'],
                api_key=settings.PRSM_CONFIG['API_KEY'],
                timeout=settings.PRSM_CONFIG['DEFAULT_TIMEOUT']
            )
            logger.info("PRSM client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PRSM client: {e}")
            self.client = None
    
    async def query(self, prompt: str, user, session_id: Optional[str] = None, 
                   context_allocation: int = 50) -> AIQuery:
        """Process AI query and store in database"""
        if not self.client:
            raise Exception("PRSM client not initialized")
        
        # Check user quota
        quota, created = UserAIQuota.objects.get_or_create(
            user=user,
            defaults={'monthly_quota': 1000, 'used_quota': 0}
        )
        
        if not quota.has_quota(context_allocation):
            raise Exception("Insufficient FTNS quota")
        
        # Get or create session
        session = None
        if session_id:
            try:
                session = AISession.objects.get(session_id=session_id, user=user)
            except AISession.DoesNotExist:
                pass
        
        # Create query record
        query_record = AIQuery.objects.create(
            user=user,
            session=session,
            query_id=f"q_{user.id}_{timezone.now().timestamp()}",
            prompt=prompt,
            status='processing'
        )
        
        try:
            # Process with PRSM
            response = await self.client.query(
                prompt=prompt,
                user_id=str(user.id),
                context_allocation=context_allocation,
                session_id=session_id
            )
            
            # Update query record
            query_record.mark_completed(
                response=response.final_answer,
                ftns_cost=response.ftns_charged,
                processing_time=response.processing_time,
                quality_score=response.quality_score
            )
            
            # Update user quota
            quota.use_quota(response.ftns_charged)
            
            logger.info(f"Query {query_record.query_id} completed successfully")
            
        except Exception as e:
            query_record.mark_failed(str(e))
            logger.error(f"Query {query_record.query_id} failed: {e}")
            raise
        
        return query_record
    
    async def create_session(self, user, initial_context: Dict[str, Any] = None) -> AISession:
        """Create new AI session"""
        session = AISession.objects.create(
            user=user,
            session_id=f"sess_{user.id}_{timezone.now().timestamp()}",
            context=initial_context or {}
        )
        
        logger.info(f"Created session {session.session_id} for user {user.username}")
        return session
    
    def get_cached_response(self, prompt: str, user_id: int) -> Optional[str]:
        """Get cached response for identical prompt"""
        cache_key = f"prsm:response:{user_id}:{hash(prompt)}"
        return cache.get(cache_key)
    
    def cache_response(self, prompt: str, user_id: int, response: str):
        """Cache response for future use"""
        cache_key = f"prsm:response:{user_id}:{hash(prompt)}"
        cache.set(cache_key, response, timeout=settings.PRSM_CONFIG['CACHE_TTL'])
    
    async def health_check(self) -> bool:
        """Check PRSM service health"""
        if not self.client:
            return False
        
        try:
            return await self.client.ping()
        except Exception:
            return False

# Global service instance
prsm_service = PRSMService()
```

## Views and URL Configuration

### Async Views

```python
# ai_assistant/views.py
import json
import asyncio
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404
from django.contrib import messages
from django.core.paginator import Paginator
from asgiref.sync import sync_to_async
from .services import prsm_service
from .models import AIQuery, AISession
from .forms import QueryForm

@login_required
def dashboard(request):
    """AI assistant dashboard"""
    recent_queries = AIQuery.objects.filter(user=request.user)[:10]
    active_sessions = AISession.objects.filter(user=request.user, is_active=True)
    
    context = {
        'recent_queries': recent_queries,
        'active_sessions': active_sessions,
        'form': QueryForm()
    }
    return render(request, 'ai_assistant/dashboard.html', context)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
async def ai_query_async(request):
    """Async AI query endpoint"""
    try:
        data = json.loads(request.body)
        prompt = data.get('prompt')
        session_id = data.get('session_id')
        context_allocation = data.get('context_allocation', 50)
        
        if not prompt:
            return JsonResponse({'error': 'Missing prompt'}, status=400)
        
        # Convert user to async-safe object
        user = await sync_to_async(lambda: request.user)()
        
        # Process query
        query_record = await prsm_service.query(
            prompt=prompt,
            user=user,
            session_id=session_id,
            context_allocation=context_allocation
        )
        
        return JsonResponse({
            'query_id': query_record.query_id,
            'answer': query_record.response,
            'cost': float(query_record.ftns_cost),
            'processing_time': query_record.processing_time,
            'quality_score': query_record.quality_score,
            'status': query_record.status
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
def ai_query_sync(request):
    """Synchronous AI query endpoint (for non-async Django)"""
    try:
        data = json.loads(request.body)
        prompt = data.get('prompt')
        
        # Check cache first
        cached_response = prsm_service.get_cached_response(prompt, request.user.id)
        if cached_response:
            return JsonResponse({
                'answer': cached_response,
                'from_cache': True
            })
        
        # Run async query in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            query_record = loop.run_until_complete(
                prsm_service.query(prompt, user=request.user)
            )
            
            # Cache the response
            prsm_service.cache_response(
                prompt, request.user.id, query_record.response
            )
            
            return JsonResponse({
                'query_id': query_record.query_id,
                'answer': query_record.response,
                'cost': float(query_record.ftns_cost),
                'processing_time': query_record.processing_time,
                'from_cache': False
            })
        finally:
            loop.close()
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def query_history(request):
    """Query history view"""
    queries = AIQuery.objects.filter(user=request.user)
    paginator = Paginator(queries, 20)
    
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_cost': sum(q.ftns_cost for q in queries if q.status == 'completed')
    }
    return render(request, 'ai_assistant/history.html', context)

@login_required
def session_detail(request, session_id):
    """Session detail view"""
    session = get_object_or_404(AISession, session_id=session_id, user=request.user)
    queries = AIQuery.objects.filter(session=session)
    
    context = {
        'session': session,
        'queries': queries
    }
    return render(request, 'ai_assistant/session_detail.html', context)

@csrf_exempt
@require_http_methods(["POST"])
@login_required
async def create_session(request):
    """Create new AI session"""
    try:
        data = json.loads(request.body)
        initial_context = data.get('context', {})
        
        user = await sync_to_async(lambda: request.user)()
        session = await prsm_service.create_session(user, initial_context)
        
        return JsonResponse({
            'session_id': session.session_id,
            'created_at': session.created_at.isoformat()
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
async def health_check(request):
    """Health check endpoint"""
    prsm_healthy = await prsm_service.health_check()
    
    return JsonResponse({
        'status': 'healthy' if prsm_healthy else 'degraded',
        'prsm_connected': prsm_healthy
    })
```

### Forms

```python
# ai_assistant/forms.py
from django import forms
from .models import AIQuery

class QueryForm(forms.Form):
    prompt = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Ask me anything...'
        }),
        max_length=2000,
        help_text="Enter your question or request"
    )
    
    context_allocation = forms.IntegerField(
        initial=50,
        min_value=10,
        max_value=500,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': 10
        }),
        help_text="FTNS tokens to allocate for this query"
    )
    
    use_session = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text="Maintain conversation context"
    )
```

### URL Configuration

```python
# ai_assistant/urls.py
from django.urls import path
from . import views

app_name = 'ai_assistant'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('query/async/', views.ai_query_async, name='query_async'),
    path('query/sync/', views.ai_query_sync, name='query_sync'),
    path('session/create/', views.create_session, name='create_session'),
    path('session/<str:session_id>/', views.session_detail, name='session_detail'),
    path('history/', views.query_history, name='history'),
    path('health/', views.health_check, name='health'),
]
```

```python
# ai_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ai/', include('ai_assistant.urls')),
    path('accounts/', include('django.contrib.auth.urls')),
]
```

## Templates

### Base Template

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}AI Assistant{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'ai_assistant:dashboard' %}">🤖 AI Assistant</a>
            
            {% if user.is_authenticated %}
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="{% url 'ai_assistant:dashboard' %}">Dashboard</a>
                    <a class="nav-link" href="{% url 'ai_assistant:history' %}">History</a>
                    <a class="nav-link" href="{% url 'logout' %}">Logout ({{ user.username }})</a>
                </div>
            {% else %}
                <div class="navbar-nav ms-auto">
                    <a class="nav-link" href="{% url 'login' %}">Login</a>
                </div>
            {% endif %}
        </div>
    </nav>
    
    <main class="container mt-4">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}
        
        {% block content %}
        {% endblock %}
    </main>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}
    {% endblock %}
</body>
</html>
```

### Dashboard Template

```html
<!-- templates/ai_assistant/dashboard.html -->
{% extends 'base.html' %}

{% block title %}AI Assistant Dashboard{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">💬 Ask AI Assistant</h5>
            </div>
            <div class="card-body">
                <form id="queryForm">
                    {% csrf_token %}
                    <div class="mb-3">
                        {{ form.prompt.label_tag }}
                        {{ form.prompt }}
                        <div class="form-text">{{ form.prompt.help_text }}</div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            {{ form.context_allocation.label_tag }}
                            {{ form.context_allocation }}
                            <div class="form-text">{{ form.context_allocation.help_text }}</div>
                        </div>
                        <div class="col-md-6">
                            <div class="form-check mt-4">
                                {{ form.use_session }}
                                <label class="form-check-label" for="{{ form.use_session.id_for_label }}">
                                    {{ form.use_session.label }}
                                </label>
                                <div class="form-text">{{ form.use_session.help_text }}</div>
                            </div>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary" id="submitBtn">
                        <span class="spinner-border spinner-border-sm d-none" id="spinner"></span>
                        Submit Query
                    </button>
                </form>
                
                <div id="response" class="mt-4 d-none">
                    <h6>Response:</h6>
                    <div class="alert alert-success" id="responseContent"></div>
                    <small class="text-muted" id="responseMeta"></small>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card mb-3">
            <div class="card-header">
                <h6 class="card-title mb-0">📊 Quick Stats</h6>
            </div>
            <div class="card-body">
                <p><strong>Total Queries:</strong> {{ recent_queries|length }}</p>
                <p><strong>Active Sessions:</strong> {{ active_sessions|length }}</p>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h6 class="card-title mb-0">🕒 Recent Queries</h6>
            </div>
            <div class="card-body">
                {% for query in recent_queries %}
                    <div class="mb-2 pb-2 border-bottom">
                        <small class="text-muted">{{ query.created_at|date:"M d, H:i" }}</small>
                        <p class="mb-1">{{ query.prompt|truncatechars:50 }}</p>
                        <span class="badge badge-{{ query.status|lower }}">
                            {{ query.get_status_display }}
                        </span>
                    </div>
                {% empty %}
                    <p class="text-muted">No recent queries</p>
                {% endfor %}
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.getElementById('queryForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const spinner = document.getElementById('spinner');
    const responseDiv = document.getElementById('response');
    const responseContent = document.getElementById('responseContent');
    const responseMeta = document.getElementById('responseMeta');
    
    // Show loading state
    submitBtn.disabled = true;
    spinner.classList.remove('d-none');
    responseDiv.classList.add('d-none');
    
    try {
        const formData = new FormData(this);
        const data = {
            prompt: formData.get('prompt'),
            context_allocation: parseInt(formData.get('context_allocation')),
            use_session: formData.get('use_session') === 'on'
        };
        
        const response = await fetch('{% url "ai_assistant:query_async" %}', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': formData.get('csrfmiddlewaretoken')
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            responseContent.textContent = result.answer;
            responseMeta.innerHTML = `
                Cost: ${result.cost} FTNS | 
                Time: ${result.processing_time.toFixed(2)}s | 
                Quality: ${result.quality_score}/100
            `;
            responseDiv.classList.remove('d-none');
        } else {
            responseContent.textContent = 'Error: ' + result.error;
            responseContent.className = 'alert alert-danger';
            responseDiv.classList.remove('d-none');
        }
        
    } catch (error) {
        responseContent.textContent = 'Network error: ' + error.message;
        responseContent.className = 'alert alert-danger';
        responseDiv.classList.remove('d-none');
    } finally {
        submitBtn.disabled = false;
        spinner.classList.add('d-none');
    }
});
</script>
{% endblock %}
```

## Admin Interface

```python
# ai_assistant/admin.py
from django.contrib import admin
from .models import AISession, AIQuery, UserAIQuota

@admin.register(AISession)
class AISessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'user', 'created_at', 'is_active']
    list_filter = ['is_active', 'created_at']
    search_fields = ['session_id', 'user__username']
    readonly_fields = ['session_id', 'created_at', 'updated_at']

@admin.register(AIQuery)
class AIQueryAdmin(admin.ModelAdmin):
    list_display = ['query_id', 'user', 'status', 'ftns_cost', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['query_id', 'user__username', 'prompt']
    readonly_fields = ['query_id', 'created_at', 'completed_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'session')

@admin.register(UserAIQuota)
class UserAIQuotaAdmin(admin.ModelAdmin):
    list_display = ['user', 'used_quota', 'monthly_quota', 'quota_percentage']
    list_filter = ['last_reset']
    search_fields = ['user__username']
    
    def quota_percentage(self, obj):
        if obj.monthly_quota > 0:
            return f"{(obj.used_quota / obj.monthly_quota * 100):.1f}%"
        return "0%"
    quota_percentage.short_description = "Usage %"
```

## Background Tasks with Celery

```python
# ai_assistant/tasks.py
from celery import shared_task
from django.contrib.auth.models import User
from .services import prsm_service
import asyncio

@shared_task
def process_ai_query_background(user_id: int, prompt: str, context_allocation: int = 50):
    """Background task for processing AI queries"""
    try:
        user = User.objects.get(id=user_id)
        
        # Run async query in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            query_record = loop.run_until_complete(
                prsm_service.query(
                    prompt=prompt,
                    user=user,
                    context_allocation=context_allocation
                )
            )
            return {
                'success': True,
                'query_id': query_record.query_id,
                'response': query_record.response,
                'cost': float(query_record.ftns_cost)
            }
        finally:
            loop.close()
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@shared_task
def cleanup_old_sessions():
    """Cleanup expired AI sessions"""
    from django.utils import timezone
    from datetime import timedelta
    
    cutoff_date = timezone.now() - timedelta(days=7)
    expired_sessions = AISession.objects.filter(
        updated_at__lt=cutoff_date,
        is_active=True
    )
    
    count = expired_sessions.count()
    expired_sessions.update(is_active=False)
    
    return f"Deactivated {count} expired sessions"
```

## Testing

```python
# ai_assistant/tests.py
from django.test import TestCase, AsyncTestCase
from django.contrib.auth.models import User
from django.urls import reverse
from unittest.mock import AsyncMock, patch
from .models import AIQuery, AISession, UserAIQuota
from .services import prsm_service

class AIQueryModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass'
        )
    
    def test_query_creation(self):
        query = AIQuery.objects.create(
            user=self.user,
            query_id='test_123',
            prompt='Test prompt'
        )
        self.assertEqual(query.status, 'pending')
        self.assertEqual(str(query), 'Query test_123 - testuser')
    
    def test_mark_completed(self):
        query = AIQuery.objects.create(
            user=self.user,
            query_id='test_123',
            prompt='Test prompt'
        )
        
        query.mark_completed(
            response='Test response',
            ftns_cost=25.5,
            processing_time=1.2,
            quality_score=85
        )
        
        self.assertEqual(query.status, 'completed')
        self.assertEqual(query.response, 'Test response')
        self.assertIsNotNone(query.completed_at)

class PRSMServiceTest(AsyncTestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass'
        )
    
    @patch('ai_assistant.services.PRSMClient')
    async def test_query_processing(self, mock_client_class):
        # Mock PRSM client response
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.final_answer = 'Test AI response'
        mock_response.ftns_charged = 25.0
        mock_response.processing_time = 1.5
        mock_response.quality_score = 90
        
        mock_client.query.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        # Test service query
        prsm_service.client = mock_client
        query_record = await prsm_service.query('Test prompt', self.user)
        
        self.assertEqual(query_record.status, 'completed')
        self.assertEqual(query_record.response, 'Test AI response')
        self.assertEqual(float(query_record.ftns_cost), 25.0)

class ViewsTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass'
        )
        self.client.login(username='testuser', password='testpass')
    
    def test_dashboard_view(self):
        response = self.client.get(reverse('ai_assistant:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'AI Assistant')
    
    def test_query_history_view(self):
        # Create test query
        AIQuery.objects.create(
            user=self.user,
            query_id='test_123',
            prompt='Test prompt',
            status='completed'
        )
        
        response = self.client.get(reverse('ai_assistant:history'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test prompt')
```

## Deployment

### Production Settings

```python
# settings/production.py
from .base import *
import os

DEBUG = False
ALLOWED_HOSTS = ['your-domain.com']

# Security
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# PRSM Configuration
PRSM_CONFIG = {
    'BASE_URL': os.getenv('PRSM_API_URL'),
    'API_KEY': os.getenv('PRSM_API_KEY'),
    'DEFAULT_TIMEOUT': 30,
    'MAX_RETRIES': 3,
    'CACHE_TTL': 300,
}

# Celery
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Run migrations
RUN python manage.py migrate

EXPOSE 8000

CMD ["gunicorn", "ai_project.wsgi:application", "--bind", "0.0.0.0:8000"]
```

---

**Next Steps:**
- [Flask Integration](./flask-integration.md)
- [React Integration](./react-integration.md)
- [API Documentation](../api-documentation/)
- [Production Deployment](../platform-integration/)
