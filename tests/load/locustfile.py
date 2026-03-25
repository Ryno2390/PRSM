"""
PRSM Load Test - Locust Configuration

Simulates realistic user workloads on the PRSM API to establish
performance baselines and identify bottlenecks.

Requirements:
    pip install locust

Run:
    # Quick test (50 users, 5 second ramp-up, 2 minute duration)
    locust -f tests/load/locustfile.py --headless -u 50 -r 5 -t 120s

    # With web UI for interactive testing
    locust -f tests/load/locustfile.py

    # Distributed testing (master)
    locust -f tests/load/locustfile.py --master

    # Distributed testing (worker)
    locust -f tests/load/locustfile.py --worker --master-host=<master-ip>

Prerequisites:
    - PRSM node running at http://localhost:8000
    - Test user accounts or auto-registration enabled

Expected Performance Baselines (Phase 7):
    - Health endpoint: < 10ms median, < 50ms p99
    - FTNS balance: < 50ms median, < 200ms p99
    - Marketplace browse: < 100ms median, < 500ms p99
    - Query endpoint: < 2000ms median (AI call), < 5000ms p99
"""

import json
import uuid
import random
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner


class PRSMUser(HttpUser):
    """
    Simulates a PRSM user performing typical operations.

    User behaviors weighted by frequency:
    - Health checks: 10 (most frequent)
    - FTNS balance: 5
    - Marketplace browse: 3
    - Query submission: 2 (expensive, less frequent)
    - Metrics: 1
    """

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """
        Initialize user session.

        Attempts to authenticate and get JWT token.
        Falls back to anonymous access if auth fails.
        """
        self.token = None
        self.headers = {}

        # Try to authenticate
        self._authenticate()

    def _authenticate(self):
        """Authenticate and store JWT token."""
        # Generate unique test user
        test_username = f"loadtest_{uuid.uuid4().hex[:8]}"
        test_password = "LoadTest123!"

        try:
            # Try login first (in case user exists)
            resp = self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": test_username,
                    "password": test_password
                },
                catch_response=True
            )

            if resp.status_code == 200:
                data = resp.json()
                self.token = data.get("access_token")
                self.headers = {"Authorization": f"Bearer {self.token}"}
                return

            # Try registration if login failed
            resp = self.client.post(
                "/api/v1/auth/register",
                json={
                    "username": test_username,
                    "password": test_password,
                    "email": f"{test_username}@loadtest.local"
                },
                catch_response=True
            )

            if resp.status_code in (200, 201):
                # Login after registration
                resp = self.client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": test_username,
                        "password": test_password
                    },
                    catch_response=True
                )

                if resp.status_code == 200:
                    data = resp.json()
                    self.token = data.get("access_token")
                    self.headers = {"Authorization": f"Bearer {self.token}"}

        except Exception:
            # Continue without authentication
            pass

    @task(10)
    def check_health(self):
        """
        Check service health.

        Most frequent operation - load balancers and monitoring
        hit this frequently.
        """
        self.client.get("/health")

    @task(5)
    def get_ftns_balance(self):
        """
        Check FTNS token balance.

        Common operation for users to check their balance.
        Requires authentication (falls back silently if not auth'd).
        """
        self.client.get(
            "/api/v1/ftns/balance",
            headers=self.headers
        )

    @task(3)
    def browse_marketplace(self):
        """
        Browse marketplace listings.

        Simulates user browsing available models/datasets.
        """
        # Random pagination for variety
        offset = random.randint(0, 100)
        limit = random.randint(10, 50)

        self.client.get(
            f"/api/v1/marketplace/listings?offset={offset}&limit={limit}",
            headers=self.headers
        )

    @task(2)
    def submit_query(self):
        """
        Submit an AI query.

        This is the expensive operation - calls external AI APIs.
        Weighted lower to avoid overwhelming AI backends.
        """
        prompts = [
            "Explain the FTNS token economy in one sentence",
            "What is the purpose of the PRSM marketplace?",
            "How does the distillation process work?",
            "Describe the governance voting mechanism",
            "What are the benefits of staking FTNS tokens?",
        ]

        prompt = random.choice(prompts)

        self.client.post(
            "/api/v1/query",
            json={
                "prompt": prompt,
                "nwtn_context_allocation": random.randint(5, 20),
            },
            headers=self.headers,
            timeout=30  # AI calls can take time
        )

    @task(1)
    def get_metrics(self):
        """
        Get system metrics.

        Used by monitoring and dashboards.
        """
        self.client.get("/health/metrics")


class PRSMPowerUser(PRSMUser):
    """
    Simulates a power user with higher activity levels.

    Power users submit more queries and interact more with
    the marketplace.
    """

    wait_time = between(0.5, 1.5)  # Shorter wait times

    @task(15)  # More frequent health checks
    def check_health(self):
        self.client.get("/health")

    @task(10)  # More balance checks
    def get_ftns_balance(self):
        self.client.get("/api/v1/ftns/balance", headers=self.headers)

    @task(5)  # More marketplace browsing
    def browse_marketplace(self):
        offset = random.randint(0, 50)
        self.client.get(
            f"/api/v1/marketplace/listings?offset={offset}&limit=20",
            headers=self.headers
        )

    @task(5)  # More queries
    def submit_query(self):
        prompts = [
            "Analyze the current token distribution",
            "Compare performance metrics across models",
            "Generate a summary of recent governance proposals",
            "Evaluate the marketplace growth trends",
        ]
        prompt = random.choice(prompts)
        self.client.post(
            "/api/v1/query",
            json={"prompt": prompt, "nwtn_context_allocation": 15},
            headers=self.headers,
            timeout=30
        )


# === Event Handlers for Reporting ===

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts - log configuration."""
    print("\n" + "=" * 60)
    print("PRSM Load Test Starting")
    print("=" * 60)
    print(f"Target host: {environment.host}")
    if isinstance(environment.runner, MasterRunner):
        print("Running in MASTER mode")
    elif isinstance(environment.runner, WorkerRunner):
        print("Running in WORKER mode")
    else:
        print("Running in STANDALONE mode")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - log summary."""
    print("\n" + "=" * 60)
    print("PRSM Load Test Complete")
    print("=" * 60)

    if hasattr(environment.stats, 'total'):
        stats = environment.stats.total
        print(f"Total requests: {stats.num_requests}")
        print(f"Total failures: {stats.num_failures}")
        print(f"Average response time: {stats.avg_response_time:.2f}ms")
        print(f"Median response time: {stats.median_response_time:.2f}ms")
        print(f"95th percentile: {stats.get_response_time_percentile(0.95):.2f}ms")
        print(f"99th percentile: {stats.get_response_time_percentile(0.99):.2f}ms")
        print(f"Requests/sec: {stats.total_rps:.2f}")

    print("=" * 60 + "\n")


# === Custom Tasks for Specific Testing ===

class APIStressTest(HttpUser):
    """
    Stress test for API endpoints.

    Used for load testing specific endpoints at high volume
    to identify performance bottlenecks.
    """

    wait_time = between(0.1, 0.5)  # Very short wait - aggressive testing

    @task
    def stress_health(self):
        """Hammer health endpoint."""
        self.client.get("/health")

    @task
    def stress_ftns(self):
        """Rapid FTNS balance checks."""
        self.client.get("/api/v1/ftns/balance")


# === Configuration for Different Test Scenarios ===

# Standard load test configuration
STANDARD_CONFIG = {
    "user_classes": [PRSMUser],
    "spawn_rate": 5,  # Users per second
    "run_time": "2m",
    "host": "http://localhost:8000"
}

# Stress test configuration
STRESS_CONFIG = {
    "user_classes": [APIStressTest],
    "spawn_rate": 50,  # Aggressive ramp-up
    "run_time": "30s",
    "host": "http://localhost:8000"
}

# Power user simulation
POWER_USER_CONFIG = {
    "user_classes": [PRSMPowerUser],
    "spawn_rate": 2,
    "run_time": "5m",
    "host": "http://localhost:8000"
}
