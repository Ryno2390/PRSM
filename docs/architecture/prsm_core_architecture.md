# PRSM Core Architecture Analysis

This document provides a high-level overview of the PRSM codebase's architecture, focusing on the contents of the `/prsm/` directory.

## High-Level Analysis

The PRSM application is structured as a **Modular Monolith**. It is a single, deployable application, but its internal components are organized into distinct, loosely-coupled modules that each handle a specific business capability. This approach combines the deployment simplicity of a monolith with the organizational benefits of a more distributed architecture.

Key characteristics observed:

*   **Service-Oriented Internal Structure:** The `api` directory contains numerous router files (e.g., `teams_api.py`, `auth_api.py`, `payment_api.py`), each corresponding to a different functional domain. These are all integrated into the main FastAPI application in `prsm/api/main.py`, which acts as a "gateway" to the different services within the monolith.
*   **Centralized Core Services:** A `core` directory provides shared functionalities like configuration management (`config.py`), database connectivity (`database.py`), and clients for external services like Redis (`redis_client.py`), IPFS (`ipfs_client.py`), and vector databases (`vector_db.py`).
*   **Asynchronous by Design:** The application heavily utilizes Python's `asyncio` framework and `async`/`await` syntax, particularly with the FastAPI web framework and SQLAlchemy's async support. This is crucial for building a high-performance, I/O-bound application that can handle many concurrent requests.

## Design Patterns

Several key design patterns are in use throughout the codebase:

*   **Dependency Injection:** FastAPI's `Depends` system is used to inject dependencies, such as the current user or database sessions, into API endpoints. This promotes loose coupling and testability.
*   **Repository Pattern:** The query helper classes in `prsm/core/database.py` (e.g., `SessionQueries`, `FTNSQueries`) encapsulate data access logic, separating the business logic of the services from the persistence layer.
*   **Configuration Provider:** The `PRSMSettings` class, managed with `pydantic-settings`, acts as a centralized configuration provider for the entire application.
*   **Lifespan Management (Startup/Shutdown Events):** The `lifespan` async context manager in `prsm/api/main.py` elegantly handles the initialization and cleanup of resources like database connections, Redis clients, and IPFS clients, ensuring the application starts and stops gracefully.

## Architectural Diagram

The following diagram illustrates the high-level architecture of the PRSM application:

```mermaid
graph TD
    subgraph "External World"
        A[Users / Clients]
    end

    subgraph "PRSM Application"
        B[API Gateway (FastAPI)]

        subgraph "Core Services"
            C[Configuration]
            D[Database (PostgreSQL w/ SQLAlchemy)]
            E[Caching (Redis)]
            F[Distributed Storage (IPFS)]
            G[Vector DB]
        end

        subgraph "Application Modules"
            H[Auth]
            I[Teams]
            J[Marketplace]
            K[Governance]
            L[Payments (FTNS)]
            M[Agents]
            N[...]
        end
    end

    A --> B
    B --> H
    B --> I
    B --> J
    B --> K
    B --> L
    B --> M
    B --> N

    H --> D
    I --> D
    J --> D
    K --> D
    L --> D
    M --> D

    subgraph "External Services"
        O[PostgreSQL]
        P[Redis]
        Q[IPFS Network]
        R[Vector DBs (e.g., Pinecone)]
        S[AI Models (e.g., OpenAI)]
    end

    D -- Manages Connection to --> O
    E -- Manages Connection to --> P
    F -- Manages Connection to --> Q
    G -- Manages Connection to --> R

    M -- Uses --> S

    H -- Uses --> C
    I -- Uses --> C
    J -- Uses --> C
    K -- Uses --> C
    L -- Uses --> C
    M -- Uses --> C