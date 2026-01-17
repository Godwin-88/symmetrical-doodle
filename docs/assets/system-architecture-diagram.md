# System Architecture Diagrams

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[React Frontend<br/>localhost:5173]
        UI --> |WebSocket| WS[WebSocket Service]
        UI --> |HTTP/REST| API[API Service]
    end
    
    subgraph "API Gateway"
        LB[Load Balancer<br/>Rate Limiting]
        API --> LB
        WS --> LB
    end
    
    subgraph "Service Layer"
        IL[Intelligence Layer<br/>Python FastAPI<br/>localhost:8000]
        EC[Execution Core<br/>Rust Axum<br/>localhost:8001]
        SE[Simulation Engine<br/>Rust Tokio<br/>localhost:8002]
        
        LB --> IL
        LB --> EC
        LB --> SE
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL<br/>Time-series Data<br/>localhost:5432)]
        NEO[(Neo4j Aura<br/>Graph Database<br/>Cloud)]
        REDIS[(Redis<br/>Cache & Queue<br/>localhost:6379)]
        
        IL --> PG
        IL --> NEO
        IL --> REDIS
        EC --> PG
        EC --> REDIS
        SE --> PG
    end
    
    subgraph "External APIs"
        DERIV[Deriv API<br/>Demo Trading]
        MARKET[Market Data<br/>Providers]
        
        IL --> DERIV
        EC --> DERIV
        IL --> MARKET
    end
    
    style UI fill:#ff8c00,stroke:#333,stroke-width:2px,color:#000
    style IL fill:#4CAF50,stroke:#333,stroke-width:2px,color:#fff
    style EC fill:#2196F3,stroke:#333,stroke-width:2px,color:#fff
    style SE fill:#9C27B0,stroke:#333,stroke-width:2px,color:#fff
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant IL as Intelligence Layer
    participant EC as Execution Core
    participant SE as Simulation Engine
    participant DB as Database
    participant EXT as External APIs
    
    U->>F: User Action
    F->>IL: API Request
    IL->>DB: Query Data
    DB-->>IL: Return Data
    IL->>EXT: External API Call
    EXT-->>IL: Market Data
    IL->>EC: Execution Signal
    EC->>EXT: Place Order
    EXT-->>EC: Order Confirmation
    EC->>DB: Store Trade
    IL-->>F: Response Data
    F-->>U: Update UI
    
    Note over F,DB: Real-time Updates via WebSocket
    IL->>F: WebSocket Update
    F->>U: Live Data Update
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Frontend Components"
        M[Markets]
        I[Intelligence]
        P[Portfolio]
        S[Simulation]
        D[Data & Models]
    end
    
    subgraph "Services"
        MS[Markets Service]
        IS[Intelligence Service]
        PS[Portfolio Service]
        SS[Simulation Service]
        DS[Data Models Service]
    end
    
    subgraph "Backend APIs"
        MA[Market Analytics]
        ML[ML Pipeline]
        PM[Portfolio Manager]
        BT[Backtesting]
        TR[Training]
    end
    
    M --> MS --> MA
    I --> IS --> ML
    P --> PS --> PM
    S --> SS --> BT
    D --> DS --> TR
    
    style M fill:#ff8c00,stroke:#333,stroke-width:2px
    style I fill:#ff8c00,stroke:#333,stroke-width:2px
    style P fill:#ff8c00,stroke:#333,stroke-width:2px
    style S fill:#ff8c00,stroke:#333,stroke-width:2px
    style D fill:#ff8c00,stroke:#333,stroke-width:2px
```

## Database Schema Overview

```mermaid
erDiagram
    PORTFOLIOS ||--o{ POSITIONS : contains
    POSITIONS ||--o{ TRADES : generates
    STRATEGIES ||--o{ TRADES : executes
    ASSETS ||--o{ POSITIONS : held_in
    ASSETS ||--o{ MARKET_DATA : has
    
    PORTFOLIOS {
        uuid id PK
        string name
        decimal total_value
        string status
        timestamp created_at
    }
    
    POSITIONS {
        uuid id PK
        uuid portfolio_id FK
        uuid asset_id FK
        decimal quantity
        decimal avg_price
        decimal unrealized_pnl
    }
    
    TRADES {
        uuid id PK
        uuid position_id FK
        uuid strategy_id FK
        decimal quantity
        decimal price
        string side
        timestamp executed_at
    }
    
    STRATEGIES {
        uuid id PK
        string name
        string type
        json parameters
        boolean active
    }
    
    ASSETS {
        uuid id PK
        string symbol
        string name
        string asset_type
        json metadata
    }
    
    MARKET_DATA {
        uuid id PK
        uuid asset_id FK
        decimal price
        decimal volume
        timestamp timestamp
    }
```

## Neo4j Graph Schema

```mermaid
graph TB
    subgraph "Asset Relationships"
        A1[Asset: EURUSD]
        A2[Asset: GBPUSD]
        A3[Asset: USDJPY]
        
        A1 -.->|correlates_with| A2
        A2 -.->|correlates_with| A3
        A1 -.->|influences| A3
    end
    
    subgraph "Strategy Network"
        S1[Strategy: Trend Following]
        S2[Strategy: Mean Reversion]
        S3[Strategy: Pairs Trading]
        
        S1 -->|uses| M1[Model: LSTM]
        S2 -->|uses| M2[Model: VAE]
        S3 -->|uses| M3[Model: GCN]
        
        S1 -.->|competes_with| S2
        S3 -->|depends_on| S1
    end
    
    subgraph "Portfolio Graph"
        P1[Portfolio: Main]
        P1 -->|contains| POS1[Position: EURUSD Long]
        P1 -->|contains| POS2[Position: GBPUSD Short]
        
        POS1 -->|holds| A1
        POS2 -->|holds| A2
    end
    
    style A1 fill:#4CAF50,stroke:#333,stroke-width:2px
    style S1 fill:#2196F3,stroke:#333,stroke-width:2px
    style P1 fill:#FF9800,stroke:#333,stroke-width:2px
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV_F[Frontend Dev Server<br/>npm run dev]
        DEV_B[Backend Services<br/>Local Docker]
        DEV_DB[Local Databases<br/>Docker Compose]
    end
    
    subgraph "Staging Environment"
        STAGE_F[Frontend Build<br/>Static Files]
        STAGE_B[Backend Services<br/>Containers]
        STAGE_DB[Staging Databases<br/>Cloud]
    end
    
    subgraph "Production Environment"
        PROD_LB[Load Balancer<br/>SSL Termination]
        PROD_F[Frontend<br/>CDN + Static Hosting]
        PROD_B[Backend Services<br/>Container Orchestration]
        PROD_DB[Production Databases<br/>Managed Services]
        PROD_MON[Monitoring<br/>Logging & Metrics]
        
        PROD_LB --> PROD_F
        PROD_LB --> PROD_B
        PROD_B --> PROD_DB
        PROD_B --> PROD_MON
    end
    
    DEV_F -.->|Build & Test| STAGE_F
    STAGE_F -.->|Deploy| PROD_F
    DEV_B -.->|CI/CD| STAGE_B
    STAGE_B -.->|Deploy| PROD_B
```

## Security Architecture

```mermaid
graph TB
    subgraph "External Access"
        USER[User Browser]
        API_CLIENT[API Client]
    end
    
    subgraph "Security Layer"
        WAF[Web Application Firewall]
        LB[Load Balancer + SSL]
        AUTH[Authentication Service]
        RBAC[Role-Based Access Control]
    end
    
    subgraph "Application Layer"
        FE[Frontend Application]
        BE[Backend Services]
    end
    
    subgraph "Data Layer"
        DB[Encrypted Databases]
        SECRETS[Secrets Management]
    end
    
    USER --> WAF
    API_CLIENT --> WAF
    WAF --> LB
    LB --> AUTH
    AUTH --> RBAC
    RBAC --> FE
    RBAC --> BE
    BE --> DB
    BE --> SECRETS
    
    style WAF fill:#f44336,stroke:#333,stroke-width:2px,color:#fff
    style AUTH fill:#ff9800,stroke:#333,stroke-width:2px,color:#fff
    style RBAC fill:#ff9800,stroke:#333,stroke-width:2px,color:#fff
```

## Monitoring & Observability

```mermaid
graph TB
    subgraph "Application Layer"
        APP1[Frontend]
        APP2[Intelligence Layer]
        APP3[Execution Core]
        APP4[Simulation Engine]
    end
    
    subgraph "Monitoring Stack"
        LOGS[Centralized Logging<br/>Structured JSON]
        METRICS[Metrics Collection<br/>Time-series DB]
        TRACES[Distributed Tracing<br/>Request Tracking]
        HEALTH[Health Checks<br/>Service Status]
    end
    
    subgraph "Alerting & Dashboards"
        DASH[Monitoring Dashboard<br/>Real-time Metrics]
        ALERT[Alert Manager<br/>Notifications]
        REPORT[Reporting<br/>Analytics]
    end
    
    APP1 --> LOGS
    APP2 --> LOGS
    APP3 --> LOGS
    APP4 --> LOGS
    
    APP1 --> METRICS
    APP2 --> METRICS
    APP3 --> METRICS
    APP4 --> METRICS
    
    APP1 --> TRACES
    APP2 --> TRACES
    APP3 --> TRACES
    APP4 --> TRACES
    
    APP1 --> HEALTH
    APP2 --> HEALTH
    APP3 --> HEALTH
    APP4 --> HEALTH
    
    LOGS --> DASH
    METRICS --> DASH
    TRACES --> DASH
    HEALTH --> DASH
    
    DASH --> ALERT
    DASH --> REPORT
    
    style DASH fill:#4CAF50,stroke:#333,stroke-width:2px,color:#fff
    style ALERT fill:#f44336,stroke:#333,stroke-width:2px,color:#fff
```

---

**Note**: These diagrams use Mermaid syntax and can be rendered in most modern markdown viewers, including GitHub, GitLab, and documentation platforms that support Mermaid.