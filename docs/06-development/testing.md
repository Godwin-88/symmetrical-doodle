# Testing Guide

## Overview

The algorithmic trading system includes comprehensive testing at all levels: unit tests, integration tests, end-to-end tests, and property-based tests.

## Testing Strategy

### Test Pyramid
```
    E2E Tests (Few)
   ┌─────────────────┐
   │  User Journeys  │
   └─────────────────┘
  ┌───────────────────┐
  │ Integration Tests │  (Some)
  │  API + Database   │
  └───────────────────┘
 ┌─────────────────────┐
 │    Unit Tests       │  (Many)
 │  Functions + Logic  │
 └─────────────────────┘
```

### Testing Levels

1. **Unit Tests** - Individual functions and components
2. **Integration Tests** - Service interactions and database operations
3. **End-to-End Tests** - Complete user workflows
4. **Property-Based Tests** - Mathematical properties and invariants
5. **Performance Tests** - Load and stress testing

## Frontend Testing

### Setup
```bash
cd frontend
npm install
```

### Unit Tests (Jest + React Testing Library)

**Test Files Location**: `frontend/src/**/*.test.ts(x)`

**Running Tests**:
```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run specific test file
npm test Markets.test.tsx
```

**Example Component Test**:
```typescript
// frontend/src/app/components/__tests__/Markets.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Markets } from '../Markets';
import { marketsService } from '../../services/marketsService';

// Mock the service
jest.mock('../../services/marketsService');
const mockMarketsService = marketsService as jest.Mocked<typeof marketsService>;

describe('Markets Component', () => {
  beforeEach(() => {
    mockMarketsService.getMarketData.mockResolvedValue({
      symbol: 'EURUSD',
      price: 1.0850,
      change: 0.0025
    });
  });

  it('displays market data correctly', async () => {
    render(<Markets />);
    
    await waitFor(() => {
      expect(screen.getByText('EURUSD')).toBeInTheDocument();
      expect(screen.getByText('1.0850')).toBeInTheDocument();
    });
  });

  it('handles watchlist addition', async () => {
    render(<Markets />);
    
    const addButton = screen.getByText('Add to Watchlist');
    fireEvent.click(addButton);
    
    await waitFor(() => {
      expect(mockMarketsService.addToWatchlist).toHaveBeenCalledWith('EURUSD');
    });
  });
});
```

**Service Tests**:
```typescript
// frontend/src/services/__tests__/marketsService.test.ts
import { marketsService } from '../marketsService';

describe('Markets Service', () => {
  it('fetches market data with fallback', async () => {
    // Mock fetch to fail (simulate backend down)
    global.fetch = jest.fn().mockRejectedValue(new Error('Network error'));
    
    const data = await marketsService.getMarketData('EURUSD');
    
    // Should return mock data
    expect(data).toHaveProperty('symbol', 'EURUSD');
    expect(data).toHaveProperty('price');
    expect(typeof data.price).toBe('number');
  });

  it('uses backend data when available', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        symbol: 'EURUSD',
        price: 1.0850
      })
    });
    
    const data = await marketsService.getMarketData('EURUSD');
    
    expect(data.price).toBe(1.0850);
  });
});
```

### Integration Tests

**Testing API Integration**:
```typescript
// frontend/src/integration/__tests__/api.test.ts
import { apiClient } from '../services/api';

describe('API Integration', () => {
  it('handles backend connectivity', async () => {
    const health = await apiClient.get('/health');
    expect(health.status).toBe('healthy');
  });

  it('falls back to mock data gracefully', async () => {
    // Simulate backend down
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockRejectedValue(new Error('Connection failed'));
    
    const markets = await apiClient.get('/api/v1/markets/data');
    expect(markets).toBeDefined();
    
    global.fetch = originalFetch;
  });
});
```

### E2E Tests (Playwright)

**Setup**:
```bash
npm install @playwright/test
npx playwright install
```

**Test Files**: `frontend/e2e/**/*.spec.ts`

**Example E2E Test**:
```typescript
// frontend/e2e/trading-workflow.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Trading Workflow', () => {
  test('complete trading journey', async ({ page }) => {
    // Navigate to application
    await page.goto('http://localhost:5173');
    
    // Check markets data loads
    await expect(page.locator('[data-testid="markets-tab"]')).toBeVisible();
    await page.click('[data-testid="markets-tab"]');
    await expect(page.locator('[data-testid="market-data"]')).toBeVisible();
    
    // Add to watchlist
    await page.click('[data-testid="add-watchlist-btn"]');
    await expect(page.locator('[data-testid="watchlist-item"]')).toBeVisible();
    
    // Navigate to portfolio
    await page.click('[data-testid="portfolio-tab"]');
    await expect(page.locator('[data-testid="portfolio-summary"]')).toBeVisible();
    
    // Create new portfolio
    await page.click('[data-testid="create-portfolio-btn"]');
    await page.fill('[data-testid="portfolio-name"]', 'Test Portfolio');
    await page.fill('[data-testid="initial-capital"]', '100000');
    await page.click('[data-testid="save-portfolio-btn"]');
    
    // Verify portfolio created
    await expect(page.locator('text=Test Portfolio')).toBeVisible();
  });
});
```

**Running E2E Tests**:
```bash
# Run all E2E tests
npx playwright test

# Run specific test
npx playwright test trading-workflow.spec.ts

# Run with UI
npx playwright test --ui

# Generate test report
npx playwright show-report
```

## Backend Testing

### Python Tests (Intelligence Layer)

**Setup**:
```bash
cd intelligence-layer
pip install -e ".[test]"
```

**Running Tests**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=intelligence_layer

# Run specific test file
pytest tests/test_market_analytics.py

# Run with verbose output
pytest -v

# Run property-based tests
pytest tests/test_*_property.py
```

**Unit Test Example**:
```python
# intelligence-layer/tests/test_market_analytics.py
import pytest
from intelligence_layer.market_analytics import MarketAnalytics
from unittest.mock import Mock, patch

class TestMarketAnalytics:
    def setup_method(self):
        self.analytics = MarketAnalytics()
    
    def test_calculate_correlation(self):
        # Test data
        data1 = [1.0, 1.1, 1.2, 1.1, 1.0]
        data2 = [2.0, 2.2, 2.4, 2.2, 2.0]
        
        correlation = self.analytics.calculate_correlation(data1, data2)
        
        assert 0.9 <= correlation <= 1.0
    
    @patch('intelligence_layer.market_analytics.external_api')
    def test_fetch_market_data_with_fallback(self, mock_api):
        # Simulate API failure
        mock_api.get_data.side_effect = Exception("API Error")
        
        data = self.analytics.get_market_data("EURUSD")
        
        # Should return mock data
        assert data is not None
        assert data['symbol'] == 'EURUSD'
    
    def test_regime_detection(self):
        # Test regime detection with known data
        trending_data = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        regime = self.analytics.detect_regime(trending_data)
        
        assert regime['type'] == 'trending'
        assert regime['confidence'] > 0.7
```

**Property-Based Test Example**:
```python
# intelligence-layer/tests/test_portfolio_property.py
import pytest
from hypothesis import given, strategies as st
from intelligence_layer.portfolio import Portfolio

class TestPortfolioProperties:
    @given(
        initial_capital=st.floats(min_value=1000, max_value=1000000),
        positions=st.lists(
            st.tuples(
                st.text(min_size=3, max_size=6),  # symbol
                st.floats(min_value=-1000, max_value=1000)  # quantity
            ),
            min_size=0,
            max_size=10
        )
    )
    def test_portfolio_value_conservation(self, initial_capital, positions):
        """Portfolio total value should equal sum of position values plus cash"""
        portfolio = Portfolio(initial_capital=initial_capital)
        
        for symbol, quantity in positions:
            if quantity != 0:
                portfolio.add_position(symbol, quantity, price=100.0)
        
        # Property: Total value = Cash + Position values
        total_value = portfolio.get_total_value()
        cash = portfolio.get_cash()
        position_values = sum(pos.get_market_value() for pos in portfolio.positions)
        
        assert abs(total_value - (cash + position_values)) < 0.01
```

### Rust Tests (Execution Core)

**Running Tests**:
```bash
cd execution-core

# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_portfolio_management

# Run with coverage
cargo tarpaulin --out Html
```

**Unit Test Example**:
```rust
// execution-core/src/portfolio.rs
#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::new("test-portfolio", dec!(100000));
        
        assert_eq!(portfolio.name(), "test-portfolio");
        assert_eq!(portfolio.total_value(), dec!(100000));
        assert_eq!(portfolio.cash(), dec!(100000));
    }

    #[test]
    fn test_add_position() {
        let mut portfolio = Portfolio::new("test", dec!(100000));
        
        portfolio.add_position(
            "EURUSD".to_string(),
            dec!(10000),
            dec!(1.0850)
        ).unwrap();
        
        assert_eq!(portfolio.positions().len(), 1);
        assert!(portfolio.cash() < dec!(100000));
    }

    #[test]
    fn test_risk_limits() {
        let mut portfolio = Portfolio::new("test", dec!(100000));
        portfolio.set_max_position_size(dec!(0.1)); // 10% max
        
        // Should fail - position too large
        let result = portfolio.add_position(
            "EURUSD".to_string(),
            dec!(50000), // 50% of portfolio
            dec!(1.0850)
        );
        
        assert!(result.is_err());
    }
}
```

**Property-Based Test Example**:
```rust
// execution-core/src/risk.rs
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_var_calculation_properties(
            returns in prop::collection::vec(-0.1f64..0.1f64, 100..1000),
            confidence in 0.9f64..0.99f64
        ) {
            let var = calculate_var(&returns, confidence);
            
            // Property: VaR should be negative (loss)
            prop_assert!(var <= 0.0);
            
            // Property: Higher confidence should give higher VaR (more negative)
            let var_higher = calculate_var(&returns, confidence + 0.01);
            prop_assert!(var_higher <= var);
        }
    }
}
```

## Integration Testing

### Database Tests

**PostgreSQL Integration**:
```python
# intelligence-layer/tests/test_database_integration.py
import pytest
from sqlalchemy import create_engine
from intelligence_layer.database import Database

@pytest.fixture
def test_db():
    # Use test database
    engine = create_engine("postgresql://test:test@localhost:5432/test_db")
    db = Database(engine)
    db.create_tables()
    yield db
    db.drop_tables()

def test_portfolio_crud(test_db):
    # Create
    portfolio_id = test_db.create_portfolio("Test Portfolio", 100000.0)
    assert portfolio_id is not None
    
    # Read
    portfolio = test_db.get_portfolio(portfolio_id)
    assert portfolio['name'] == "Test Portfolio"
    
    # Update
    test_db.update_portfolio(portfolio_id, name="Updated Portfolio")
    updated = test_db.get_portfolio(portfolio_id)
    assert updated['name'] == "Updated Portfolio"
    
    # Delete
    test_db.delete_portfolio(portfolio_id)
    deleted = test_db.get_portfolio(portfolio_id)
    assert deleted is None
```

**Neo4j Integration**:
```python
# intelligence-layer/tests/test_neo4j_integration.py
import pytest
from neo4j import GraphDatabase
from intelligence_layer.graph_analytics import GraphAnalytics

@pytest.fixture
def test_graph():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "test"))
    analytics = GraphAnalytics(driver)
    
    # Clean test data
    with driver.session() as session:
        session.run("MATCH (n:TestNode) DETACH DELETE n")
    
    yield analytics
    
    # Cleanup
    with driver.session() as session:
        session.run("MATCH (n:TestNode) DETACH DELETE n")
    driver.close()

def test_graph_operations(test_graph):
    # Create test nodes
    test_graph.create_asset_relationship("EURUSD", "GBPUSD", "correlates_with", 0.75)
    
    # Query relationships
    correlations = test_graph.get_correlations("EURUSD")
    assert len(correlations) == 1
    assert correlations[0]['target'] == "GBPUSD"
    assert correlations[0]['strength'] == 0.75
```

### API Integration Tests

**FastAPI Integration**:
```python
# intelligence-layer/tests/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from intelligence_layer.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_market_data_endpoint():
    response = client.get("/api/v1/markets/data?symbol=EURUSD&timeframe=1h")
    assert response.status_code == 200
    
    data = response.json()
    assert data["symbol"] == "EURUSD"
    assert "data" in data

def test_model_training_endpoint():
    payload = {
        "model_id": "lstm-001",
        "dataset_id": "dataset-001",
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32
        }
    }
    
    response = client.post("/api/v1/models/lstm-001/train", json=payload)
    assert response.status_code == 202  # Accepted for async processing
```

## Performance Testing

### Load Testing (Locust)

**Setup**:
```bash
pip install locust
```

**Load Test Script**:
```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class TradingSystemUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login or setup
        pass
    
    @task(3)
    def get_market_data(self):
        self.client.get("/api/v1/markets/data?symbol=EURUSD&timeframe=1h")
    
    @task(2)
    def get_portfolio(self):
        self.client.get("/api/v1/portfolios/portfolio-001")
    
    @task(1)
    def run_analysis(self):
        self.client.post("/api/v1/graph/analyze", json={
            "algorithm": "pagerank",
            "parameters": {"damping_factor": 0.85}
        })
```

**Running Load Tests**:
```bash
# Start load test
locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Command line load test
locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 60s --headless
```

### Stress Testing

**Database Stress Test**:
```python
# tests/stress/test_database_stress.py
import asyncio
import pytest
from concurrent.futures import ThreadPoolExecutor

async def test_concurrent_portfolio_operations():
    """Test database under concurrent load"""
    
    async def create_portfolio(i):
        # Simulate concurrent portfolio creation
        return await db.create_portfolio(f"Portfolio-{i}", 100000.0)
    
    # Create 100 portfolios concurrently
    tasks = [create_portfolio(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    
    # All should succeed
    assert len(results) == 100
    assert all(r is not None for r in results)
```

## Test Data Management

### Test Fixtures

**Market Data Fixtures**:
```python
# tests/fixtures/market_data.py
import pytest
from datetime import datetime, timedelta

@pytest.fixture
def sample_market_data():
    """Generate sample OHLCV data"""
    base_time = datetime.now()
    data = []
    
    for i in range(100):
        data.append({
            'timestamp': base_time + timedelta(hours=i),
            'open': 1.0850 + (i * 0.0001),
            'high': 1.0860 + (i * 0.0001),
            'low': 1.0840 + (i * 0.0001),
            'close': 1.0855 + (i * 0.0001),
            'volume': 1000000 + (i * 1000)
        })
    
    return data

@pytest.fixture
def trending_data():
    """Generate trending market data"""
    return [1.0 + (i * 0.01) for i in range(50)]

@pytest.fixture
def mean_reverting_data():
    """Generate mean-reverting market data"""
    import math
    return [1.0 + 0.1 * math.sin(i * 0.1) for i in range(50)]
```

### Mock Services

**External API Mocks**:
```python
# tests/mocks/external_apis.py
from unittest.mock import Mock

class MockDerivAPI:
    def __init__(self):
        self.connected = True
        self.balance = 10000.0
    
    def get_ticks(self, symbol):
        return {
            'symbol': symbol,
            'bid': 1.0850,
            'ask': 1.0852,
            'timestamp': '2026-01-17T10:00:00Z'
        }
    
    def place_order(self, symbol, side, amount):
        return {
            'order_id': 'order-123',
            'status': 'filled',
            'fill_price': 1.0851
        }

@pytest.fixture
def mock_deriv_api():
    return MockDerivAPI()
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: cd frontend && npm ci
      - name: Run tests
        run: cd frontend && npm test -- --coverage --watchAll=false
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  backend-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      neo4j:
        image: neo4j:5
        env:
          NEO4J_AUTH: neo4j/test
        options: >-
          --health-cmd "cypher-shell -u neo4j -p test 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          cd intelligence-layer
          pip install -e ".[test]"
      - name: Run tests
        run: |
          cd intelligence-layer
          pytest --cov=intelligence_layer --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  rust-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run tests
        run: |
          cd execution-core
          cargo test
          cd ../simulation-engine
          cargo test

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install Playwright
        run: |
          cd frontend
          npm ci
          npx playwright install
      - name: Start services
        run: |
          docker-compose up -d
          ./scripts/start-all.sh &
          sleep 30
      - name: Run E2E tests
        run: |
          cd frontend
          npx playwright test
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: playwright-report
          path: frontend/playwright-report/
```

## Test Coverage Goals

### Coverage Targets
- **Unit Tests**: > 80% line coverage
- **Integration Tests**: > 70% API endpoint coverage
- **E2E Tests**: > 90% critical user journey coverage

### Coverage Reports
```bash
# Frontend coverage
cd frontend && npm run test:coverage

# Python coverage
cd intelligence-layer && pytest --cov=intelligence_layer --cov-report=html

# Rust coverage
cd execution-core && cargo tarpaulin --out Html
```

## Best Practices

### Test Organization
1. **Arrange-Act-Assert** pattern
2. **One assertion per test** (when possible)
3. **Descriptive test names**
4. **Independent tests** (no test dependencies)
5. **Fast execution** (< 1 second per unit test)

### Mock Strategy
1. **Mock external dependencies**
2. **Use real objects for internal logic**
3. **Verify interactions, not implementations**
4. **Reset mocks between tests**

### Data Management
1. **Use factories for test data**
2. **Clean up after tests**
3. **Isolate test databases**
4. **Use transactions for rollback**

---

**Last Updated**: January 2026  
**Coverage Status**: 85% overall  
**Next**: [API Reference](./api-reference.md)