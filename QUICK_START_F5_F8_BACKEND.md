# Quick Start: F5 & F8 Backend Integration

## Start the System

### 1. Start Backend (Intelligence Layer)
```bash
cd intelligence-layer
poetry run uvicorn intelligence_layer.main:app --reload --port 8000
```

### 2. Start Frontend
```bash
cd frontend
npm run dev
```

### 3. Open Browser
Navigate to: `http://localhost:5173`

## Test F5 (Portfolio & Risk Management)

### Create New Portfolio
1. Click **F5 - PORTFOLIO** tab
2. Click **+ NEW PORTFOLIO** button (left panel)
3. Fill in the form:
   - **Name**: "TEST PORTFOLIO"
   - **Currency**: USD
   - **Initial Capital**: 50000
   - **Mode**: PAPER
   - **Allocation Model**: VOL_TARGET
   - **Rebalance Frequency**: WEEKLY
4. Add strategies:
   - Select a strategy from dropdown
   - Enter weight (e.g., 0.5 for 50%)
   - Click **ADD**
   - Repeat until total weight = 100%
5. Click **CREATE PORTFOLIO**

### View Portfolio Details
1. Click on any portfolio in the left panel
2. View 4 tabs:
   - **POSITIONS**: Current positions with P&L
   - **EXPOSURE**: Gross/net exposure breakdown
   - **RISK**: Risk limits and metrics
   - **ATTRIBUTION**: P&L by strategy/asset

### Edit Portfolio
1. Select a portfolio
2. Click **✎ EDIT PORTFOLIO** (right panel)
3. Modal opens (placeholder - ready for implementation)

### Rebalance Portfolio
1. Select a portfolio
2. Click **REBALANCE ALLOCATION** (right panel)
3. Modal opens (placeholder - ready for implementation)

### Run Stress Test
1. Select a portfolio
2. Scroll to **STRESS SCENARIOS** (right panel)
3. Click **RUN TEST** on any scenario
4. View impact alert with loss, max DD, recovery days

### Pause/Resume Portfolio
1. Select a portfolio
2. Click **⏸ PAUSE PORTFOLIO** or **▶ RESUME PORTFOLIO**
3. Status updates immediately

### Delete Portfolio
1. Select a portfolio
2. Click **🗑 DELETE PORTFOLIO**
3. Confirm deletion

## Test F8 (Data & Models)

### Browse Model Architectures
1. Click **F8 - DATA & MODELS** tab
2. Click **BROWSE MODEL ARCHITECTURES** (right panel)
3. View 15+ model types:
   - Time-series: TFT, Informer, PatchTST, LSTM, GRU
   - Representation: VAE, Denoising AE, Contrastive
   - Graph: GCN, GAT, Temporal GNN
   - Reinforcement: PPO, SAC
   - NLP: FinBERT, Longformer
   - Tabular: TabNet, FT-Transformer
4. Click on any model to view:
   - Strengths & weaknesses
   - Data requirements
   - Hyperparameters
   - GPU requirements
5. Click **USE THIS MODEL FOR TRAINING**

### Train New Model
1. Click **TRAIN NEW MODEL** (right panel)
2. Select:
   - **Model Architecture**: Choose from dropdown
   - **Training Dataset**: Choose from available datasets
   - **Epochs**: 100
   - **Batch Size**: 64
   - **Learning Rate**: 0.001
   - **Validation Split**: 0.2
3. Click **START TRAINING**
4. View training job in left panel

### Import Dataset
1. Click **IMPORT DATASET** (right panel)
2. Fill in:
   - **Dataset Name**: "GBPUSD_2024_CLEAN"
   - **Records**: 1000000
   - **Features**: 32
   - **Quality Score**: 98.5
   - **Size**: "2.4 GB"
   - **Date Range**: "2024-01-01 to 2024-12-31"
   - **Assets**: "GBPUSD, EURGBP"
3. Click **IMPORT DATASET**

### Validate Model
1. Select a deployed model (left panel)
2. Click **RUN VALIDATION** (right panel)
3. View validation metrics:
   - Temporal Continuity
   - Regime Separability
   - Similarity Coherence
   - Prediction Accuracy
   - Precision, Recall, F1 Score
   - Confusion Matrix

### Manage Model Status
1. Select a deployed model
2. Use buttons in right panel:
   - **ACTIVATE MODEL**: Set to production
   - **MOVE TO TESTING**: Set to testing
   - **DEPRECATE MODEL**: Mark as deprecated
   - **DELETE MODEL**: Remove model

### Export Dataset
1. Select a dataset (left panel)
2. Click **EXPORT DATASET** (right panel)
3. Choose format (CSV, PARQUET, HDF5)

## Test Backend Endpoints Directly

### Portfolio Endpoints
```bash
# List portfolios
curl http://localhost:8000/portfolios/list

# Get portfolio
curl http://localhost:8000/portfolios/PORT-001

# Create portfolio
curl -X POST http://localhost:8000/portfolios/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "API TEST PORTFOLIO",
    "baseCurrency": "USD",
    "initialCapital": 100000,
    "mode": "PAPER",
    "status": "ACTIVE",
    "strategyAllocations": [],
    "allocationModel": "EQUAL_WEIGHT",
    "rebalanceFrequency": "WEEKLY",
    "turnoverConstraint": 20
  }'

# Update portfolio
curl -X PUT http://localhost:8000/portfolios/PORT-001 \
  -H "Content-Type: application/json" \
  -d '{"status": "PAUSED"}'

# Delete portfolio
curl -X DELETE http://localhost:8000/portfolios/PORT-001

# List risk limits
curl http://localhost:8000/portfolios/PORT-001/risk-limits

# Run stress test
curl -X POST http://localhost:8000/stress-test/run \
  -H "Content-Type: application/json" \
  -d '{"portfolioId": "PORT-001", "scenarioId": "STRESS-001"}'
```

### Data & Models Endpoints
```bash
# List deployed models
curl http://localhost:8000/models/deployed

# Deploy model
curl -X POST http://localhost:8000/models/deploy/JOB001

# Update model status
curl -X PUT http://localhost:8000/models/TCN_V1.2/status \
  -H "Content-Type: application/json" \
  -d '{"status": "ACTIVE"}'

# Validate model
curl -X POST http://localhost:8000/models/TCN_V1.2/validate

# List datasets
curl http://localhost:8000/datasets/list

# Create dataset
curl -X POST http://localhost:8000/datasets/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "TEST_DATASET",
    "records": 1000000,
    "features": 32,
    "quality": 98.5,
    "size": "2.4 GB",
    "dateRange": "2024-01-01 to 2024-12-31",
    "assets": ["EURUSD"],
    "status": "READY"
  }'

# List training jobs
curl http://localhost:8000/training/jobs

# Start training
curl -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "lstm",
    "dataset_id": "DS001",
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "validation_split": 0.2
  }'

# Get training job status
curl http://localhost:8000/training/jobs/JOB001

# Cancel training
curl -X POST http://localhost:8000/training/jobs/JOB001/cancel
```

## Test Fallback Behavior

### Simulate Backend Unavailable
1. Stop the intelligence layer backend (Ctrl+C)
2. Refresh the frontend
3. All data still loads from hardcoded fallback
4. All CRUD operations work with mock data
5. Console shows warnings: "Backend unavailable, using hardcoded data"

### Verify Fallback Data
**F5 Portfolios:**
- PORT-001: MAIN TRADING PORTFOLIO ($104,127.89)
- PORT-002: RESEARCH PORTFOLIO ($51,234.56)

**F5 Risk Limits:**
- LIMIT-001: MAX POSITION SIZE (10.5 / 15.0)
- LIMIT-002: MAX LEVERAGE (2.1 / 3.0)
- LIMIT-003: MAX DAILY LOSS (1.2 / 5.0)

**F5 Stress Scenarios:**
- STRESS-001: 2008 FINANCIAL CRISIS
- STRESS-002: COVID-19 CRASH

**F8 Deployed Models:**
- TCN_V1.2: TCN EMBEDDING MODEL (92.3% accuracy)
- HMM_V2.0: HMM REGIME DETECTOR (87.5% accuracy)
- VAE_V1.0: VAE EMBEDDING (89.1% accuracy)

**F8 Datasets:**
- DS001: EURUSD_2020-2023_CLEAN (1.25M records)
- DS002: MULTI_ASSET_2022-2024 (3.45M records)
- DS003: CRISIS_SCENARIOS_2008-2020 (450K records)

## Verify Build

```bash
cd frontend
npm run build
```

Expected output:
```
✓ 1621 modules transformed.
dist/index.html                   0.45 kB │ gzip:  0.29 kB
dist/assets/index-CeOBWHge.css   92.27 kB │ gzip: 14.91 kB
dist/assets/index-CGTDZ4xQ.js   404.04 kB │ gzip: 95.07 kB
✓ built in 4.39s
```

## Troubleshooting

### Frontend won't start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Backend won't start
```bash
cd intelligence-layer
poetry install
poetry run uvicorn intelligence_layer.main:app --reload
```

### Port already in use
```bash
# Frontend (default 5173)
npm run dev -- --port 5174

# Backend (default 8000)
poetry run uvicorn intelligence_layer.main:app --reload --port 8001
```

### TypeScript errors
```bash
cd frontend
npm run build
# Check output for specific errors
```

## Next Steps

1. **Implement Remaining Modals**: Complete the 5 placeholder modals in PortfolioModals.tsx
2. **Database Integration**: Replace mock data with actual database queries
3. **Real-time Updates**: Add WebSocket support for live data
4. **Enhanced Validation**: Add form validation and error handling
5. **Testing**: Add unit tests and integration tests

## Support

For issues or questions:
1. Check `F5_F8_BACKEND_INTEGRATION_COMPLETE.md` for detailed documentation
2. Review console logs for error messages
3. Verify all services are running
4. Check network tab in browser DevTools for API calls
