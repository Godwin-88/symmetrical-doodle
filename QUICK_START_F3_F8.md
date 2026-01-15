# Quick Start: F3 (Intelligence) & F8 (Data & Models)

## Running the Application

```bash
cd frontend
npm run dev
```

Navigate to the Intelligence (F3) or Data & Models (F8) tabs.

---

## F3: INTELLIGENCE - Quick Actions

### 1. Browse ML Models
**Button**: "BROWSE MODELS (18)"
- View 18 production-grade ML architectures
- Filter by category, use case, production readiness
- Select model for embedding configuration

### 2. Configure Embeddings
**Button**: "CONFIGURE EMBEDDINGS"
- Create new embedding configs (Asset ID, Window Size, Features, Model Type)
- Edit existing configs
- Delete configs
- Generate embeddings

### 3. Configure Regimes
**Button**: "CONFIGURE REGIMES"
- Create custom regime definitions
- Set volatility, trend, liquidity parameters
- Define duration constraints
- Edit/delete regimes

### 4. Train Regime Model
**Button**: "TRAIN MODEL"
- Train HMM on mock historical data
- View training progress
- Refresh regime detections

### 5. Run Graph Analytics
**Button**: "CONFIGURE GRAPH"
- PageRank (asset importance)
- Louvain (community detection)
- Betweenness (systemic risk)

---

## F8: DATA & MODELS - Quick Actions

### 1. Browse Model Architectures
**Button**: "BROWSE MODEL ARCHITECTURES (18)"
- View all available ML models
- See strengths, weaknesses, requirements
- Select model for training

### 2. Train New Model
**Button**: "TRAIN NEW MODEL"
- Select model architecture
- Select training dataset
- Configure hyperparameters (epochs, batch size, learning rate)
- Start training job

### 3. Import Dataset
**Button**: "IMPORT DATASET"
- Add new training dataset
- Specify records, features, quality
- Define date range and assets

### 4. Run Validation
**Button**: "RUN VALIDATION"
- Validate deployed model
- View 7 validation metrics
- See confusion matrix
- Export results

### 5. Manage Models
**Buttons**: "ACTIVATE", "MOVE TO TESTING", "DEPRECATE", "DELETE"
- Change model status
- Deploy from training jobs
- Remove deprecated models

---

## Mock Data Available

### F3 (Intelligence)
- 2 embedding configs (EURUSD, GBPUSD)
- 3 regime configs (LOW_VOL_TRENDING, HIGH_VOL_RANGING, CRISIS)
- Live regime detections
- Graph features

### F8 (Data & Models)
- 3 deployed models (TCN, HMM, VAE)
- 3 training datasets (EURUSD, Multi-Asset, Crisis)
- 1 active training job (VAE, 45/100 epochs)
- Validation metrics with confusion matrix

---

## Key Differences

**F3 = Runtime Operations**
- Generate embeddings NOW
- Detect regimes NOW
- Analyze correlations NOW
- For traders and analysts

**F8 = Development Operations**
- Train models for LATER
- Validate performance BEFORE deployment
- Manage datasets for TRAINING
- For ML engineers and data scientists

---

## All Buttons Work Offline

Both F3 and F8 have automatic mock data fallback:
- ✅ No backend required
- ✅ All CRUD operations functional
- ✅ All buttons clickable
- ✅ All modals working
- ✅ Status indicators show "OFFLINE" mode

---

## Build & Deploy

```bash
# Development
npm run dev

# Production build
npm run build

# Output: 358.64 KB (gzipped: 86.98 KB)
```

---

## Documentation

- `F3_VS_F8_COMPARISON.md` - Detailed comparison
- `F8_IMPLEMENTATION_COMPLETE.md` - Full implementation details
- `INTELLIGENCE_FEATURES_GUIDE.md` - F3 features
- `ML_MODELS_COMPLETE.md` - Model registry
- `STRATEGIES_COMPLETE.md` - Trading strategies

---

## Support

All 33 buttons are functional with mock data. The system works completely offline and is ready for backend integration.
