# Quick Start - ML Models Offline Support

## 🚀 What's New

All 18 production-grade financial ML models are now available in the frontend **without requiring the backend to be running**. The Intelligence page works seamlessly offline and automatically connects to real algorithms when the backend starts.

---

## 📋 Quick Reference

### Available Models (18 Total)

#### Time-Series Forecasting (5)
1. **TFT** - Temporal Fusion Transformer (state-of-the-art)
2. **Informer** - Efficient long-sequence forecasting
3. **PatchTST** - Fast and efficient
4. **LSTM** - Classic, still relevant
5. **GRU** - Lightweight variant

#### Representation Learning (3)
6. **VAE** - Variational Autoencoder (regime detection)
7. **Denoising AE** - Noise reduction
8. **Contrastive** - Self-supervised learning

#### Graph Neural Networks (3)
9. **GCN** - Graph Convolutional Network
10. **GAT** - Graph Attention Network
11. **Temporal GNN** - Time-evolving graphs

#### Reinforcement Learning (2)
12. **PPO** - Proximal Policy Optimization
13. **SAC** - Soft Actor-Critic

#### NLP Models (2)
14. **FinBERT** - Financial text analysis
15. **Longformer** - Long documents

#### Tabular Models (3)
16. **TabNet** - Interpretable tabular learning
17. **FT-Transformer** - State-of-the-art tabular

---

## 🎯 How to Use

### 1. Browse Models (Offline)
```
1. Open the application
2. Navigate to Intelligence page (F3)
3. Click "BROWSE MODELS" button
4. See all 18 models instantly (no backend needed)
```

### 2. Filter Models
```
- By Category: time_series, representation, graph, reinforcement, nlp, tabular
- By Use Case: price_forecasting, regime_detection, trading_execution, etc.
- Production Ready: Toggle to show only production-ready models
```

### 3. View Model Details
```
1. Click on any model in the list
2. See complete specifications:
   - Description
   - Strengths & Weaknesses
   - Best use cases
   - Performance characteristics
   - Resource requirements
   - Hyperparameters
```

### 4. Use a Model
```
1. Browse and select a model
2. Click "USE THIS MODEL"
3. Model is added to embedding configuration
4. Configure parameters
5. Generate embedding (uses mock data offline)
```

### 5. Connect to Backend (Optional)
```
1. Start backend server
2. Frontend automatically detects connection
3. Model configurations now connect to real algorithms
4. Training and inference use actual implementations
```

---

## 💻 For Developers

### Import Models Service
```typescript
import { 
  listModels, 
  getModelDetails, 
  recommendModels,
  getModelCategories,
  getUseCases 
} from '@/services/modelsService';
```

### List All Models
```typescript
const models = await listModels();
// Returns all 18 models (offline or online)
```

### Filter Models
```typescript
// By category
const timeSeriesModels = await listModels({ 
  category: 'time_series' 
});

// By use case
const forecastingModels = await listModels({ 
  use_case: 'price_forecasting' 
});

// Production ready only
const prodModels = await listModels({ 
  production_ready: true 
});
```

### Get Model Details
```typescript
const tft = await getModelDetails('tft');
console.log(tft.name); // "Temporal Fusion Transformer"
console.log(tft.strengths); // Array of strengths
console.log(tft.hyperparameters); // Default config
```

### Get Recommendations
```typescript
const recommendations = await recommendModels('price_forecasting');
console.log(recommendations.count); // Number of recommended models
console.log(recommendations.recommendations); // Array of models
```

### Get Categories
```typescript
const categories = await getModelCategories();
// Returns: [
//   { id: 'time_series', name: 'Time Series', count: 5 },
//   { id: 'representation', name: 'Representation', count: 3 },
//   ...
// ]
```

### Get Use Cases
```typescript
const useCases = await getUseCases();
// Returns: [
//   { id: 'price_forecasting', name: 'Price Forecasting', count: 5 },
//   { id: 'regime_detection', name: 'Regime Detection', count: 4 },
//   ...
// ]
```

---

## 🔧 Technical Details

### Offline Behavior
- All functions try backend API first
- On failure, automatically use hardcoded data
- No manual switching required
- Console warnings for debugging
- Same data structure as backend

### Online Behavior
- Backend APIs called when available
- May return additional models
- Configuration connects to real algorithms
- Training uses actual implementations
- Seamless transition from offline

### Error Handling
```typescript
try {
  const models = await listModels();
  // Use models
} catch (error) {
  // Error only thrown if model not found
  // Network errors are handled internally
}
```

---

## 📊 Model Selection Guide

### For Price Forecasting
**Recommended:** TFT, PatchTST, LSTM
- TFT: Best accuracy, needs large dataset
- PatchTST: Fast, works with less data
- LSTM: Simple, battle-tested

### For Regime Detection
**Recommended:** VAE, Contrastive, GCN
- VAE: Probabilistic, interpretable
- Contrastive: Works with unlabeled data
- GCN: Captures network effects

### For Trading Execution
**Recommended:** PPO, SAC
- PPO: Stable, industry standard
- SAC: Sample efficient, off-policy

### For Credit Risk / Fraud
**Recommended:** TabNet, FT-Transformer
- TabNet: Interpretable, regulatory compliant
- FT-Transformer: State-of-the-art accuracy

### For News Analysis
**Recommended:** FinBERT, Longformer
- FinBERT: Domain-specific, fast
- Longformer: Handles long documents

---

## 🎓 Model Characteristics

### Latency Classes
- **Low:** < 10ms inference (LSTM, GRU, PPO, SAC, TabNet)
- **Medium:** 10-100ms (Informer, GCN, GAT, FinBERT, FT-Transformer)
- **High:** > 100ms (TFT, Temporal GNN, Longformer)

### Data Requirements
- **Small:** < 1,000 samples (LSTM, GRU, GCN, GAT)
- **Medium:** 1,000-10,000 samples (PatchTST, VAE, TabNet, FT-Transformer, FinBERT)
- **Large:** > 10,000 samples (TFT, Informer, Contrastive, Temporal GNN, PPO, SAC, Longformer)

### Explainability
- **High:** TabNet, GAT (attention-based, interpretable)
- **Medium:** TFT, VAE, GCN, FinBERT, Longformer, FT-Transformer
- **Low:** Informer, PatchTST, LSTM, GRU, Denoising AE, Contrastive, Temporal GNN, PPO, SAC

### GPU Requirements
- **Required:** TFT, Informer, Contrastive, Temporal GNN, FinBERT, Longformer
- **Optional:** All others (can run on CPU)

---

## 🧪 Testing

### Test File
Open `frontend/test-models-offline.html` in browser to run interactive tests:
1. List all models
2. Filter by category
3. Get model details
4. Get recommendations
5. Get categories
6. Get use cases

### Manual Testing
```bash
# Build frontend
cd frontend
npm run build

# Start dev server
npm run dev

# Open http://localhost:5173
# Navigate to Intelligence page (F3)
# Click "BROWSE MODELS"
# Verify all 18 models appear
```

---

## 📝 Files Reference

### Core Implementation
- `frontend/src/services/modelsService.ts` - Service with hardcoded models
- `frontend/src/app/components/Intelligence.tsx` - UI integration
- `intelligence-layer/src/intelligence_layer/model_registry.py` - Backend registry

### Documentation
- `ML_MODELS_COMPLETE.md` - Complete model specifications
- `ML_MODELS_OFFLINE_COMPLETE.md` - Offline implementation details
- `TASK_COMPLETION_SUMMARY.md` - Task completion summary
- `QUICK_START_MODELS.md` - This file

### Testing
- `frontend/test-models-offline.html` - Interactive test page

---

## ✅ Status

**COMPLETE** - All 18 models hardcoded, all functions have fallback logic, build successful, no errors.

Ready to use! 🚀
