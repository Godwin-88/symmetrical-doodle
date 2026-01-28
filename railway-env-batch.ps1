# Railway Environment Variables - Individual Commands
# This script sets each variable individually for better error handling

Write-Host "Setting Railway environment variables..." -ForegroundColor Yellow

# Database Configuration
railway variables set "DATABASE_URL=postgresql://postgres:-&q6+A5u84f+65H@db.kajjmtzpdfybslxcidws.supabase.co:5432/postgres"
railway variables set NEO4J_URI=neo4j+s://33c17f32.databases.neo4j.io
railway variables set NEO4J_USERNAME=neo4j
railway variables set NEO4J_PASSWORD=W5VICDGU9JtBNpgmHvSIZnHjZh-SMmS9r4zyni-Ewfg
railway variables set NEO4J_DATABASE=neo4j

# Redis Configuration
railway variables set REDIS_URL=redis://localhost:6379

# Intelligence Layer Configuration
railway variables set INTELLIGENCE_API_HOST=0.0.0.0
railway variables set INTELLIGENCE_API_PORT=8000
railway variables set INTELLIGENCE_DEBUG=true
railway variables set INTELLIGENCE_LOGGING__LEVEL=INFO

# Deriv API Configuration
railway variables set DERIV_APP_ID=118029
railway variables set DERIV_API_TOKEN=gxF5pHUCgjDTOGI
railway variables set "DERIV_WEBSOCKET_URL=wss://ws.derivws.com/websockets/v3?app_id=118029"
railway variables set DERIV_DEMO_MODE=true
railway variables set DERIV_MAX_POSITION_SIZE=1.0
railway variables set DERIV_MAX_DAILY_TRADES=50
railway variables set DERIV_MAX_DAILY_LOSS=1000.0

# Risk Management
railway variables set TRADING_RISK_LIMITS__MAX_POSITION_SIZE=100000.0
railway variables set TRADING_RISK_LIMITS__MAX_DRAWDOWN=0.05
railway variables set TRADING_RISK_LIMITS__MAX_DAILY_LOSS=10000.0

# Logging Configuration
railway variables set TRADING_LOGGING__LEVEL=info
railway variables set TRADING_LOGGING__FORMAT=json

# Frontend Configuration (will be updated after deployment)
railway variables set VITE_API_BASE_URL=https://your-railway-app.railway.app
railway variables set VITE_REAL_TIME_UPDATES=true
railway variables set VITE_DEBUG_MODE=true

Write-Host "All environment variables have been set successfully!" -ForegroundColor Green
Write-Host "You can verify them with: railway variables" -ForegroundColor Cyan