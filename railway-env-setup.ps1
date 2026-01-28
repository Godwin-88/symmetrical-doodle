# Railway Environment Variables Setup Script
# Run this script to set all environment variables at once

railway variables set `
  --variables "DATABASE_URL=postgresql://postgres:-&q6+A5u84f+65H@db.kajjmtzpdfybslxcidws.supabase.co:5432/postgres" `
  --variables "NEO4J_URI=neo4j+s://33c17f32.databases.neo4j.io" `
  --variables "NEO4J_USERNAME=neo4j" `
  --variables "NEO4J_PASSWORD=W5VICDGU9JtBNpgmHvSIZnHjZh-SMmS9r4zyni-Ewfg" `
  --variables "NEO4J_DATABASE=neo4j" `
  --variables "REDIS_URL=redis://localhost:6379" `
  --variables "INTELLIGENCE_API_HOST=0.0.0.0" `
  --variables "INTELLIGENCE_API_PORT=8000" `
  --variables "INTELLIGENCE_DEBUG=true" `
  --variables "INTELLIGENCE_LOGGING__LEVEL=INFO" `
  --variables "DERIV_APP_ID=118029" `
  --variables "DERIV_API_TOKEN=gxF5pHUCgjDTOGI" `
  --variables "DERIV_WEBSOCKET_URL=wss://ws.derivws.com/websockets/v3?app_id=118029" `
  --variables "DERIV_DEMO_MODE=true" `
  --variables "DERIV_MAX_POSITION_SIZE=1.0" `
  --variables "DERIV_MAX_DAILY_TRADES=50" `
  --variables "DERIV_MAX_DAILY_LOSS=1000.0" `
  --variables "TRADING_RISK_LIMITS__MAX_POSITION_SIZE=100000.0" `
  --variables "TRADING_RISK_LIMITS__MAX_DRAWDOWN=0.05" `
  --variables "TRADING_RISK_LIMITS__MAX_DAILY_LOSS=10000.0" `
  --variables "TRADING_LOGGING__LEVEL=info" `
  --variables "TRADING_LOGGING__FORMAT=json" `
  --variables "VITE_API_BASE_URL=https://your-railway-app.railway.app" `
  --variables "VITE_REAL_TIME_UPDATES=true" `
  --variables "VITE_DEBUG_MODE=true"

Write-Host "All environment variables have been set successfully!" -ForegroundColor Green