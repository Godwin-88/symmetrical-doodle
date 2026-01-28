# Vercel Environment Variables Update Script
# Run this to connect your frontend to the Railway backend

Write-Host "Updating Vercel environment variables..." -ForegroundColor Yellow

# Install Vercel CLI if not already installed
# npm install -g vercel

# Set the backend API URL to your Railway deployment
vercel env add VITE_API_BASE_URL production
# When prompted, enter: https://upbeat-enchantment-production.up.railway.app

vercel env add VITE_EXECUTION_API_URL production  
# When prompted, enter: https://upbeat-enchantment-production.up.railway.app

vercel env add VITE_WS_URL production
# When prompted, enter: wss://upbeat-enchantment-production.up.railway.app

# Redeploy to apply the new environment variables
vercel --prod

Write-Host "Frontend will be redeployed with new backend URLs!" -ForegroundColor Green
Write-Host "Your connected application will be available at: https://algo-terminal.vercel.app" -ForegroundColor Cyan