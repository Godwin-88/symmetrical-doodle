# Common Issues and Solutions

This document consolidates common issues you might encounter and their solutions.

## Quick Diagnostics

### System Health Check
```bash
# Test all connections
python scripts/test-neo4j-aura.py
python scripts/test-deriv-connection.py

# Check service status
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:5173
```

## Database Issues

### Neo4j Aura Connection Failed
**Problem**: "Unable to retrieve routing information"
**Solution**: 
1. Go to https://console.neo4j.io/
2. Select instance `33c17f32`
3. Add your IP to allowlist (use `0.0.0.0/0` for development)
4. Wait 1-2 minutes and retry

### PostgreSQL Connection Issues
**Problem**: "Connection refused on port 5432"
**Solution**:
```bash
# Check if Docker is running
docker ps

# Restart database services
docker-compose down
docker-compose up -d
```

### Redis Connection Issues
**Problem**: "Redis connection failed"
**Solution**:
```bash
# Check Redis status
docker exec -it <redis-container> redis-cli ping

# Should return PONG
```

## Service Startup Issues

### Port Already in Use
**Problem**: "Port 5173 already in use"
**Solution**:
```bash
# Find process using port
netstat -ano | findstr :5173

# Kill the process (Windows)
taskkill /PID <process-id> /F

# Kill the process (Linux/Mac)
kill -9 <process-id>
```

### Python Dependencies Missing
**Problem**: "ModuleNotFoundError"
**Solution**:
```bash
cd intelligence-layer
pip install -e .

# Or install specific packages
pip install fastapi uvicorn sqlalchemy neo4j redis
```

### Rust Compilation Errors
**Problem**: "cargo build failed"
**Solution**:
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

## Frontend Issues

### Build Failures
**Problem**: "npm run build failed"
**Solution**:
```bash
cd frontend

# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Try building again
npm run build
```

### TypeScript Errors
**Problem**: "Type errors in frontend"
**Solution**:
```bash
# Check TypeScript version
npx tsc --version

# Run type checking
npm run type-check

# Fix common issues
npm run lint --fix
```

## Integration Issues

### Deriv API Connection
**Problem**: "Deriv WebSocket connection failed"
**Solution**:
1. Check API token in `.env`
2. Verify app ID is correct (118029)
3. Ensure demo mode is enabled
4. Check network connectivity

### External API Rate Limits
**Problem**: "Too many requests"
**Solution**:
1. Implement exponential backoff
2. Add request queuing
3. Use caching for repeated requests
4. Check API documentation for limits

## Performance Issues

### Slow Database Queries
**Problem**: "Queries taking too long"
**Solution**:
1. Add database indexes
2. Optimize query patterns
3. Use connection pooling
4. Monitor query execution plans

### High Memory Usage
**Problem**: "System running out of memory"
**Solution**:
1. Increase Docker memory limits
2. Optimize data structures
3. Implement pagination
4. Use streaming for large datasets

### Slow Frontend Loading
**Problem**: "Frontend takes long to load"
**Solution**:
1. Enable gzip compression
2. Optimize bundle size
3. Implement code splitting
4. Use CDN for static assets

## Development Issues

### Hot Reload Not Working
**Problem**: "Changes not reflected"
**Solution**:
```bash
# Restart development server
npm run dev

# Clear browser cache
Ctrl+Shift+R (or Cmd+Shift+R on Mac)

# Check file watchers
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
```

### Environment Variables Not Loading
**Problem**: "Config values undefined"
**Solution**:
1. Check `.env` file exists
2. Verify variable names match
3. Restart services after changes
4. Check for typos in variable names

## Deployment Issues

### Docker Container Crashes
**Problem**: "Container exits immediately"
**Solution**:
```bash
# Check container logs
docker logs <container-name>

# Run container interactively
docker run -it <image-name> /bin/bash

# Check resource limits
docker stats
```

### SSL/TLS Certificate Issues
**Problem**: "Certificate verification failed"
**Solution**:
1. Update certificates: `pip install --upgrade certifi`
2. Check system time is correct
3. Verify certificate chain
4. Use insecure mode for development only

## Monitoring and Debugging

### Enable Debug Logging
```bash
# Python services
export INTELLIGENCE_DEBUG=true
export INTELLIGENCE_LOGGING__LEVEL=DEBUG

# Rust services
export RUST_LOG=debug

# Frontend
export VITE_DEBUG_MODE=true
```

### Health Check Endpoints
- Intelligence Layer: `http://localhost:8000/health`
- Execution Core: `http://localhost:8001/health`
- Simulation Engine: `http://localhost:8002/health`
- Frontend: `http://localhost:5173`

### Log Locations
- Docker logs: `docker logs <container-name>`
- Application logs: `logs/` directory
- System logs: `/var/log/` (Linux) or Event Viewer (Windows)

## Getting Help

### Documentation
- [System Architecture](../02-architecture/system-architecture.md)
- [Component Overview](../02-architecture/component-overview.md)
- [Integration Guide](../06-development/frontend-backend-integration.md)

### Diagnostic Commands
```bash
# System information
uname -a                    # Linux/Mac
systeminfo                 # Windows

# Network connectivity
ping google.com
telnet localhost 5432

# Disk space
df -h                       # Linux/Mac
dir                         # Windows

# Memory usage
free -h                     # Linux
top                         # Linux/Mac
tasklist                    # Windows
```

### Support Checklist
Before seeking help, please provide:
1. Operating system and version
2. Error messages (full stack trace)
3. Steps to reproduce the issue
4. System logs and diagnostic output
5. Configuration files (sanitized)

---

**Last Updated**: January 2026  
**For additional help**: Check specific troubleshooting guides in this section