# NautilusTrader Integration Troubleshooting

## Overview

This guide provides comprehensive troubleshooting information for the NautilusTrader integration, covering common issues, diagnostic procedures, and resolution steps.

## Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [Common Issues](#common-issues)
3. [Service-Specific Troubleshooting](#service-specific-troubleshooting)
4. [Performance Issues](#performance-issues)
5. [Data Synchronization Problems](#data-synchronization-problems)
6. [Risk Management Issues](#risk-management-issues)
7. [UI-Specific Problems](#ui-specific-problems)
8. [Logging and Monitoring](#logging-and-monitoring)
9. [Emergency Procedures](#emergency-procedures)
10. [Support Escalation](#support-escalation)

## Quick Diagnostic Checklist

### System Health Check

Run this quick checklist to identify the scope of issues:

```bash
# 1. Check service status
curl -s http://localhost:8000/health | jq '.'

# 2. Check database connectivity
psql -h localhost -U nautilus -d nautilus_db -c "SELECT 1;"

# 3. Check Redis connectivity
redis-cli ping

# 4. Check log files for errors
tail -n 50 logs/nautilus-integration.log | grep ERROR

# 5. Check system resources
htop
df -h
free -m

# 6. Check network connectivity
ping google.com
telnet localhost 8000
```

### Service Status Verification

| Service | Check Command | Expected Result |
|---------|---------------|-----------------|
| Main API | `curl http://localhost:8000/health` | `{"status": "healthy"}` |
| F8 Integration | `curl http://localhost:8000/health/f8` | `{"status": "connected"}` |
| Risk Gate | `curl http://localhost:8000/health/risk` | `{"status": "active"}` |
| Database | `pg_isready -h localhost -p 5432` | `accepting connections` |
| Redis | `redis-cli ping` | `PONG` |
| NautilusTrader | Check process list | Process running |

## Common Issues

### 1. Service Connection Failures

#### Symptom
- Services showing "disconnected" status in UI
- API calls returning connection errors
- Timeout errors in logs

#### Possible Causes
- Network connectivity issues
- Service not running
- Configuration errors
- Firewall blocking connections
- Port conflicts

#### Diagnostic Steps
```bash
# Check if services are running
ps aux | grep nautilus
systemctl status nautilus-integration

# Check port availability
netstat -tulpn | grep :8000
lsof -i :8000

# Test network connectivity
telnet localhost 8000
curl -v http://localhost:8000/health

# Check firewall rules
sudo ufw status
sudo iptables -L
```

#### Resolution Steps
1. **Restart Services**:
   ```bash
   systemctl restart nautilus-integration
   systemctl restart redis
   systemctl restart postgresql
   ```

2. **Check Configuration**:
   ```bash
   # Verify configuration files
   cat config/.env | grep -v PASSWORD
   
   # Validate configuration
   python -c "from nautilus_integration.core.config import load_config; print(load_config())"
   ```

3. **Fix Port Conflicts**:
   ```bash
   # Find process using port
   sudo lsof -i :8000
   
   # Kill conflicting process
   sudo kill -9 <PID>
   ```

### 2. Authentication and Authorization Errors

#### Symptom
- Login failures
- "Unauthorized" errors
- Token expiration issues

#### Diagnostic Steps
```bash
# Check authentication service
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'

# Verify token validity
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/user/profile

# Check user database
psql -d nautilus_db -c "SELECT id, email, is_active FROM users WHERE email='test@example.com';"
```

#### Resolution Steps
1. **Reset User Password**:
   ```bash
   python scripts/reset_password.py --email test@example.com
   ```

2. **Clear Token Cache**:
   ```bash
   redis-cli FLUSHDB
   ```

3. **Recreate User**:
   ```bash
   python scripts/create_user.py --email test@example.com --password newpassword
   ```

### 3. Database Connection Issues

#### Symptom
- Database connection errors
- Slow query performance
- Transaction timeout errors

#### Diagnostic Steps
```bash
# Check database status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U nautilus -d nautilus_db -c "SELECT version();"

# Check active connections
psql -d nautilus_db -c "SELECT count(*) FROM pg_stat_activity;"

# Check for long-running queries
psql -d nautilus_db -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"
```

#### Resolution Steps
1. **Restart Database**:
   ```bash
   sudo systemctl restart postgresql
   ```

2. **Kill Long-Running Queries**:
   ```bash
   psql -d nautilus_db -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval '10 minutes';"
   ```

3. **Optimize Database**:
   ```bash
   psql -d nautilus_db -c "VACUUM ANALYZE;"
   psql -d nautilus_db -c "REINDEX DATABASE nautilus_db;"
   ```

### 4. Memory and Resource Issues

#### Symptom
- Out of memory errors
- High CPU usage
- Slow response times

#### Diagnostic Steps
```bash
# Check memory usage
free -m
ps aux --sort=-%mem | head -10

# Check CPU usage
top -o %CPU
htop

# Check disk space
df -h
du -sh /var/log/*

# Check file descriptors
lsof | wc -l
ulimit -n
```

#### Resolution Steps
1. **Increase Memory Limits**:
   ```bash
   # Edit systemd service file
   sudo systemctl edit nautilus-integration
   
   # Add:
   [Service]
   MemoryLimit=2G
   ```

2. **Clean Up Logs**:
   ```bash
   # Rotate logs
   sudo logrotate -f /etc/logrotate.d/nautilus-integration
   
   # Clean old logs
   find /var/log -name "*.log" -mtime +7 -delete
   ```

3. **Optimize Application**:
   ```bash
   # Restart with optimized settings
   export PYTHONOPTIMIZE=1
   systemctl restart nautilus-integration
   ```

## Service-Specific Troubleshooting

### F8 Integration Orchestrator

#### Common Issues
- Order processing failures
- Position synchronization errors
- Risk check timeouts

#### Diagnostic Commands
```bash
# Check orchestrator status
curl http://localhost:8000/api/orchestrator/status

# View recent orders
curl http://localhost:8000/api/orders?limit=10

# Check risk metrics
curl http://localhost:8000/api/risk/metrics
```

#### Log Analysis
```bash
# Filter orchestrator logs
grep "F8IntegrationOrchestrator" logs/nautilus-integration.log | tail -50

# Check for errors
grep "ERROR.*orchestrator" logs/nautilus-integration.log

# Monitor real-time logs
tail -f logs/nautilus-integration.log | grep orchestrator
```

### Live Trading Risk Gate

#### Common Issues
- Risk limit violations
- Validation failures
- Performance bottlenecks

#### Diagnostic Commands
```bash
# Check risk gate status
curl http://localhost:8000/api/risk/status

# View risk limits
curl http://localhost:8000/api/risk/limits

# Check recent validations
curl http://localhost:8000/api/risk/validations?limit=20
```

#### Configuration Validation
```python
# Validate risk configuration
python -c "
from nautilus_integration.services.live_trading_risk_gate import LiveTradingRiskGate
from nautilus_integration.core.config import load_config

config = load_config()
risk_gate = LiveTradingRiskGate(config)
print('Risk limits:', risk_gate.get_risk_limits())
"
```

### Strategy Translation Service

#### Common Issues
- Strategy conversion errors
- Parameter validation failures
- Performance degradation

#### Diagnostic Commands
```bash
# Check strategy service
curl http://localhost:8000/api/strategies/status

# List active strategies
curl http://localhost:8000/api/strategies

# Validate specific strategy
curl http://localhost:8000/api/strategies/validate -d '{"strategy_id": "test-strategy"}'
```

## Performance Issues

### Slow API Response Times

#### Investigation Steps
1. **Identify Slow Endpoints**:
   ```bash
   # Analyze access logs
   awk '{print $7, $10}' /var/log/nginx/access.log | sort | uniq -c | sort -nr
   
   # Check response times
   grep "response_time" logs/nautilus-integration.log | awk '{print $NF}' | sort -n
   ```

2. **Database Query Analysis**:
   ```sql
   -- Enable query logging
   ALTER SYSTEM SET log_statement = 'all';
   ALTER SYSTEM SET log_min_duration_statement = 1000;
   SELECT pg_reload_conf();
   
   -- Find slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   ```

3. **Application Profiling**:
   ```python
   # Add profiling to critical functions
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Your code here
   
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative').print_stats(10)
   ```

#### Optimization Steps
1. **Database Optimization**:
   ```sql
   -- Add missing indexes
   CREATE INDEX CONCURRENTLY idx_orders_timestamp ON orders(timestamp);
   CREATE INDEX CONCURRENTLY idx_positions_instrument ON positions(instrument);
   
   -- Update statistics
   ANALYZE;
   ```

2. **Application Optimization**:
   ```python
   # Enable connection pooling
   DATABASE_CONFIG = {
       'pool_size': 20,
       'max_overflow': 30,
       'pool_timeout': 30,
       'pool_recycle': 3600
   }
   ```

3. **Caching Implementation**:
   ```python
   # Add Redis caching
   from redis import Redis
   import json
   
   redis_client = Redis(host='localhost', port=6379, db=0)
   
   def get_cached_data(key):
       data = redis_client.get(key)
       return json.loads(data) if data else None
   
   def set_cached_data(key, data, ttl=300):
       redis_client.setex(key, ttl, json.dumps(data))
   ```

### High Memory Usage

#### Investigation Steps
```bash
# Memory profiling
python -m memory_profiler scripts/profile_memory.py

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python -m nautilus_integration.main

# Monitor memory over time
while true; do
    ps aux | grep nautilus | awk '{print $6}' | head -1
    sleep 60
done
```

#### Resolution Steps
1. **Optimize Data Structures**:
   ```python
   # Use generators instead of lists
   def get_orders():
       for order in Order.objects.all():
           yield order
   
   # Clear unused variables
   del large_data_structure
   import gc; gc.collect()
   ```

2. **Implement Memory Limits**:
   ```python
   import resource
   
   # Set memory limit (in bytes)
   resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, -1))
   ```

## Data Synchronization Problems

### Position Synchronization Issues

#### Symptoms
- Position discrepancies between systems
- Sync failures in logs
- Inconsistent portfolio values

#### Diagnostic Steps
```bash
# Check sync status
curl http://localhost:8000/api/sync/status

# Compare positions
curl http://localhost:8000/api/positions/compare

# View sync history
curl http://localhost:8000/api/sync/history?limit=20
```

#### Manual Synchronization
```python
# Force position sync
python scripts/force_position_sync.py

# Reconcile specific instrument
python scripts/reconcile_positions.py --instrument EURUSD

# Full portfolio reconciliation
python scripts/full_reconciliation.py --dry-run
```

### Data Quality Issues

#### Common Problems
- Stale market data
- Missing price updates
- Incorrect calculations

#### Validation Scripts
```python
# Data quality check
python scripts/data_quality_check.py

# Price feed validation
python scripts/validate_price_feeds.py

# Calculate position values
python scripts/recalculate_positions.py
```

## Risk Management Issues

### Risk Limit Violations

#### Investigation Steps
```bash
# Check current risk metrics
curl http://localhost:8000/api/risk/current

# View violation history
curl http://localhost:8000/api/risk/violations

# Check limit configurations
curl http://localhost:8000/api/risk/limits
```

#### Emergency Risk Controls
```python
# Emergency position closure
python scripts/emergency_close_positions.py

# Activate kill switch
python scripts/activate_kill_switch.py

# Reset risk limits
python scripts/reset_risk_limits.py --emergency
```

### Risk Calculation Errors

#### Diagnostic Steps
```python
# Validate risk calculations
python scripts/validate_risk_calculations.py

# Recalculate VaR
python scripts/recalculate_var.py

# Check correlation matrices
python scripts/check_correlations.py
```

## UI-Specific Problems

### Frontend Connection Issues

#### Symptoms
- UI showing "disconnected" status
- WebSocket connection failures
- Real-time updates not working

#### Diagnostic Steps
```bash
# Check WebSocket connections
netstat -an | grep :8080

# Test WebSocket endpoint
wscat -c ws://localhost:8080/ws

# Check browser console for errors
# Open browser developer tools and check console
```

#### Resolution Steps
1. **Restart Frontend Services**:
   ```bash
   cd frontend
   npm run build
   npm run start
   ```

2. **Clear Browser Cache**:
   - Hard refresh (Ctrl+F5)
   - Clear browser cache and cookies
   - Disable browser extensions

3. **Check CORS Configuration**:
   ```python
   # Update CORS settings
   CORS_ALLOWED_ORIGINS = [
       "http://localhost:3000",
       "https://your-domain.com"
   ]
   ```

### UI Performance Issues

#### Investigation Steps
```javascript
// Browser performance monitoring
console.time('page-load');
window.addEventListener('load', () => {
    console.timeEnd('page-load');
});

// Memory usage monitoring
setInterval(() => {
    if (performance.memory) {
        console.log('Memory usage:', performance.memory.usedJSHeapSize / 1024 / 1024, 'MB');
    }
}, 5000);
```

#### Optimization Steps
1. **Code Splitting**:
   ```javascript
   // Lazy load components
   const LazyComponent = React.lazy(() => import('./LazyComponent'));
   ```

2. **Optimize Bundle Size**:
   ```bash
   # Analyze bundle
   npm run build -- --analyze
   
   # Remove unused dependencies
   npm prune
   ```

## Logging and Monitoring

### Log Analysis

#### Key Log Files
- `/var/log/nautilus-integration/main.log` - Main application logs
- `/var/log/nautilus-integration/error.log` - Error logs
- `/var/log/nautilus-integration/performance.log` - Performance metrics
- `/var/log/nginx/access.log` - Web server access logs
- `/var/log/postgresql/postgresql.log` - Database logs

#### Log Analysis Commands
```bash
# Find errors in last hour
find /var/log -name "*.log" -mmin -60 -exec grep -l "ERROR" {} \;

# Count error types
grep "ERROR" logs/nautilus-integration.log | awk '{print $4}' | sort | uniq -c

# Monitor real-time errors
tail -f logs/nautilus-integration.log | grep --color=always "ERROR\|CRITICAL"

# Extract performance metrics
grep "PERFORMANCE" logs/nautilus-integration.log | awk '{print $NF}' | sort -n
```

### Monitoring Setup

#### Prometheus Metrics
```python
# Add custom metrics
from prometheus_client import Counter, Histogram, Gauge

order_counter = Counter('orders_total', 'Total orders processed')
response_time = Histogram('response_time_seconds', 'Response time')
active_positions = Gauge('active_positions', 'Number of active positions')
```

#### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Nautilus Integration Monitoring",
    "panels": [
      {
        "title": "Order Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(orders_total[5m])"
          }
        ]
      },
      {
        "title": "Response Times",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, response_time_seconds)"
          }
        ]
      }
    ]
  }
}
```

## Emergency Procedures

### System Emergency Stop

#### When to Use
- Critical system failures
- Risk limit breaches
- Data corruption detected
- Security incidents

#### Emergency Stop Procedure
```bash
# 1. Activate kill switch
curl -X POST http://localhost:8000/api/emergency/kill-switch

# 2. Stop all trading
python scripts/emergency_stop_trading.py

# 3. Close all positions (if safe)
python scripts/emergency_close_positions.py --confirm

# 4. Notify stakeholders
python scripts/send_emergency_notification.py

# 5. Create incident report
python scripts/create_incident_report.py
```

### Data Recovery Procedures

#### Database Recovery
```bash
# 1. Stop application
systemctl stop nautilus-integration

# 2. Restore from backup
pg_restore -d nautilus_db /backups/latest_backup.sql

# 3. Verify data integrity
python scripts/verify_data_integrity.py

# 4. Restart application
systemctl start nautilus-integration
```

#### Configuration Recovery
```bash
# 1. Restore configuration files
cp /backups/config/.env config/.env

# 2. Validate configuration
python scripts/validate_config.py

# 3. Restart services
systemctl restart nautilus-integration
```

## Support Escalation

### Escalation Levels

#### Level 1: Self-Service
- Use this troubleshooting guide
- Check system logs
- Restart services
- Apply common fixes

#### Level 2: Technical Support
- Contact: support@trading-system.com
- Provide: System logs, error messages, steps to reproduce
- Response time: 4 hours during business hours

#### Level 3: Engineering Team
- Contact: engineering@trading-system.com
- Provide: Complete system state, configuration files, database dumps
- Response time: 2 hours for critical issues

#### Level 4: Emergency Response
- Contact: emergency@trading-system.com or +1-800-EMERGENCY
- For: System-wide failures, security incidents, data corruption
- Response time: 30 minutes, 24/7

### Information to Provide

#### System Information
```bash
# Collect system information
python scripts/collect_system_info.py > system_info.txt

# Include:
# - System specifications
# - Software versions
# - Configuration files (sanitized)
# - Recent logs
# - Error messages
# - Steps to reproduce
```

#### Log Collection
```bash
# Create support bundle
python scripts/create_support_bundle.py

# This creates a zip file containing:
# - Application logs
# - System logs
# - Configuration files
# - Database schema
# - Performance metrics
```

### Contact Information

- **General Support**: support@trading-system.com
- **Technical Issues**: tech-support@trading-system.com
- **Security Issues**: security@trading-system.com
- **Emergency Hotline**: +1-800-TRADING (24/7)
- **Documentation**: https://docs.trading-system.com
- **Status Page**: https://status.trading-system.com

---

*This troubleshooting guide is maintained by the Support and Engineering teams. Last updated: January 2026*