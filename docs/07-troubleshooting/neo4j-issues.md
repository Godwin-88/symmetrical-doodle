# Neo4j Aura Troubleshooting Guide

## Current Status

✅ **Configuration Complete**
- Instance ID: `33c17f32`
- Connection URI: `neo4j+s://33c17f32.databases.neo4j.io`
- Username: `neo4j`
- Password: Configured

❌ **Connection Issue Detected**
```
Unable to retrieve routing information
```

## Common Causes & Solutions

### 1. IP Address Not Whitelisted (Most Common)

**Problem**: Neo4j Aura blocks connections from non-whitelisted IPs by default.

**Solution**:
1. Go to https://console.neo4j.io/
2. Log in to your account
3. Click on your database instance (`33c17f32`)
4. Go to **"Connection"** or **"Security"** tab
5. Find **"IP Allowlist"** or **"Whitelist"** section
6. Add your current IP address OR use `0.0.0.0/0` for development (allows all IPs)
7. Save changes and wait 1-2 minutes for changes to propagate

**To find your IP**:
```powershell
# Windows
curl ifconfig.me

# Or visit
# https://whatismyipaddress.com/
```

### 2. Database Instance Paused/Stopped

**Problem**: Neo4j Aura free tier instances auto-pause after inactivity.

**Solution**:
1. Go to https://console.neo4j.io/
2. Find your instance
3. Check status - should show "Running" (green)
4. If paused, click **"Resume"** or **"Start"**
5. Wait 1-2 minutes for instance to start
6. Retry connection

### 3. Incorrect Connection URI

**Problem**: Wrong instance ID or domain.

**Solution**:
1. Go to https://console.neo4j.io/
2. Click on your database instance
3. Find **"Connection URI"** - should look like:
   ```
   neo4j+s://33c17f32.databases.neo4j.io
   ```
4. Copy the exact URI
5. Update `.env` file if different

### 4. Wrong Credentials

**Problem**: Incorrect username or password.

**Solution**:
1. Verify username is `neo4j` (default)
2. Check password in Neo4j Aura console
3. If forgotten, reset password:
   - Go to instance settings
   - Click "Reset Password"
   - Update `.env` with new password

### 5. Firewall/Network Issues

**Problem**: Corporate firewall or VPN blocking connection.

**Solution**:
- Disable VPN temporarily
- Check corporate firewall rules
- Try from different network (mobile hotspot)
- Ensure port 7687 is not blocked

### 6. SSL/TLS Certificate Issues

**Problem**: Outdated SSL certificates or Python packages.

**Solution**:
```powershell
# Update neo4j driver
pip install --upgrade neo4j

# Update certifi (SSL certificates)
pip install --upgrade certifi

# Retry connection
python scripts/test-neo4j-aura.py
```

## Step-by-Step Resolution

### Step 1: Verify Instance is Running

```powershell
# Check Neo4j Aura console
# https://console.neo4j.io/
# Status should be: "Running" (green indicator)
```

### Step 2: Whitelist Your IP

```powershell
# Get your public IP
curl ifconfig.me

# Add this IP to Neo4j Aura allowlist
# Or use 0.0.0.0/0 for development (allows all)
```

### Step 3: Test Connection

```powershell
python scripts/test-neo4j-aura.py
```

### Step 4: Check Logs

If still failing, check detailed error:

```python
# Create test-neo4j-debug.py
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to: {uri}")
print(f"User: {user}")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("✅ SUCCESS!")
    driver.close()
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
```

## Alternative: Use Local Neo4j

If Neo4j Aura continues to have issues, you can use local Neo4j via Docker:

### Switch to Local Neo4j

1. **Update `.env`**:
```bash
# Comment out Aura configuration
# INTELLIGENCE_DATABASE__NEO4J_URL=neo4j+s://33c17f32.databases.neo4j.io

# Use local Neo4j
INTELLIGENCE_DATABASE__NEO4J_URL=bolt://localhost:7687
INTELLIGENCE_DATABASE__NEO4J_USER=neo4j
INTELLIGENCE_DATABASE__NEO4J_PASSWORD=password

TRADING_DATABASE__NEO4J_URL=bolt://localhost:7687
TRADING_DATABASE__NEO4J_USER=neo4j
TRADING_DATABASE__NEO4J_PASSWORD=password
```

2. **Start Docker Neo4j**:
```powershell
docker-compose up -d neo4j
```

3. **Test Connection**:
```powershell
python scripts/test-neo4j-aura.py
```

## Quick Checklist

Before contacting support, verify:

- [ ] Neo4j Aura instance is **Running** (not paused)
- [ ] Your IP address is **whitelisted** in Aura console
- [ ] Connection URI is **correct**: `neo4j+s://33c17f32.databases.neo4j.io`
- [ ] Username is `neo4j`
- [ ] Password is correct (no extra spaces)
- [ ] Internet connection is working
- [ ] No VPN or firewall blocking connection
- [ ] Neo4j Python driver is up to date: `pip install --upgrade neo4j`

## Testing Different Configurations

### Test with Environment Variables

```powershell
# Test with NEO4J_URI variables
python scripts/test-neo4j-aura.py
```

### Test with Direct Connection

```python
from neo4j import GraphDatabase

# Direct test
driver = GraphDatabase.driver(
    "neo4j+s://33c17f32.databases.neo4j.io",
    auth=("neo4j", "W5VICDGU9JtBNpgmHvSIZnHjZh-SMmS9r4zyni-Ewfg")
)

try:
    driver.verify_connectivity()
    print("✅ Connected!")
except Exception as e:
    print(f"❌ Failed: {e}")
finally:
    driver.close()
```

## Getting Help

### Neo4j Aura Support

- **Console**: https://console.neo4j.io/
- **Documentation**: https://neo4j.com/docs/aura/
- **Community**: https://community.neo4j.com/
- **Support**: support@neo4j.com

### Check Instance Details

In Neo4j Aura console, verify:
1. **Instance Name**: Instance01
2. **Instance ID**: 33c17f32
3. **Status**: Running (green)
4. **Region**: Check which region it's hosted in
5. **Version**: Neo4j version
6. **Storage**: Available storage
7. **IP Allowlist**: Your IP should be listed

## Next Steps

Once connection is working:

1. ✅ Run `python scripts/test-neo4j-aura.py` - should pass all tests
2. ✅ Start trading system: `scripts/start-all.ps1`
3. ✅ Verify graph data is being stored
4. ✅ Monitor in Neo4j Aura console

## Summary

**Most likely issue**: IP address not whitelisted in Neo4j Aura console.

**Quick fix**:
1. Go to https://console.neo4j.io/
2. Select your instance
3. Add your IP to allowlist (or use 0.0.0.0/0 for dev)
4. Wait 1-2 minutes
5. Retry: `python scripts/test-neo4j-aura.py`

Your configuration is correct - just need to whitelist your IP address!
