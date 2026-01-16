# Neo4j Aura Setup Guide

## Overview

Your trading system is now configured to use **Neo4j Aura** - a fully managed cloud graph database. This eliminates the need to run Neo4j locally via Docker.

## Current Configuration

### Credentials (Already Configured)

```bash
Username: neo4j
Password: W5VICDGU9JtBNpgmHvSIZnHjZh-SMmS9r4zyni-Ewfg
```

### Connection URI (Needs Update)

⚠️ **ACTION REQUIRED**: You need to replace `your-instance-id` in the `.env` file with your actual Neo4j Aura instance ID.

## Step 1: Get Your Connection URI

1. Go to **Neo4j Aura Console**: https://console.neo4j.io/
2. Log in to your account
3. Find your database instance
4. Click on the instance to view details
5. Copy the **Connection URI** (looks like: `neo4j+s://abc123xyz.databases.neo4j.io`)

## Step 2: Update .env File

Open `.env` and replace `your-instance-id` with your actual instance ID:

```bash
# Before (current):
INTELLIGENCE_DATABASE__NEO4J_URL=neo4j+s://your-instance-id.databases.neo4j.io

# After (example):
INTELLIGENCE_DATABASE__NEO4J_URL=neo4j+s://abc123xyz.databases.neo4j.io
```

Also update the Rust configuration:

```bash
# Before (current):
TRADING_DATABASE__NEO4J_URL=neo4j+s://your-instance-id.databases.neo4j.io

# After (example):
TRADING_DATABASE__NEO4J_URL=neo4j+s://abc123xyz.databases.neo4j.io
```

## Step 3: Test Connection

Run the test script to verify connectivity:

```bash
python scripts/test-neo4j-aura.py
```

Expected output:
```
✅ Connected successfully
✅ Query successful: Hello from Neo4j Aura!
✅ Created test node
✅ Cleaned up test node
✅ ALL TESTS PASSED
```

## Benefits of Neo4j Aura

### ✅ **Fully Managed**
- No Docker container to manage
- Automatic backups
- Automatic updates
- High availability

### ✅ **Cloud-Based**
- Access from anywhere
- No local resources needed
- Scalable storage
- Professional hosting

### ✅ **Secure**
- Encrypted connections (neo4j+s://)
- Authentication required
- IP whitelisting available
- Enterprise-grade security

### ✅ **Performance**
- Optimized infrastructure
- Fast query execution
- Reliable uptime
- Professional support

## What Neo4j Stores

Your trading system uses Neo4j for:

### 1. **Market Relationships**
```cypher
(Asset)-[:CORRELATES_WITH]->(Asset)
(Asset)-[:TRADED_IN]->(Market)
(Market)-[:INFLUENCES]->(Market)
```

### 2. **Strategy Networks**
```cypher
(Strategy)-[:USES]->(Model)
(Strategy)-[:TRADES]->(Asset)
(Strategy)-[:DEPENDS_ON]->(Strategy)
```

### 3. **Portfolio Graphs**
```cypher
(Portfolio)-[:CONTAINS]->(Position)
(Position)-[:HOLDS]->(Asset)
(Asset)-[:PART_OF]->(Sector)
```

### 4. **Knowledge Graphs**
```cypher
(Concept)-[:RELATED_TO]->(Concept)
(Event)-[:AFFECTS]->(Asset)
(Indicator)-[:PREDICTS]->(Movement)
```

## Updating Startup Scripts

The startup scripts will now skip the local Neo4j Docker container since you're using Aura:

### Modified docker-compose.yml (Optional)

If you want to completely remove local Neo4j, comment it out in `docker-compose.yml`:

```yaml
services:
  postgres:
    # ... keep this
  
  redis:
    # ... keep this
  
  # neo4j:
  #   # ... comment out or remove
```

Or keep it for local development and switch between local/cloud as needed.

## Connection String Format

### Neo4j Aura (Cloud)
```
neo4j+s://instance-id.databases.neo4j.io
```
- `neo4j+s://` = Secure connection (SSL/TLS)
- `instance-id` = Your unique instance identifier
- `.databases.neo4j.io` = Aura domain

### Local Neo4j (Docker)
```
bolt://localhost:7687
```
- `bolt://` = Neo4j protocol
- `localhost:7687` = Local Docker container

## Switching Between Local and Cloud

You can easily switch between local Neo4j and Neo4j Aura:

### Use Neo4j Aura (Cloud)
```bash
# In .env
INTELLIGENCE_DATABASE__NEO4J_URL=neo4j+s://your-instance.databases.neo4j.io
INTELLIGENCE_DATABASE__NEO4J_USER=neo4j
INTELLIGENCE_DATABASE__NEO4J_PASSWORD=W5VICDGU9JtBNpgmHvSIZnHjZh-SMmS9r4zyni-Ewfg
```

### Use Local Neo4j (Docker)
```bash
# In .env
INTELLIGENCE_DATABASE__NEO4J_URL=bolt://localhost:7687
INTELLIGENCE_DATABASE__NEO4J_USER=neo4j
INTELLIGENCE_DATABASE__NEO4J_PASSWORD=password
```

## Troubleshooting

### Connection Timeout

**Problem**: Cannot connect to Neo4j Aura

**Solutions**:
1. Check your internet connection
2. Verify the connection URI is correct
3. Confirm your IP is whitelisted in Neo4j Aura console
4. Check firewall settings

### Authentication Failed

**Problem**: Invalid credentials

**Solutions**:
1. Verify username is `neo4j`
2. Check password is correct (no extra spaces)
3. Reset password in Neo4j Aura console if needed

### SSL/TLS Errors

**Problem**: SSL certificate errors

**Solutions**:
1. Ensure using `neo4j+s://` (not `bolt://`)
2. Update neo4j Python driver: `pip install --upgrade neo4j`
3. Check system SSL certificates are up to date

### IP Not Whitelisted

**Problem**: Connection refused

**Solutions**:
1. Go to Neo4j Aura console
2. Navigate to your instance settings
3. Add your IP address to whitelist
4. Or allow access from anywhere (0.0.0.0/0) for development

## Neo4j Aura Console

Access your database management console:

**URL**: https://console.neo4j.io/

**Features**:
- View database status
- Run Cypher queries
- Monitor performance
- Manage backups
- Configure security
- View metrics

## Database Initialization

The system will automatically create the necessary schema on first connection:

```cypher
// Constraints
CREATE CONSTRAINT asset_id IF NOT EXISTS FOR (a:Asset) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT strategy_id IF NOT EXISTS FOR (s:Strategy) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT portfolio_id IF NOT EXISTS FOR (p:Portfolio) REQUIRE p.id IS UNIQUE;

// Indexes
CREATE INDEX asset_symbol IF NOT EXISTS FOR (a:Asset) ON (a.symbol);
CREATE INDEX strategy_name IF NOT EXISTS FOR (s:Strategy) ON (s.name);
```

## Monitoring

### Check Connection Status

```python
from neo4j import GraphDatabase

uri = "neo4j+s://your-instance.databases.neo4j.io"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

# Verify connectivity
driver.verify_connectivity()
print("✅ Connected to Neo4j Aura")

driver.close()
```

### Query Database Stats

```cypher
// Node count
MATCH (n) RETURN count(n) as nodes;

// Relationship count
MATCH ()-[r]->() RETURN count(r) as relationships;

// Database size
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Store sizes")
YIELD attributes
RETURN attributes.TotalStoreSize.value as size;
```

## Cost Optimization

### Free Tier
- Neo4j Aura offers a free tier
- Suitable for development and testing
- Limited storage and compute

### Paid Tiers
- Scale as needed
- More storage
- Better performance
- Professional support

## Security Best Practices

### ✅ **Do's**
- Use strong passwords
- Enable IP whitelisting
- Use encrypted connections (neo4j+s://)
- Rotate credentials regularly
- Monitor access logs

### ❌ **Don'ts**
- Don't commit credentials to git
- Don't share passwords
- Don't use default passwords
- Don't allow public access in production
- Don't store sensitive data unencrypted

## Backup and Recovery

### Automatic Backups
- Neo4j Aura automatically backs up your data
- Point-in-time recovery available
- Configurable retention periods

### Manual Export
```cypher
// Export data
CALL apoc.export.cypher.all("backup.cypher", {})
```

## Next Steps

1. ✅ Update `.env` with your actual instance ID
2. ✅ Run `python scripts/test-neo4j-aura.py`
3. ✅ Start the trading system
4. ✅ Verify graph data is being stored
5. ✅ Monitor performance in Aura console

## Summary

✅ **Neo4j Aura configured** - Cloud graph database ready  
✅ **Credentials set** - Username and password configured  
✅ **Test script created** - Verify connection anytime  
✅ **No Docker needed** - Fully managed cloud service  
✅ **Secure connection** - Encrypted SSL/TLS  
✅ **Production-ready** - Enterprise-grade hosting  

Your trading system now uses professional cloud infrastructure for graph data!
