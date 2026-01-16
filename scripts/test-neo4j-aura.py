#!/usr/bin/env python3
"""
Test Neo4j Aura Connection
Verifies connection to Neo4j Aura cloud database
"""

import os
import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    from neo4j import GraphDatabase
except ImportError:
    print("❌ neo4j package not installed")
    print("Install with: pip install neo4j")
    sys.exit(1)


def test_neo4j_connection():
    """Test Neo4j Aura connection"""
    print("=" * 60)
    print("NEO4J AURA CONNECTION TEST")
    print("=" * 60)
    print()
    
    # Get connection details from environment
    uri = os.getenv("INTELLIGENCE_DATABASE__NEO4J_URL") or os.getenv("TRADING_DATABASE__NEO4J_URL")
    user = os.getenv("INTELLIGENCE_DATABASE__NEO4J_USER") or os.getenv("TRADING_DATABASE__NEO4J_USER")
    password = os.getenv("INTELLIGENCE_DATABASE__NEO4J_PASSWORD") or os.getenv("TRADING_DATABASE__NEO4J_PASSWORD")
    
    print(f"Configuration:")
    print(f"  URI: {uri}")
    print(f"  User: {user}")
    print(f"  Password: {'*' * len(password) if password else 'NOT SET'}")
    print()
    
    if not uri or not user or not password:
        print("❌ ERROR: Neo4j credentials not configured")
        print("Please set the following in .env:")
        print("  INTELLIGENCE_DATABASE__NEO4J_URL=neo4j+s://your-instance.databases.neo4j.io")
        print("  INTELLIGENCE_DATABASE__NEO4J_USER=neo4j")
        print("  INTELLIGENCE_DATABASE__NEO4J_PASSWORD=your_password")
        return False
    
    if "your-instance-id" in uri:
        print("❌ ERROR: Please replace 'your-instance-id' with your actual Neo4j Aura instance ID")
        print()
        print("Your Neo4j Aura connection URI should look like:")
        print("  neo4j+s://abc123xyz.databases.neo4j.io")
        print()
        print("Find it in your Neo4j Aura console at: https://console.neo4j.io/")
        return False
    
    try:
        print("Connecting to Neo4j Aura...")
        
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Verify connectivity
        driver.verify_connectivity()
        print("✅ Connected successfully")
        print()
        
        # Test query
        print("Running test query...")
        with driver.session() as session:
            result = session.run("RETURN 'Hello from Neo4j Aura!' as message")
            record = result.single()
            print(f"✅ Query successful: {record['message']}")
            print()
        
        # Get database info
        print("Database Information:")
        with driver.session() as session:
            # Get Neo4j version
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            for record in result:
                print(f"  Name: {record['name']}")
                print(f"  Version: {record['versions'][0]}")
                print(f"  Edition: {record['edition']}")
            print()
            
            # Count nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()['count']
            print(f"  Total nodes: {count}")
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            count = result.single()['count']
            print(f"  Total relationships: {count}")
            print()
        
        # Test write operation
        print("Testing write operation...")
        with driver.session() as session:
            result = session.run(
                "CREATE (n:TestNode {name: $name, timestamp: datetime()}) RETURN n",
                name="Connection Test"
            )
            node = result.single()['n']
            print(f"✅ Created test node: {dict(node)}")
            
            # Clean up test node
            session.run("MATCH (n:TestNode {name: $name}) DELETE n", name="Connection Test")
            print("✅ Cleaned up test node")
            print()
        
        driver.close()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("Your Neo4j Aura database is properly configured!")
        print("The trading system can now use Neo4j Aura for:")
        print("  - Graph-based market analysis")
        print("  - Strategy relationship tracking")
        print("  - Portfolio network analysis")
        print("  - Knowledge graph storage")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check your Neo4j Aura instance is running")
        print("  2. Verify the connection URI is correct")
        print("  3. Confirm username and password are correct")
        print("  4. Check firewall/network settings")
        print("  5. Ensure your IP is whitelisted in Neo4j Aura")
        print()
        print("Neo4j Aura Console: https://console.neo4j.io/")
        return False


if __name__ == "__main__":
    success = test_neo4j_connection()
    sys.exit(0 if success else 1)
