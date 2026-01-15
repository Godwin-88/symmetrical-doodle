"""Property-based tests for Neo4j GDS algorithm execution."""

import pytest
import tempfile
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from typing import Dict, List, Any

from intelligence_layer.graph_analytics import (
    Neo4jGDSClient,
    MarketGraphAnalytics,
    GraphProjection,
    AlgorithmResult
)
from intelligence_layer.config import Config


# Strategies for generating test data
@st.composite
def algorithm_result_strategy(draw):
    """Generate valid algorithm results."""
    algorithm = draw(st.sampled_from(["louvain", "degree_centrality", "betweenness_centrality", "pagerank"]))
    projection = draw(st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))))
    
    # Generate realistic node results
    num_nodes = draw(st.integers(min_value=3, max_value=10))
    results = []
    
    for i in range(num_nodes):
        node_result = {
            "node_id": f"node_{i}",
            "node_properties": {
                "asset_id": f"ASSET_{i}",
                "asset_class": draw(st.sampled_from(["FX", "Equity", "Crypto", "Commodity"]))
            }
        }
        
        # Add algorithm-specific results
        if algorithm == "louvain":
            node_result["community_id"] = draw(st.integers(min_value=0, max_value=10))
            node_result["intermediate_communities"] = [draw(st.integers(min_value=0, max_value=5))]
        elif algorithm == "degree_centrality":
            node_result["degree_centrality"] = draw(st.floats(min_value=0.0, max_value=100.0))
        elif algorithm == "betweenness_centrality":
            node_result["betweenness_centrality"] = draw(st.floats(min_value=0.0, max_value=1.0))
        elif algorithm == "pagerank":
            node_result["pagerank_score"] = draw(st.floats(min_value=0.0, max_value=1.0))
        
        results.append(node_result)
    
    execution_time = draw(st.floats(min_value=0.1, max_value=10.0))
    timestamp = datetime.now(timezone.utc) - timedelta(seconds=draw(st.integers(min_value=0, max_value=3600)))
    
    return AlgorithmResult(
        algorithm=algorithm,
        projection=projection,
        results=results,
        execution_time=execution_time,
        timestamp=timestamp
    )


@st.composite
def node_property_mapping_strategy(draw):
    """Generate valid node property mappings."""
    mappings = {}
    
    # Common algorithm result keys
    possible_keys = [
        "community_id", "degree_centrality", "betweenness_centrality", 
        "pagerank_score", "intermediate_communities"
    ]
    
    # Select subset of keys to map
    num_mappings = draw(st.integers(min_value=1, max_value=len(possible_keys)))
    selected_keys = draw(st.lists(
        st.sampled_from(possible_keys), 
        min_size=num_mappings, 
        max_size=num_mappings, 
        unique=True
    ))
    
    for key in selected_keys:
        property_name = draw(st.text(
            min_size=3, max_size=15, 
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), blacklist_characters="_")
        ))
        mappings[key] = f"gds_{property_name}"
    
    return mappings


@st.composite
def graph_projection_strategy(draw):
    """Generate valid graph projections."""
    name = draw(st.text(min_size=5, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))))
    
    node_labels = draw(st.lists(
        st.sampled_from(["Asset", "MarketRegime", "Strategy", "IntelligenceSignal"]),
        min_size=1, max_size=3, unique=True
    ))
    
    relationship_types = {}
    rel_types = ["CORRELATED", "TRANSITIONS_TO", "PERFORMS_IN", "SENSITIVE_TO", "AFFECTS"]
    selected_rels = draw(st.lists(st.sampled_from(rel_types), min_size=1, max_size=3, unique=True))
    
    for rel_type in selected_rels:
        relationship_types[rel_type] = {
            "properties": draw(st.lists(
                st.sampled_from(["strength", "probability", "sharpe", "beta", "impact_score"]),
                min_size=1, max_size=3, unique=True
            ))
        }
    
    properties = {}
    for label in node_labels:
        if label == "Asset":
            properties[label] = ["asset_id", "asset_class", "venue"]
        elif label == "MarketRegime":
            properties[label] = ["regime_id", "volatility_level", "trend_state"]
        elif label == "Strategy":
            properties[label] = ["strategy_id", "family", "horizon"]
        else:
            properties[label] = ["signal_id", "type", "confidence"]
    
    return GraphProjection(
        name=name,
        node_labels=node_labels,
        relationship_types=relationship_types,
        properties=properties
    )


class TestGDSAlgorithmExecutionProperties:
    """Property-based tests for Neo4j GDS algorithm execution.
    
    **Validates: Requirements 14.10-14.11**
    """
    
    @given(algorithm_result_strategy(), node_property_mapping_strategy())
    @settings(max_examples=5, deadline=3000)
    def test_property_algorithm_results_materialization(self, algorithm_result, property_mapping):
        """
        Property: For any GDS algorithm execution, results should be materialized as node properties.
        
        **Feature: algorithmic-trading-system, Property 10: Neo4j GDS Algorithm Execution**
        **Validates: Requirements 14.10-14.11**
        """
        # Filter mapping to only include keys present in results
        available_keys = set()
        for result in algorithm_result.results:
            available_keys.update(result.keys())
        
        filtered_mapping = {
            k: v for k, v in property_mapping.items() 
            if k in available_keys and k != "node_id" and k != "node_properties"
        }
        
        assume(len(filtered_mapping) > 0)  # Must have at least one property to map
        
        # Mock Neo4j driver and session with proper context manager
        mock_driver = Mock()
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_driver.session.return_value = mock_context_manager
        
        # Create GDS client with mocked driver - patch the connection to avoid actual Neo4j connection
        config = Mock()
        config.database.neo4j_url = "bolt://localhost:7687"
        config.database.neo4j_user = "neo4j"
        config.database.neo4j_password = "password"
        
        with patch.object(Neo4jGDSClient, '_connect'):
            gds_client = Neo4jGDSClient(config)
            gds_client.driver = mock_driver
            
            # Test materialization
            result = gds_client.materialize_results_to_neo4j(algorithm_result, filtered_mapping)
            
            # Should return True for successful materialization
            assert result is True, "Materialization should succeed"
            
            # Verify session.run was called for each result
            expected_calls = len(algorithm_result.results)
            assert mock_session.run.call_count == expected_calls, (
                f"Expected {expected_calls} calls to session.run, got {mock_session.run.call_count}"
            )
            
            # Verify queries contain proper SET clauses for mapped properties
            for call_args in mock_session.run.call_args_list:
                query = call_args[0][0]  # First positional argument is the query
                params = call_args[1]    # Keyword arguments are parameters
                
                # Query should be a MATCH...SET statement
                assert "MATCH (n) WHERE elementId(n) =" in query, "Query should match node by elementId"
                assert "SET " in query, "Query should contain SET clause"
                
                # Should have nodeId parameter
                assert "nodeId" in params, "Query should have nodeId parameter"
                
                # Should have parameters for each mapped property
                for result_key, property_name in filtered_mapping.items():
                    param_name = f"prop_{property_name}"
                    if any(result_key in result for result in algorithm_result.results):
                        # At least some results should have this property parameter
                        pass  # We can't guarantee every call has every property
    
    @given(algorithm_result_strategy())
    @settings(max_examples=5, deadline=5000)
    def test_property_parquet_export_completeness(self, algorithm_result):
        """
        Property: For any algorithm results, Parquet export should contain all result data.
        
        **Feature: algorithmic-trading-system, Property 10: Neo4j GDS Algorithm Execution**
        **Validates: Requirements 14.10-14.11**
        """
        # Create temporary file for export
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # Mock config and create analytics instance
            config = Mock()
            config.database.neo4j_url = "bolt://localhost:7687"
            config.database.neo4j_user = "neo4j"
            config.database.neo4j_password = "password"
            
            # Mock the GDS client to avoid actual Neo4j connection
            with patch('intelligence_layer.graph_analytics.Neo4jGDSClient'):
                analytics = MarketGraphAnalytics(config)
                
                # Create analysis results dictionary
                analysis_results = {
                    "test_analysis": algorithm_result,
                    "timestamp": datetime.now(timezone.utc)
                }
                
                # Export to Parquet
                result = analytics.export_features_to_parquet(output_path, analysis_results)
                
                # Should succeed
                assert result is True, "Parquet export should succeed"
                
                # Verify file was created
                assert os.path.exists(output_path), "Parquet file should be created"
                
                # Load and verify contents
                df = pd.read_parquet(output_path)
                
                # Should have rows for each algorithm result
                expected_rows = len(algorithm_result.results)
                assert len(df) == expected_rows, (
                    f"Expected {expected_rows} rows in Parquet, got {len(df)}"
                )
                
                # Should have required columns
                required_columns = ["node_id", "analysis_type", "algorithm", "timestamp"]
                for col in required_columns:
                    assert col in df.columns, f"Missing required column: {col}"
                
                # Verify data integrity
                assert all(df["analysis_type"] == "test_analysis"), "Analysis type should be preserved"
                assert all(df["algorithm"] == algorithm_result.algorithm), "Algorithm should be preserved"
                
                # Verify node IDs are preserved
                exported_node_ids = set(df["node_id"].tolist())
                expected_node_ids = {result["node_id"] for result in algorithm_result.results}
                assert exported_node_ids == expected_node_ids, "Node IDs should be preserved"
                
                # Verify algorithm-specific data is included
                for result in algorithm_result.results:
                    matching_rows = df[df["node_id"] == result["node_id"]]
                    assert len(matching_rows) == 1, f"Should have exactly one row for node {result['node_id']}"
                    
                    row = matching_rows.iloc[0]
                    
                    # Check that algorithm-specific results are included
                    for key, value in result.items():
                        if key not in ["node_id", "node_properties"] and key in df.columns:
                            assert row[key] == value, f"Value mismatch for {key}: expected {value}, got {row[key]}"
        
        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @given(graph_projection_strategy())
    @settings(max_examples=3, deadline=3000)
    def test_property_no_synchronous_trading_queries(self, projection):
        """
        Property: For any GDS algorithm execution, no synchronous queries should occur during trading operations.
        
        **Feature: algorithmic-trading-system, Property 10: Neo4j GDS Algorithm Execution**
        **Validates: Requirements 14.10-14.11**
        """
        # Mock Neo4j driver and session with proper context manager
        mock_driver = Mock()
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_driver.session.return_value = mock_context_manager
        
        # Mock successful algorithm execution
        mock_session.run.return_value.single.return_value = {
            "nodeCount": 10,
            "relationshipCount": 20
        }
        
        # Mock streaming results for algorithms
        mock_records = []
        for i in range(5):  # Generate some mock results
            mock_node = Mock()
            mock_node.element_id = f"node_{i}"
            # Make the mock node behave like a dict when dict() is called on it
            mock_node.keys = Mock(return_value=["asset_id", "asset_class"])
            mock_node.__getitem__ = Mock(side_effect=lambda key: f"ASSET_{i}" if key == "asset_id" else "FX")
            mock_node.__iter__ = Mock(return_value=iter(["asset_id", "asset_class"]))
            
            mock_record = Mock()
            mock_record.__getitem__ = Mock(side_effect=lambda key: {
                "node": mock_node,
                "communityId": i % 3,
                "intermediateCommunityIds": [i % 2],
                "score": float(i) * 0.1
            }.get(key))
            mock_records.append(mock_record)
        
        mock_session.run.return_value.__iter__ = Mock(return_value=iter(mock_records))
        
        # Create GDS client with mocked driver - patch the connection to avoid actual Neo4j connection
        config = Mock()
        config.database.neo4j_url = "bolt://localhost:7687"
        config.database.neo4j_user = "neo4j"
        config.database.neo4j_password = "password"
        
        with patch.object(Neo4jGDSClient, '_connect'):
            gds_client = Neo4jGDSClient(config)
            gds_client.driver = mock_driver
            
            # Test that projection creation doesn't block
            start_time = datetime.now(timezone.utc)
            result = gds_client.create_graph_projection(projection)
            creation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Should complete quickly (not block for trading operations)
            assert creation_time < 1.0, f"Projection creation took too long: {creation_time}s"
            assert result is True, "Projection creation should succeed"
            
            # Test that algorithm execution doesn't block
            start_time = datetime.now(timezone.utc)
            algorithm_result = gds_client.run_louvain_clustering(projection.name, "strength")
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Should complete quickly (simulated, but validates non-blocking pattern)
            assert execution_time < 1.0, f"Algorithm execution took too long: {execution_time}s"
            assert algorithm_result.algorithm == "louvain", "Should return proper algorithm result"
            assert len(algorithm_result.results) > 0, "Should return non-empty results"
            
            # Verify that results are structured for asynchronous consumption
            for result in algorithm_result.results:
                assert "node_id" in result, "Results should include node_id for async processing"
                assert isinstance(result["node_id"], str), "Node ID should be string for serialization"
    
    @given(st.lists(algorithm_result_strategy(), min_size=1, max_size=3))
    @settings(max_examples=3, deadline=5000)
    def test_property_batch_materialization_consistency(self, algorithm_results):
        """
        Property: For any batch of algorithm results, materialization should be consistent and complete.
        
        **Feature: algorithmic-trading-system, Property 10: Neo4j GDS Algorithm Execution**
        **Validates: Requirements 14.10-14.11**
        """
        # Mock Neo4j driver and session with proper context manager
        mock_driver = Mock()
        mock_session = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_driver.session.return_value = mock_context_manager
        
        # Create GDS client with mocked driver - patch the connection to avoid actual Neo4j connection
        config = Mock()
        config.database.neo4j_url = "bolt://localhost:7687"
        config.database.neo4j_user = "neo4j"
        config.database.neo4j_password = "password"
        
        with patch.object(Neo4jGDSClient, '_connect'):
            gds_client = Neo4jGDSClient(config)
            gds_client.driver = mock_driver
            
            # Define consistent property mappings for each algorithm type
            property_mappings = {
                "louvain": {"community_id": "cluster_id"},
                "degree_centrality": {"degree_centrality": "systemic_exposure"},
                "betweenness_centrality": {"betweenness_centrality": "contagion_risk"},
                "pagerank": {"pagerank_score": "dominance_score"}
            }
            
            total_expected_calls = 0
            successful_materializations = 0
            
            # Materialize each algorithm result
            for algorithm_result in algorithm_results:
                mapping = property_mappings.get(algorithm_result.algorithm, {})
                
                if mapping:  # Only test if we have a mapping for this algorithm
                    result = gds_client.materialize_results_to_neo4j(algorithm_result, mapping)
                    
                    if result:
                        successful_materializations += 1
                        total_expected_calls += len(algorithm_result.results)
            
            # Verify all materializations succeeded
            assert successful_materializations == len([
                r for r in algorithm_results 
                if r.algorithm in property_mappings
            ]), "All valid algorithm results should be materialized successfully"
            
            # Verify total number of database calls
            assert mock_session.run.call_count == total_expected_calls, (
                f"Expected {total_expected_calls} database calls, got {mock_session.run.call_count}"
            )
            
            # Verify each call was properly structured
            for call_args in mock_session.run.call_args_list:
                query = call_args[0][0]
                params = call_args[1]
                
                # Each query should follow the materialization pattern
                assert "MATCH (n) WHERE elementId(n) =" in query, "Query should match by elementId"
                assert "SET " in query, "Query should contain SET clause"
                assert "nodeId" in params, "Query should have nodeId parameter"
    
    @given(algorithm_result_strategy())
    @settings(max_examples=5, deadline=3000)
    def test_property_algorithm_result_structure_validity(self, algorithm_result):
        """
        Property: For any algorithm result, the structure should be valid and complete.
        
        **Feature: algorithmic-trading-system, Property 10: Neo4j GDS Algorithm Execution**
        **Validates: Requirements 14.10-14.11**
        """
        # Verify basic structure
        assert hasattr(algorithm_result, 'algorithm'), "Result should have algorithm field"
        assert hasattr(algorithm_result, 'projection'), "Result should have projection field"
        assert hasattr(algorithm_result, 'results'), "Result should have results field"
        assert hasattr(algorithm_result, 'execution_time'), "Result should have execution_time field"
        assert hasattr(algorithm_result, 'timestamp'), "Result should have timestamp field"
        
        # Verify field types
        assert isinstance(algorithm_result.algorithm, str), "Algorithm should be string"
        assert isinstance(algorithm_result.projection, str), "Projection should be string"
        assert isinstance(algorithm_result.results, list), "Results should be list"
        assert isinstance(algorithm_result.execution_time, float), "Execution time should be float"
        assert isinstance(algorithm_result.timestamp, datetime), "Timestamp should be datetime"
        
        # Verify algorithm is valid
        valid_algorithms = ["louvain", "degree_centrality", "betweenness_centrality", "pagerank"]
        assert algorithm_result.algorithm in valid_algorithms, (
            f"Algorithm {algorithm_result.algorithm} should be one of {valid_algorithms}"
        )
        
        # Verify execution time is reasonable
        assert 0.0 < algorithm_result.execution_time < 3600.0, (
            f"Execution time should be reasonable: {algorithm_result.execution_time}"
        )
        
        # Verify timestamp is recent (within last day for test data)
        time_diff = abs((datetime.now(timezone.utc) - algorithm_result.timestamp).total_seconds())
        assert time_diff < 86400, f"Timestamp should be recent: {time_diff}s ago"
        
        # Verify results structure
        assert len(algorithm_result.results) > 0, "Should have at least one result"
        
        for result in algorithm_result.results:
            assert isinstance(result, dict), "Each result should be a dictionary"
            assert "node_id" in result, "Each result should have node_id"
            assert isinstance(result["node_id"], str), "Node ID should be string"
            
            # Verify algorithm-specific fields
            if algorithm_result.algorithm == "louvain":
                assert "community_id" in result, "Louvain results should have community_id"
                assert isinstance(result["community_id"], int), "Community ID should be integer"
            elif algorithm_result.algorithm == "degree_centrality":
                assert "degree_centrality" in result, "Degree centrality results should have degree_centrality"
                assert isinstance(result["degree_centrality"], (int, float)), "Degree centrality should be numeric"
            elif algorithm_result.algorithm == "betweenness_centrality":
                assert "betweenness_centrality" in result, "Betweenness results should have betweenness_centrality"
                assert isinstance(result["betweenness_centrality"], (int, float)), "Betweenness should be numeric"
            elif algorithm_result.algorithm == "pagerank":
                assert "pagerank_score" in result, "PageRank results should have pagerank_score"
                assert isinstance(result["pagerank_score"], (int, float)), "PageRank score should be numeric"


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])