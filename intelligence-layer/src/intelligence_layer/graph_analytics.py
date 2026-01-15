"""Neo4j Graph Data Science integration for market analytics."""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, ClientError

from .models import GraphFeatureSnapshot
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class GraphProjection:
    """Graph projection configuration."""
    name: str
    node_labels: List[str]
    relationship_types: Dict[str, Dict[str, Any]]
    properties: Dict[str, List[str]]


@dataclass
class AlgorithmResult:
    """Result from a GDS algorithm execution."""
    algorithm: str
    projection: str
    results: List[Dict[str, Any]]
    execution_time: float
    timestamp: datetime


class Neo4jGDSClient:
    """Client for Neo4j Graph Data Science operations."""
    
    def __init__(self, config: Config):
        """
        Initialize Neo4j GDS client.
        
        Args:
            config: Configuration object with Neo4j connection details
        """
        self.config = config
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.database.neo4j_url,
                auth=(
                    self.config.database.neo4j_user,
                    self.config.database.neo4j_password
                )
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def create_graph_projection(self, projection: GraphProjection) -> bool:
        """
        Create a graph projection for GDS algorithms.
        
        Args:
            projection: Graph projection configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Drop existing projection if it exists
                drop_query = f"CALL gds.graph.drop('{projection.name}', false)"
                session.run(drop_query)
                
                # Build projection query
                node_labels = projection.node_labels
                relationships = {}
                
                for rel_type, config in projection.relationship_types.items():
                    relationships[rel_type] = {
                        "properties": config.get("properties", [])
                    }
                
                # Create projection
                create_query = """
                CALL gds.graph.project(
                    $projectionName,
                    $nodeLabels,
                    $relationships
                )
                """
                
                result = session.run(
                    create_query,
                    projectionName=projection.name,
                    nodeLabels=node_labels,
                    relationships=relationships
                )
                
                record = result.single()
                if record:
                    logger.info(f"Created graph projection '{projection.name}' with "
                              f"{record['nodeCount']} nodes and {record['relationshipCount']} relationships")
                    return True
                
        except ClientError as e:
            logger.error(f"Failed to create graph projection '{projection.name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating projection '{projection.name}': {e}")
            return False
        
        return False
    
    def run_louvain_clustering(
        self, 
        projection_name: str, 
        relationship_weight_property: Optional[str] = None
    ) -> AlgorithmResult:
        """
        Run Louvain clustering algorithm.
        
        Args:
            projection_name: Name of the graph projection
            relationship_weight_property: Property to use as relationship weight
            
        Returns:
            Algorithm execution result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            with self.driver.session() as session:
                # Build query
                if relationship_weight_property:
                    query = f"""
                    CALL gds.louvain.stream('{projection_name}', {{
                        relationshipWeightProperty: '{relationship_weight_property}'
                    }})
                    YIELD nodeId, communityId, intermediateCommunityIds
                    RETURN gds.util.asNode(nodeId) AS node, communityId, intermediateCommunityIds
                    """
                else:
                    query = f"""
                    CALL gds.louvain.stream('{projection_name}')
                    YIELD nodeId, communityId, intermediateCommunityIds
                    RETURN gds.util.asNode(nodeId) AS node, communityId, intermediateCommunityIds
                    """
                
                result = session.run(query)
                results = []
                
                for record in result:
                    node = record["node"]
                    results.append({
                        "node_id": node.element_id,
                        "node_properties": dict(node),
                        "community_id": record["communityId"],
                        "intermediate_communities": record["intermediateCommunityIds"]
                    })
                
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                logger.info(f"Louvain clustering completed on '{projection_name}' "
                          f"in {execution_time:.2f}s, found {len(set(r['community_id'] for r in results))} communities")
                
                return AlgorithmResult(
                    algorithm="louvain",
                    projection=projection_name,
                    results=results,
                    execution_time=execution_time,
                    timestamp=start_time
                )
                
        except Exception as e:
            logger.error(f"Failed to run Louvain clustering on '{projection_name}': {e}")
            raise
    
    def run_degree_centrality(
        self, 
        projection_name: str, 
        relationship_weight_property: Optional[str] = None
    ) -> AlgorithmResult:
        """
        Run degree centrality algorithm.
        
        Args:
            projection_name: Name of the graph projection
            relationship_weight_property: Property to use as relationship weight
            
        Returns:
            Algorithm execution result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            with self.driver.session() as session:
                if relationship_weight_property:
                    query = f"""
                    CALL gds.degree.stream('{projection_name}', {{
                        relationshipWeightProperty: '{relationship_weight_property}'
                    }})
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId) AS node, score
                    """
                else:
                    query = f"""
                    CALL gds.degree.stream('{projection_name}')
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId) AS node, score
                    """
                
                result = session.run(query)
                results = []
                
                for record in result:
                    node = record["node"]
                    results.append({
                        "node_id": node.element_id,
                        "node_properties": dict(node),
                        "degree_centrality": record["score"]
                    })
                
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                logger.info(f"Degree centrality completed on '{projection_name}' "
                          f"in {execution_time:.2f}s")
                
                return AlgorithmResult(
                    algorithm="degree_centrality",
                    projection=projection_name,
                    results=results,
                    execution_time=execution_time,
                    timestamp=start_time
                )
                
        except Exception as e:
            logger.error(f"Failed to run degree centrality on '{projection_name}': {e}")
            raise
    
    def run_betweenness_centrality(
        self, 
        projection_name: str,
        sample_size: Optional[int] = None
    ) -> AlgorithmResult:
        """
        Run betweenness centrality algorithm.
        
        Args:
            projection_name: Name of the graph projection
            sample_size: Sample size for approximation (None for exact)
            
        Returns:
            Algorithm execution result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            with self.driver.session() as session:
                if sample_size:
                    query = f"""
                    CALL gds.betweenness.stream('{projection_name}', {{
                        samplingSize: {sample_size}
                    }})
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId) AS node, score
                    """
                else:
                    query = f"""
                    CALL gds.betweenness.stream('{projection_name}')
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId) AS node, score
                    """
                
                result = session.run(query)
                results = []
                
                for record in result:
                    node = record["node"]
                    results.append({
                        "node_id": node.element_id,
                        "node_properties": dict(node),
                        "betweenness_centrality": record["score"]
                    })
                
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                logger.info(f"Betweenness centrality completed on '{projection_name}' "
                          f"in {execution_time:.2f}s")
                
                return AlgorithmResult(
                    algorithm="betweenness_centrality",
                    projection=projection_name,
                    results=results,
                    execution_time=execution_time,
                    timestamp=start_time
                )
                
        except Exception as e:
            logger.error(f"Failed to run betweenness centrality on '{projection_name}': {e}")
            raise
    
    def run_pagerank(
        self, 
        projection_name: str,
        relationship_weight_property: Optional[str] = None,
        damping_factor: float = 0.85,
        max_iterations: int = 20
    ) -> AlgorithmResult:
        """
        Run PageRank algorithm.
        
        Args:
            projection_name: Name of the graph projection
            relationship_weight_property: Property to use as relationship weight
            damping_factor: PageRank damping factor
            max_iterations: Maximum number of iterations
            
        Returns:
            Algorithm execution result
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            with self.driver.session() as session:
                config = {
                    "dampingFactor": damping_factor,
                    "maxIterations": max_iterations
                }
                
                if relationship_weight_property:
                    config["relationshipWeightProperty"] = relationship_weight_property
                
                query = f"""
                CALL gds.pageRank.stream('{projection_name}', $config)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId) AS node, score
                """
                
                result = session.run(query, config=config)
                results = []
                
                for record in result:
                    node = record["node"]
                    results.append({
                        "node_id": node.element_id,
                        "node_properties": dict(node),
                        "pagerank_score": record["score"]
                    })
                
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                logger.info(f"PageRank completed on '{projection_name}' "
                          f"in {execution_time:.2f}s")
                
                return AlgorithmResult(
                    algorithm="pagerank",
                    projection=projection_name,
                    results=results,
                    execution_time=execution_time,
                    timestamp=start_time
                )
                
        except Exception as e:
            logger.error(f"Failed to run PageRank on '{projection_name}': {e}")
            raise
    
    def materialize_results_to_neo4j(
        self, 
        algorithm_result: AlgorithmResult,
        node_property_mapping: Dict[str, str]
    ) -> bool:
        """
        Materialize algorithm results back to Neo4j as node properties.
        
        Args:
            algorithm_result: Results from algorithm execution
            node_property_mapping: Mapping of result keys to node property names
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                for result in algorithm_result.results:
                    node_id = result["node_id"]
                    
                    # Build SET clause for properties
                    set_clauses = []
                    params = {"nodeId": node_id}
                    
                    for result_key, property_name in node_property_mapping.items():
                        if result_key in result:
                            param_name = f"prop_{property_name}"
                            set_clauses.append(f"n.{property_name} = ${param_name}")
                            params[param_name] = result[result_key]
                    
                    if set_clauses:
                        query = f"""
                        MATCH (n) WHERE elementId(n) = $nodeId
                        SET {', '.join(set_clauses)}
                        """
                        
                        session.run(query, **params)
                
                logger.info(f"Materialized {len(algorithm_result.results)} results "
                          f"from {algorithm_result.algorithm} to Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Failed to materialize results to Neo4j: {e}")
            return False


class MarketGraphAnalytics:
    """High-level market graph analytics using Neo4j GDS."""
    
    def __init__(self, config: Config):
        """
        Initialize market graph analytics.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.gds_client = Neo4jGDSClient(config)
        self.projections = self._define_projections()
    
    def _define_projections(self) -> Dict[str, GraphProjection]:
        """Define standard graph projections for market analysis."""
        return {
            "asset_correlation": GraphProjection(
                name="asset_correlation_graph",
                node_labels=["Asset"],
                relationship_types={
                    "CORRELATED": {
                        "properties": ["strength", "window", "sign"]
                    }
                },
                properties={
                    "Asset": ["asset_id", "asset_class", "venue"]
                }
            ),
            "regime_transition": GraphProjection(
                name="regime_transition_graph",
                node_labels=["MarketRegime"],
                relationship_types={
                    "TRANSITIONS_TO": {
                        "properties": ["probability", "avg_duration"]
                    }
                },
                properties={
                    "MarketRegime": ["regime_id", "volatility_level", "trend_state"]
                }
            ),
            "strategy_performance": GraphProjection(
                name="strategy_performance_graph",
                node_labels=["Strategy", "MarketRegime"],
                relationship_types={
                    "PERFORMS_IN": {
                        "properties": ["sharpe", "max_dd", "sample_size"]
                    }
                },
                properties={
                    "Strategy": ["strategy_id", "family", "horizon"],
                    "MarketRegime": ["regime_id", "volatility_level"]
                }
            )
        }
    
    def setup_projections(self) -> bool:
        """
        Set up all graph projections.
        
        Returns:
            True if all projections created successfully
        """
        success = True
        for projection_name, projection in self.projections.items():
            if not self.gds_client.create_graph_projection(projection):
                success = False
                logger.error(f"Failed to create projection: {projection_name}")
        
        return success
    
    def analyze_asset_correlations(self) -> Dict[str, Any]:
        """
        Analyze asset correlation network.
        
        Returns:
            Dictionary containing analysis results
        """
        projection_name = "asset_correlation_graph"
        results = {}
        
        try:
            # Run clustering to find asset groups
            clustering_result = self.gds_client.run_louvain_clustering(
                projection_name, "strength"
            )
            
            # Run centrality to find systemically important assets
            centrality_result = self.gds_client.run_degree_centrality(
                projection_name, "strength"
            )
            
            # Run betweenness for contagion risk
            betweenness_result = self.gds_client.run_betweenness_centrality(
                projection_name, sample_size=1000
            )
            
            # Materialize results
            self.gds_client.materialize_results_to_neo4j(
                clustering_result,
                {"community_id": "cluster_id"}
            )
            
            self.gds_client.materialize_results_to_neo4j(
                centrality_result,
                {"degree_centrality": "systemic_exposure"}
            )
            
            self.gds_client.materialize_results_to_neo4j(
                betweenness_result,
                {"betweenness_centrality": "contagion_risk"}
            )
            
            results = {
                "clustering": clustering_result,
                "centrality": centrality_result,
                "betweenness": betweenness_result,
                "timestamp": datetime.now(timezone.utc)
            }
            
            logger.info("Asset correlation analysis completed")
            
        except Exception as e:
            logger.error(f"Asset correlation analysis failed: {e}")
            raise
        
        return results
    
    def analyze_regime_transitions(self) -> Dict[str, Any]:
        """
        Analyze regime transition network.
        
        Returns:
            Dictionary containing analysis results
        """
        projection_name = "regime_transition_graph"
        results = {}
        
        try:
            # Run PageRank to find dominant regimes
            pagerank_result = self.gds_client.run_pagerank(
                projection_name, "probability"
            )
            
            # Materialize results
            self.gds_client.materialize_results_to_neo4j(
                pagerank_result,
                {"pagerank_score": "dominance_score"}
            )
            
            results = {
                "pagerank": pagerank_result,
                "timestamp": datetime.now(timezone.utc)
            }
            
            logger.info("Regime transition analysis completed")
            
        except Exception as e:
            logger.error(f"Regime transition analysis failed: {e}")
            raise
        
        return results
    
    def export_features_to_parquet(
        self, 
        output_path: str,
        analysis_results: Dict[str, Any]
    ) -> bool:
        """
        Export graph features to Parquet format.
        
        Args:
            output_path: Path to save Parquet file
            analysis_results: Results from graph analysis
            
        Returns:
            True if export successful
        """
        try:
            # Combine all results into a single DataFrame
            all_features = []
            
            for analysis_type, result in analysis_results.items():
                if isinstance(result, AlgorithmResult):
                    for node_result in result.results:
                        feature_row = {
                            "node_id": node_result["node_id"],
                            "analysis_type": analysis_type,
                            "algorithm": result.algorithm,
                            "timestamp": result.timestamp.isoformat()
                        }
                        
                        # Add node properties
                        if "node_properties" in node_result:
                            feature_row.update(node_result["node_properties"])
                        
                        # Add algorithm-specific results
                        for key, value in node_result.items():
                            if key not in ["node_id", "node_properties"]:
                                feature_row[key] = value
                        
                        all_features.append(feature_row)
            
            if all_features:
                df = pd.DataFrame(all_features)
                df.to_parquet(output_path, index=False)
                logger.info(f"Exported {len(all_features)} graph features to {output_path}")
                return True
            else:
                logger.warning("No features to export")
                return False
                
        except Exception as e:
            logger.error(f"Failed to export features to Parquet: {e}")
            return False
    
    def get_graph_features_for_asset(self, asset_id: str) -> GraphFeatureSnapshot:
        """
        Get current graph features for a specific asset.
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Graph feature snapshot
        """
        try:
            with self.gds_client.driver.session() as session:
                query = """
                MATCH (a:Asset {asset_id: $asset_id})
                RETURN a.cluster_id AS cluster_id,
                       a.systemic_exposure AS centrality_score,
                       a.contagion_risk AS systemic_risk_proxy,
                       a.asset_class AS asset_class
                """
                
                result = session.run(query, asset_id=asset_id)
                record = result.single()
                
                if record:
                    return GraphFeatureSnapshot(
                        timestamp=datetime.now(timezone.utc),
                        asset_cluster_id=record["cluster_id"],
                        centrality_score=record["centrality_score"],
                        systemic_risk_proxy=record["systemic_risk_proxy"],
                        features={
                            "asset_class": record["asset_class"]
                        }
                    )
                else:
                    # Return empty snapshot if asset not found
                    return GraphFeatureSnapshot(
                        timestamp=datetime.now(timezone.utc)
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get graph features for asset {asset_id}: {e}")
            return GraphFeatureSnapshot(timestamp=datetime.now(timezone.utc))
    
    def close(self) -> None:
        """Close GDS client connection."""
        self.gds_client.close()
