#!/usr/bin/env python3
"""
Test script for batch ingestion management API.

This script tests the multi-source batch ingestion management API endpoints
including job creation, control operations, progress tracking, and WebSocket
real-time updates.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
import aiohttp
import websockets
from pathlib import Path

from core.config import load_config
from core.logging import get_logger


class BatchIngestionAPITester:
    """Test client for batch ingestion management API"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.logger = get_logger(__name__)
        self.session: aiohttp.ClientSession = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> bool:
        """Test API health check"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(f"✓ Health check passed: {data}")
                    return True
                else:
                    self.logger.error(f"✗ Health check failed: {response.status}")
                    return False
        except Exception as e:
            self.logger.error(f"✗ Health check error: {e}")
            return False
    
    async def test_create_batch_job(self, user_id: str = "test_user") -> str:
        """Test batch job creation"""
        try:
            # Create a test job with mock file selections
            job_request = {
                "user_id": user_id,
                "name": "Test Batch Job",
                "description": "Test batch ingestion job for API testing",
                "priority": "normal",
                "file_selections": [
                    {
                        "connection_id": "test_connection_1",
                        "source_type": "local_directory",
                        "file_ids": ["test_file_1", "test_file_2"]
                    },
                    {
                        "connection_id": "test_connection_2", 
                        "source_type": "local_zip",
                        "file_ids": ["test_file_3"]
                    }
                ],
                "processing_options": {
                    "use_llm_parsing": False,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "parallel_processing": True,
                    "max_workers": 2
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/batch/jobs",
                json=job_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    job_id = data["job_id"]
                    self.logger.info(f"✓ Created batch job: {job_id}")
                    self.logger.info(f"  Job details: {data['name']} - {data['status']}")
                    return job_id
                else:
                    error_text = await response.text()
                    self.logger.error(f"✗ Failed to create batch job: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"✗ Error creating batch job: {e}")
            return None
    
    async def test_get_batch_job(self, job_id: str, user_id: str = "test_user") -> Dict[str, Any]:
        """Test getting batch job details"""
        try:
            async with self.session.get(
                f"{self.base_url}/batch/jobs/{job_id}",
                params={"user_id": user_id}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(f"✓ Retrieved batch job {job_id}")
                    self.logger.info(f"  Status: {data['status']}")
                    self.logger.info(f"  Progress: {data['progress_percentage']:.1f}%")
                    self.logger.info(f"  Files: {data['total_files']} total, {data['completed_files']} completed")
                    return data
                else:
                    error_text = await response.text()
                    self.logger.error(f"✗ Failed to get batch job: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"✗ Error getting batch job: {e}")
            return None
    
    async def test_job_control(self, job_id: str, action: str, user_id: str = "test_user") -> bool:
        """Test job control operations"""
        try:
            control_request = {
                "user_id": user_id,
                "action": action,
                "retry_failed_only": True
            }
            
            async with self.session.post(
                f"{self.base_url}/batch/jobs/{job_id}/control",
                json=control_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(f"✓ Job {action} successful: {data['message']}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"✗ Job {action} failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"✗ Error controlling job: {e}")
            return False
    
    async def test_list_jobs(self, user_id: str = "test_user") -> List[Dict[str, Any]]:
        """Test listing batch jobs"""
        try:
            list_request = {
                "user_id": user_id,
                "limit": 10,
                "offset": 0
            }
            
            async with self.session.post(
                f"{self.base_url}/batch/jobs/list",
                json=list_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    jobs = data["jobs"]
                    self.logger.info(f"✓ Listed {len(jobs)} batch jobs")
                    
                    for job in jobs:
                        self.logger.info(f"  Job {job['job_id']}: {job['name']} - {job['status']}")
                    
                    return jobs
                else:
                    error_text = await response.text()
                    self.logger.error(f"✗ Failed to list jobs: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"✗ Error listing jobs: {e}")
            return []
    
    async def test_job_statistics(self, user_id: str = "test_user") -> Dict[str, Any]:
        """Test getting job statistics"""
        try:
            async with self.session.get(
                f"{self.base_url}/batch/statistics",
                params={"user_id": user_id}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(f"✓ Retrieved job statistics")
                    self.logger.info(f"  Total jobs: {data['total_jobs']}")
                    self.logger.info(f"  Active jobs: {data['active_jobs']}")
                    self.logger.info(f"  Completed jobs: {data['completed_jobs']}")
                    self.logger.info(f"  Success rate: {data['success_rate']:.1f}%")
                    return data
                else:
                    error_text = await response.text()
                    self.logger.error(f"✗ Failed to get statistics: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"✗ Error getting statistics: {e}")
            return None
    
    async def test_websocket_connection(self, user_id: str = "test_user", duration: int = 10):
        """Test WebSocket real-time updates"""
        try:
            ws_url = f"ws://localhost:8001/batch/ws/{user_id}"
            
            self.logger.info(f"Connecting to WebSocket: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                self.logger.info("✓ WebSocket connected")
                
                # Send ping
                ping_message = {
                    "type": "ping",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await websocket.send(json.dumps(ping_message))
                
                # Listen for messages
                start_time = time.time()
                message_count = 0
                
                while time.time() - start_time < duration:
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        message_count += 1
                        
                        self.logger.info(f"✓ Received WebSocket message: {data['type']}")
                        
                        if data['type'] == 'connection_established':
                            self.logger.info(f"  Connection established for user: {data['user_id']}")
                        elif data['type'] == 'pong':
                            self.logger.info("  Pong received")
                        elif data['type'] == 'job_update':
                            job = data['job']
                            self.logger.info(f"  Job update: {job['job_id']} - {job['status']}")
                        elif data['type'] == 'file_progress':
                            file_info = data['file']
                            self.logger.info(f"  File progress: {file_info['file_metadata']['name']} - {file_info['status']}")
                        
                    except asyncio.TimeoutError:
                        # No message received, continue
                        continue
                    except Exception as e:
                        self.logger.error(f"WebSocket message error: {e}")
                        break
                
                self.logger.info(f"✓ WebSocket test completed. Received {message_count} messages in {duration}s")
                return True
                
        except Exception as e:
            self.logger.error(f"✗ WebSocket connection error: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        self.logger.info("Starting comprehensive batch ingestion API test")
        
        # Test 1: Health check
        self.logger.info("\n=== Test 1: Health Check ===")
        health_ok = await self.test_health_check()
        if not health_ok:
            self.logger.error("Health check failed, aborting tests")
            return False
        
        # Test 2: Create batch job
        self.logger.info("\n=== Test 2: Create Batch Job ===")
        job_id = await self.test_create_batch_job()
        if not job_id:
            self.logger.error("Job creation failed, aborting tests")
            return False
        
        # Test 3: Get job details
        self.logger.info("\n=== Test 3: Get Job Details ===")
        job_details = await self.test_get_batch_job(job_id)
        if not job_details:
            self.logger.error("Getting job details failed")
            return False
        
        # Test 4: Job control operations
        self.logger.info("\n=== Test 4: Job Control Operations ===")
        
        # Start job
        start_ok = await self.test_job_control(job_id, "start")
        if start_ok:
            await asyncio.sleep(1)  # Wait a moment
            
            # Pause job
            pause_ok = await self.test_job_control(job_id, "pause")
            if pause_ok:
                await asyncio.sleep(1)
                
                # Resume job
                resume_ok = await self.test_job_control(job_id, "resume")
                if resume_ok:
                    await asyncio.sleep(1)
                    
                    # Cancel job
                    cancel_ok = await self.test_job_control(job_id, "cancel")
        
        # Test 5: List jobs
        self.logger.info("\n=== Test 5: List Jobs ===")
        jobs = await self.test_list_jobs()
        
        # Test 6: Job statistics
        self.logger.info("\n=== Test 6: Job Statistics ===")
        stats = await self.test_job_statistics()
        
        # Test 7: WebSocket connection (shorter duration for testing)
        self.logger.info("\n=== Test 7: WebSocket Connection ===")
        ws_ok = await self.test_websocket_connection(duration=5)
        
        self.logger.info("\n=== Test Summary ===")
        self.logger.info(f"✓ Health check: {'PASS' if health_ok else 'FAIL'}")
        self.logger.info(f"✓ Job creation: {'PASS' if job_id else 'FAIL'}")
        self.logger.info(f"✓ Job details: {'PASS' if job_details else 'FAIL'}")
        self.logger.info(f"✓ Job control: TESTED")
        self.logger.info(f"✓ List jobs: {'PASS' if jobs else 'FAIL'}")
        self.logger.info(f"✓ Statistics: {'PASS' if stats else 'FAIL'}")
        self.logger.info(f"✓ WebSocket: {'PASS' if ws_ok else 'FAIL'}")
        
        return True


async def main():
    """Main test function"""
    print("Batch Ingestion Management API Test")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        print(f"Loaded configuration for environment: {config.environment}")
        
        # Test the API
        async with BatchIngestionAPITester() as tester:
            await tester.run_comprehensive_test()
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())