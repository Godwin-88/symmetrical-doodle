"""
State management system for idempotent execution and checkpoint handling.
Ensures scripts produce identical results when run multiple times.
"""

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
from enum import Enum
import os
import sys

# Cross-platform file locking
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl, use msvcrt instead
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False
    HAS_FCNTL = False


class ExecutionPhase(Enum):
    """Execution phases for checkpoint management"""
    DISCOVERY = "discovery"
    DOWNLOAD = "download"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    AUDIT = "audit"
    COMPLETED = "completed"


class ExecutionStatus(Enum):
    """Execution status for tracking"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FileProcessingState:
    """State tracking for individual file processing"""
    file_id: str
    file_name: str
    file_hash: Optional[str] = None
    discovery_status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    download_status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    parsing_status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    chunking_status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    embedding_status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    storage_status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    last_updated: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)


@dataclass
class ExecutionState:
    """Overall execution state for idempotent runs"""
    execution_id: str
    start_time: datetime
    last_checkpoint: datetime
    current_phase: ExecutionPhase
    configuration_hash: str
    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    total_chunks_created: int = 0
    total_embeddings_generated: int = 0
    file_states: Dict[str, FileProcessingState] = None
    
    def __post_init__(self):
        if self.file_states is None:
            self.file_states = {}


class StateManager:
    """Manages execution state for idempotent operations"""
    
    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or Path(__file__).parent.parent / "data" / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "execution_state.json"
        self.lock_file = self.state_dir / "execution.lock"
        self._lock_fd: Optional[int] = None
        self._current_state: Optional[ExecutionState] = None
    
    def acquire_lock(self, timeout: int = 30) -> bool:
        """Acquire exclusive lock for state management"""
        try:
            self._lock_fd = os.open(str(self.lock_file), os.O_CREAT | os.O_WRONLY)
            
            if HAS_FCNTL:
                # Unix/Linux/macOS
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return True
                    except BlockingIOError:
                        time.sleep(0.1)
            elif HAS_MSVCRT:
                # Windows
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        msvcrt.locking(self._lock_fd, msvcrt.LK_NBLCK, 1)
                        return True
                    except OSError:
                        time.sleep(0.1)
            else:
                # Fallback: just return True (no locking)
                return True
            
            # Timeout reached
            os.close(self._lock_fd)
            self._lock_fd = None
            return False
            
        except Exception as e:
            if self._lock_fd:
                os.close(self._lock_fd)
                self._lock_fd = None
            raise RuntimeError(f"Failed to acquire lock: {e}")
    
    def release_lock(self):
        """Release the execution lock"""
        if self._lock_fd:
            try:
                if HAS_FCNTL:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                elif HAS_MSVCRT:
                    msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)
                os.close(self._lock_fd)
            except:
                pass
            finally:
                self._lock_fd = None
                
            # Clean up lock file
            try:
                self.lock_file.unlink(missing_ok=True)
            except:
                pass
    
    def __enter__(self):
        """Context manager entry"""
        if not self.acquire_lock():
            raise RuntimeError("Failed to acquire execution lock")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release_lock()
    
    def calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for idempotency checking"""
        # Create a normalized config for hashing
        normalized_config = {
            "google_drive_folder_ids": sorted(config.get("google_drive", {}).get("folder_ids", [])),
            "processing_settings": {
                "chunk_size": config.get("processing", {}).get("chunk_size", 1000),
                "chunk_overlap": config.get("processing", {}).get("chunk_overlap", 200),
                "preserve_math": config.get("processing", {}).get("preserve_math", True),
                "use_marker_llm": config.get("processing", {}).get("use_marker_llm", False),
            },
            "embedding_settings": {
                "openai_model": config.get("embeddings", {}).get("openai_model", ""),
                "batch_size": config.get("embeddings", {}).get("batch_size", 32),
            }
        }
        
        config_str = json.dumps(normalized_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def load_state(self) -> Optional[ExecutionState]:
        """Load execution state from disk"""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings back to datetime objects
            data['start_time'] = datetime.fromisoformat(data['start_time'])
            data['last_checkpoint'] = datetime.fromisoformat(data['last_checkpoint'])
            data['current_phase'] = ExecutionPhase(data['current_phase'])
            
            # Convert file states
            file_states = {}
            for file_id, file_data in data.get('file_states', {}).items():
                file_data['discovery_status'] = ExecutionStatus(file_data['discovery_status'])
                file_data['download_status'] = ExecutionStatus(file_data['download_status'])
                file_data['parsing_status'] = ExecutionStatus(file_data['parsing_status'])
                file_data['chunking_status'] = ExecutionStatus(file_data['chunking_status'])
                file_data['embedding_status'] = ExecutionStatus(file_data['embedding_status'])
                file_data['storage_status'] = ExecutionStatus(file_data['storage_status'])
                if file_data['last_updated']:
                    file_data['last_updated'] = datetime.fromisoformat(file_data['last_updated'])
                file_states[file_id] = FileProcessingState(**file_data)
            
            data['file_states'] = file_states
            self._current_state = ExecutionState(**data)
            return self._current_state
            
        except Exception as e:
            print(f"Warning: Failed to load execution state: {e}")
            return None
    
    def save_state(self, state: ExecutionState):
        """Save execution state to disk"""
        try:
            # Convert to serializable format
            data = asdict(state)
            data['start_time'] = state.start_time.isoformat()
            data['last_checkpoint'] = state.last_checkpoint.isoformat()
            data['current_phase'] = state.current_phase.value
            
            # Convert file states
            file_states_data = {}
            for file_id, file_state in state.file_states.items():
                file_data = asdict(file_state)
                file_data['discovery_status'] = file_state.discovery_status.value
                file_data['download_status'] = file_state.download_status.value
                file_data['parsing_status'] = file_state.parsing_status.value
                file_data['chunking_status'] = file_state.chunking_status.value
                file_data['embedding_status'] = file_state.embedding_status.value
                file_data['storage_status'] = file_state.storage_status.value
                if file_state.last_updated:
                    file_data['last_updated'] = file_state.last_updated.isoformat()
                file_states_data[file_id] = file_data
            
            data['file_states'] = file_states_data
            
            # Write atomically
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.state_file)
            self._current_state = state
            
        except Exception as e:
            raise RuntimeError(f"Failed to save execution state: {e}")
    
    def create_new_execution(self, config: Dict[str, Any]) -> ExecutionState:
        """Create a new execution state"""
        execution_id = f"exec_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        config_hash = self.calculate_config_hash(config)
        
        state = ExecutionState(
            execution_id=execution_id,
            start_time=datetime.now(timezone.utc),
            last_checkpoint=datetime.now(timezone.utc),
            current_phase=ExecutionPhase.DISCOVERY,
            configuration_hash=config_hash,
            file_states={}
        )
        
        self.save_state(state)
        return state
    
    def should_resume_execution(self, config: Dict[str, Any]) -> bool:
        """Check if execution should be resumed from checkpoint"""
        current_state = self.load_state()
        if not current_state:
            return False
        
        # Check if configuration has changed
        current_config_hash = self.calculate_config_hash(config)
        if current_state.configuration_hash != current_config_hash:
            print("Configuration has changed, starting fresh execution")
            return False
        
        # Check if execution is already completed
        if current_state.current_phase == ExecutionPhase.COMPLETED:
            print("Previous execution already completed")
            return False
        
        return True
    
    def update_file_state(self, file_id: str, **updates):
        """Update state for a specific file"""
        if not self._current_state:
            raise RuntimeError("No current execution state")
        
        if file_id not in self._current_state.file_states:
            # Create new file state
            self._current_state.file_states[file_id] = FileProcessingState(
                file_id=file_id,
                file_name=updates.get('file_name', file_id)
            )
        
        file_state = self._current_state.file_states[file_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(file_state, key):
                setattr(file_state, key, value)
        
        file_state.last_updated = datetime.now(timezone.utc)
        self.save_state(self._current_state)
    
    def update_phase(self, phase: ExecutionPhase):
        """Update current execution phase"""
        if not self._current_state:
            raise RuntimeError("No current execution state")
        
        self._current_state.current_phase = phase
        self._current_state.last_checkpoint = datetime.now(timezone.utc)
        self.save_state(self._current_state)
    
    def get_files_by_status(self, phase: str, status: ExecutionStatus) -> List[FileProcessingState]:
        """Get files with specific status for a phase"""
        if not self._current_state:
            return []
        
        result = []
        for file_state in self._current_state.file_states.values():
            phase_status = getattr(file_state, f"{phase}_status", ExecutionStatus.NOT_STARTED)
            if phase_status == status:
                result.append(file_state)
        
        return result
    
    def get_files_to_process(self, phase: str) -> List[FileProcessingState]:
        """Get files that need processing for a specific phase"""
        return self.get_files_by_status(phase, ExecutionStatus.NOT_STARTED)
    
    def get_completed_files(self, phase: str) -> List[FileProcessingState]:
        """Get files that have completed processing for a specific phase"""
        return self.get_files_by_status(phase, ExecutionStatus.COMPLETED)
    
    def is_file_processed(self, file_id: str, phase: str) -> bool:
        """Check if a file has been processed for a specific phase"""
        if not self._current_state or file_id not in self._current_state.file_states:
            return False
        
        file_state = self._current_state.file_states[file_id]
        phase_status = getattr(file_state, f"{phase}_status", ExecutionStatus.NOT_STARTED)
        return phase_status == ExecutionStatus.COMPLETED
    
    def mark_execution_completed(self):
        """Mark the entire execution as completed"""
        if not self._current_state:
            raise RuntimeError("No current execution state")
        
        self._current_state.current_phase = ExecutionPhase.COMPLETED
        self._current_state.last_checkpoint = datetime.now(timezone.utc)
        
        # Update statistics
        completed_files = 0
        failed_files = 0
        skipped_files = 0
        
        for file_state in self._current_state.file_states.values():
            if file_state.storage_status == ExecutionStatus.COMPLETED:
                completed_files += 1
            elif file_state.storage_status == ExecutionStatus.FAILED:
                failed_files += 1
            elif file_state.storage_status == ExecutionStatus.SKIPPED:
                skipped_files += 1
        
        self._current_state.files_processed = completed_files
        self._current_state.files_failed = failed_files
        self._current_state.files_skipped = skipped_files
        
        self.save_state(self._current_state)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of current execution"""
        if not self._current_state:
            return {"status": "no_execution"}
        
        return {
            "execution_id": self._current_state.execution_id,
            "start_time": self._current_state.start_time.isoformat(),
            "last_checkpoint": self._current_state.last_checkpoint.isoformat(),
            "current_phase": self._current_state.current_phase.value,
            "files_discovered": self._current_state.files_discovered,
            "files_processed": self._current_state.files_processed,
            "files_failed": self._current_state.files_failed,
            "files_skipped": self._current_state.files_skipped,
            "total_chunks_created": self._current_state.total_chunks_created,
            "total_embeddings_generated": self._current_state.total_embeddings_generated,
            "configuration_hash": self._current_state.configuration_hash
        }
    
    def cleanup_old_states(self, keep_days: int = 7):
        """Clean up old execution state files"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (keep_days * 24 * 60 * 60)
        
        for state_file in self.state_dir.glob("execution_state_*.json"):
            try:
                if state_file.stat().st_mtime < cutoff_time:
                    state_file.unlink()
            except:
                pass
    
    @property
    def current_state(self) -> Optional[ExecutionState]:
        """Get current execution state"""
        return self._current_state