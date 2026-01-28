#!/usr/bin/env python3
"""
Environment setup script for Google Drive Knowledge Base Ingestion System.
Creates virtual environment and installs dependencies.
"""

import os
import sys
import subprocess
import venv
from pathlib import Path
from typing import List, Optional


class EnvironmentSetup:
    """Handles virtual environment setup and dependency installation"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
        
    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment"""
        try:
            print(f"Creating virtual environment at {self.venv_path}")
            venv.create(self.venv_path, with_pip=True, clear=True)
            print("✓ Virtual environment created successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to create virtual environment: {e}")
            return False
    
    def get_pip_executable(self) -> Path:
        """Get pip executable path for the virtual environment"""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def get_python_executable(self) -> Path:
        """Get Python executable path for the virtual environment"""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        try:
            pip_path = self.get_pip_executable()
            print("Upgrading pip to latest version...")
            subprocess.run([
                str(pip_path), "install", "--upgrade", "pip"
            ], check=True, capture_output=True, text=True)
            print("✓ Pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to upgrade pip: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install dependencies from requirements.txt"""
        if not self.requirements_file.exists():
            print(f"✗ Requirements file not found: {self.requirements_file}")
            return False
            
        try:
            pip_path = self.get_pip_executable()
            print(f"Installing dependencies from {self.requirements_file}")
            
            # Install requirements
            subprocess.run([
                str(pip_path), "install", "-r", str(self.requirements_file)
            ], check=True)
            
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            return False
    
    def install_development_dependencies(self) -> bool:
        """Install additional development dependencies"""
        dev_packages = [
            "pytest-cov",
            "pytest-xdist",
            "pre-commit",
            "flake8",
            "bandit",
        ]
        
        try:
            pip_path = self.get_pip_executable()
            print("Installing development dependencies...")
            
            subprocess.run([
                str(pip_path), "install"
            ] + dev_packages, check=True)
            
            print("✓ Development dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install development dependencies: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that key packages are installed correctly"""
        key_packages = [
            "google-api-python-client",
            "supabase",
            "openai",
            "sentence-transformers",
            "marker-pdf",
            "pymupdf",
            "structlog",
            "pydantic",
            "pytest",
            "hypothesis"
        ]
        
        try:
            python_path = self.get_python_executable()
            print("Verifying package installation...")
            
            for package in key_packages:
                result = subprocess.run([
                    str(python_path), "-c", f"import {package.replace('-', '_')}"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✓ {package}")
                else:
                    print(f"✗ {package} - {result.stderr.strip()}")
                    return False
            
            print("✓ All key packages verified successfully")
            return True
        except Exception as e:
            print(f"✗ Package verification failed: {e}")
            return False
    
    def create_activation_scripts(self):
        """Create convenient activation scripts"""
        
        # Unix/Linux/macOS activation script
        unix_script = self.project_root / "activate.sh"
        unix_content = f"""#!/bin/bash
# Activation script for Google Drive Knowledge Base Ingestion environment

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
    echo "Python: $(which python)"
    echo "Pip: $(which pip)"
else
    echo "✗ Virtual environment not found at $VENV_PATH"
    exit 1
fi
"""
        
        with open(unix_script, 'w') as f:
            f.write(unix_content)
        unix_script.chmod(0o755)
        
        # Windows activation script
        windows_script = self.project_root / "activate.bat"
        windows_content = f"""@echo off
REM Activation script for Google Drive Knowledge Base Ingestion environment

set SCRIPT_DIR=%~dp0
set VENV_PATH=%SCRIPT_DIR%venv

if exist "%VENV_PATH%\\Scripts\\activate.bat" (
    call "%VENV_PATH%\\Scripts\\activate.bat"
    echo ✓ Virtual environment activated
    echo Python: %VIRTUAL_ENV%\\Scripts\\python.exe
    echo Pip: %VIRTUAL_ENV%\\Scripts\\pip.exe
) else (
    echo ✗ Virtual environment not found at %VENV_PATH%
    exit /b 1
)
"""
        
        with open(windows_script, 'w') as f:
            f.write(windows_content)
        
        print("✓ Activation scripts created")
    
    def setup_complete_environment(self, include_dev: bool = True) -> bool:
        """Complete environment setup process"""
        print("=" * 60)
        print("Google Drive Knowledge Base Ingestion - Environment Setup")
        print("=" * 60)
        
        steps = [
            ("Creating virtual environment", self.create_virtual_environment),
            ("Upgrading pip", self.upgrade_pip),
            ("Installing dependencies", self.install_dependencies),
        ]
        
        if include_dev:
            steps.append(("Installing development dependencies", self.install_development_dependencies))
        
        steps.extend([
            ("Verifying installation", self.verify_installation),
            ("Creating activation scripts", lambda: (self.create_activation_scripts(), True)[1])
        ])
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                print(f"\n✗ Setup failed at step: {step_name}")
                return False
        
        print("\n" + "=" * 60)
        print("✓ Environment setup completed successfully!")
        print("=" * 60)
        print("\nTo activate the environment:")
        print("  Unix/Linux/macOS: source activate.sh")
        print("  Windows: activate.bat")
        print("\nTo run tests:")
        print("  pytest tests/")
        print("\nTo start ingestion:")
        print("  python -m knowledge_ingestion.main")
        
        return True


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Google Drive Knowledge Base Ingestion environment")
    parser.add_argument("--no-dev", action="store_true", help="Skip development dependencies")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup(args.project_root)
    success = setup.setup_complete_environment(include_dev=not args.no_dev)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()