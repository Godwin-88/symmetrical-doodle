#!/usr/bin/env python3
"""
Setup script for LLM and RAG capabilities in the Intelligence Layer.
This script helps users configure and test their LLM setup.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"   GPU Count: {gpu_count}")
            return True
        else:
            print("⚠️  No GPU detected. Local LLM will run on CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def download_model(model_name: str, model_path: str):
    """Download a model using Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
        
        print(f"📥 Downloading {model_name} to {model_path}")
        
        # Create model directory
        os.makedirs(model_path, exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded successfully to {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False


def setup_local_llm():
    """Setup local LLM."""
    print("\n🤖 Setting up Local LLM")
    print("=" * 50)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Recommend models based on GPU availability
    if has_gpu:
        recommended_models = [
            ("meta-llama/Llama-3.1-8B-Instruct", "General purpose, good performance"),
            ("microsoft/Phi-3-mini-4k-instruct", "Smaller, faster inference"),
            ("Qwen/Qwen2.5-7B-Instruct", "Good for financial tasks"),
        ]
    else:
        recommended_models = [
            ("microsoft/Phi-3-mini-4k-instruct", "Optimized for CPU"),
            ("Qwen/Qwen2.5-3B-Instruct", "Smaller model for CPU"),
        ]
    
    print("\n📋 Recommended Models:")
    for i, (model, desc) in enumerate(recommended_models, 1):
        print(f"  {i}. {model}")
        print(f"     {desc}")
    
    choice = input(f"\nSelect model (1-{len(recommended_models)}) or enter custom model name: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(recommended_models):
        model_name = recommended_models[int(choice) - 1][0]
    else:
        model_name = choice
    
    # Set model path
    model_path = f"./models/{model_name.split('/')[-1]}"
    
    # Download model
    if input(f"\nDownload {model_name}? (y/n): ").lower() == 'y':
        download_model(model_name, model_path)
    
    return model_name, model_path


def setup_external_apis():
    """Setup external API keys."""
    print("\n🌐 Setting up External APIs")
    print("=" * 50)
    
    apis = {
        "OpenAI": {
            "key": "OPENAI_API_KEY",
            "url": "https://platform.openai.com/api-keys",
            "free": False,
            "description": "GPT-4, GPT-3.5 models"
        },
        "Groq": {
            "key": "GROQ_API_KEY", 
            "url": "https://console.groq.com/keys",
            "free": True,
            "description": "Fast Llama inference (FREE tier available)"
        },
        "Together AI": {
            "key": "TOGETHER_API_KEY",
            "url": "https://api.together.xyz/settings/api-keys", 
            "free": True,
            "description": "Various open-source models (FREE credits)"
        },
        "Anthropic": {
            "key": "ANTHROPIC_API_KEY",
            "url": "https://console.anthropic.com/",
            "free": False,
            "description": "Claude models"
        }
    }
    
    print("Available API providers:")
    for name, info in apis.items():
        status = "FREE tier" if info["free"] else "Paid"
        print(f"  • {name}: {info['description']} ({status})")
        print(f"    Get API key: {info['url']}")
    
    print("\n💡 Recommendation: Start with Groq (free) for testing")
    
    # Check existing API keys
    existing_keys = {}
    for name, info in apis.items():
        key = os.getenv(info["key"])
        if key:
            existing_keys[name] = key[:8] + "..." if len(key) > 8 else key
    
    if existing_keys:
        print(f"\n✅ Found existing API keys: {list(existing_keys.keys())}")
    
    return apis


def create_env_file():
    """Create .env file with LLM configuration."""
    print("\n📝 Creating .env configuration")
    print("=" * 50)
    
    env_path = Path(".env")
    env_llm_example = Path(".env.llm.example")
    
    if env_path.exists():
        if input("⚠️  .env file exists. Overwrite? (y/n): ").lower() != 'y':
            return
    
    if env_llm_example.exists():
        # Copy example file
        with open(env_llm_example, 'r') as f:
            content = f.read()
        
        with open(env_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Created .env file from {env_llm_example}")
        print("📝 Please edit .env file to add your API keys and configure settings")
    else:
        print("❌ .env.llm.example not found")


def test_setup():
    """Test the LLM setup."""
    print("\n🧪 Testing LLM Setup")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        import torch
        import transformers
        import sentence_transformers
        import chromadb
        print("✅ All required packages imported successfully")
        
        # Test GPU
        if torch.cuda.is_available():
            print("✅ GPU available for acceleration")
        else:
            print("⚠️  No GPU available (CPU mode)")
        
        # Test API keys
        print("\n🔑 Checking API keys...")
        api_keys = {
            "OpenAI": os.getenv("OPENAI_API_KEY"),
            "Groq": os.getenv("GROQ_API_KEY"),
            "Together": os.getenv("TOGETHER_API_KEY"),
            "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        }
        
        for name, key in api_keys.items():
            if key:
                print(f"✅ {name}: {key[:8]}...")
            else:
                print(f"⚠️  {name}: Not configured")
        
        print("\n✅ Setup test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Run: pip install -e .")
    except Exception as e:
        print(f"❌ Test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Setup LLM and RAG capabilities")
    parser.add_argument("--local", action="store_true", help="Setup local LLM only")
    parser.add_argument("--apis", action="store_true", help="Setup external APIs only")
    parser.add_argument("--test", action="store_true", help="Test setup only")
    parser.add_argument("--env", action="store_true", help="Create .env file only")
    
    args = parser.parse_args()
    
    print("🚀 LLM and RAG Setup for Intelligence Layer")
    print("=" * 60)
    
    if args.test:
        test_setup()
        return
    
    if args.env:
        create_env_file()
        return
    
    if args.local:
        setup_local_llm()
        return
    
    if args.apis:
        setup_external_apis()
        return
    
    # Full setup
    print("This script will help you set up LLM and RAG capabilities.")
    print("You can choose between:")
    print("  1. Local LLM (private, no API costs, requires GPU)")
    print("  2. External APIs (easier setup, some free options)")
    print("  3. Both (recommended for production)")
    
    choice = input("\nWhat would you like to set up? (1/2/3): ").strip()
    
    if choice in ["1", "3"]:
        setup_local_llm()
    
    if choice in ["2", "3"]:
        setup_external_apis()
    
    create_env_file()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python scripts/setup-llm.py --test")
    print("3. Start the intelligence layer: python -m intelligence_layer.main")
    print("4. Test the new endpoints at http://localhost:8000/docs")


if __name__ == "__main__":
    main()