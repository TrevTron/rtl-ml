#!/usr/bin/env python3
"""
Hugging Face Login Helper
Run this to authenticate with Hugging Face and upload the dataset
"""

from huggingface_hub import login, HfApi
import getpass

def hf_login():
    """Login to Hugging Face"""
    print("=" * 60)
    print("Hugging Face Login")
    print("=" * 60)
    print("\nGet your token from: https://huggingface.co/settings/tokens")
    print("(Create a new token with 'write' permissions if you don't have one)\n")
    
    token = getpass.getpass("Enter your HF token: ")
    
    try:
        login(token=token, add_to_git_credential=True)
        print("\n✅ Login successful!")
        
        # Test API access
        api = HfApi()
        user = api.whoami()
        print(f"✅ Authenticated as: {user['name']}")
        
        return True
    except Exception as e:
        print(f"\n❌ Login failed: {e}")
        return False

if __name__ == "__main__":
    hf_login()
