#!/usr/bin/env python3
"""
Standalone script to push a trained model to HuggingFace Hub
Usage: python scripts/push_model_to_hub.py --model_dir <path> --repo_id <repo> --branch <branch>
"""

import argparse
import os
from datetime import datetime
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()


def push_model_to_hub(model_dir, repo_id, branch="main", token=None):
    """
    Push a trained model to HuggingFace Hub
    
    Args:
        model_dir: Local directory containing the model files
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        branch: Branch name to push to (default: "main")
        token: HuggingFace token (if None, will use HF_TOKEN env var)
    """
    
    # Get token from environment if not provided
    if token is None:
        token = os.getenv('HF_TOKEN')
        if not token:
            raise ValueError("No HuggingFace token provided. Set HF_TOKEN environment variable or pass --token")
    
    # Verify model directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory does not exist: {model_dir}")
    
    print(f"\n{'='*80}")
    print(f"Pushing Model to HuggingFace Hub")
    print(f"{'='*80}")
    print(f" Model directory: {model_dir}")
    print(f" Repository: {repo_id}")
    print(f" Branch: {branch}")
    print(f"{'='*80}\n")
    
    try:
        api = HfApi()
        
        # Check if repository exists, create if it doesn't
        print(f"Checking if repository exists...")
        repo_exists = False
        try:
            api.repo_info(repo_id=repo_id, repo_type="model", token=token)
            print(f" Repository exists: {repo_id}")
            repo_exists = True
        except Exception as e:
            # If we get a 404, repo doesn't exist. If 403, we don't have read access but it might exist
            if "404" in str(e):
                print(f" Repository doesn't exist. Creating: {repo_id}")
                try:
                    create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
                    print(f" Repository created: {repo_id}")
                    repo_exists = True
                except Exception as create_error:
                    print(f" Could not create repository: {create_error}")
                    print(f" If the repository already exists, we'll try to upload anyway...")
                    repo_exists = True  # Assume it exists and try to upload
            else:
                # Might be 403 or other error - assume repo exists and try to upload
                print(f" Could not verify repository (might be permission issue)")
                print(f" Assuming repository exists and attempting upload...")
                repo_exists = True
        
        # Upload model files to the specified branch
        print(f"\n Uploading model files...")
        print(f"   This may take a few minutes depending on model size...")
        
        # Try to create the branch if it doesn't exist (will be ignored if it already exists)
        if branch != "main":
            try:
                print(f" Creating branch '{branch}' (if it doesn't exist)...")
                api.create_branch(repo_id=repo_id, branch=branch, repo_type="model", token=token)
                print(f" Branch '{branch}' ready")
            except Exception as branch_error:
                # Branch might already exist, which is fine
                if "already exists" in str(branch_error).lower() or "reference already exists" in str(branch_error).lower():
                    print(f" Branch '{branch}' already exists")
                else:
                    print(f" Note: {branch_error}")
                    print(f" Continuing with upload anyway...")
        
        # Upload using HuggingFace Hub API (same as official documentation)
        # revision parameter specifies the branch name
        print(f"   Uploading to branch: {branch}")
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            token=token,
            commit_message=f"Upload model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        print(f"\n{'='*80}")
        print(f" SUCCESS! Model pushed to HuggingFace Hub")
        print(f"{'='*80}")
        print(f" View at: https://huggingface.co/{repo_id}/tree/{branch}")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f" ERROR: Failed to push model to HuggingFace Hub {e}")
        print(f"{'='*80}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Push a trained model to HuggingFace Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push ModernBERT model
  python scripts/push_model_to_hub.py \\
    --model_dir outputs/checkpoints/modernbert_large_1116/final_model \\
    --repo_id metr-evals/content-filtering-modernBERT \\
    --branch v1-1116

  # Push Gemma model
  python scripts/push_model_to_hub.py \\
    --model_dir outputs/checkpoints/gemma3_20k_balanced_it_v2/final_model \\
    --repo_id metr-evals/content-filtering-models \\
    --branch gemma-it-balanced-1114

  # Use custom token
  python scripts/push_model_to_hub.py \\
    --model_dir outputs/checkpoints/my_model \\
    --repo_id username/my-model \\
    --token hf_xxxxxxxxxxxxx
        """
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Path to the local model directory containing model files'
    )
    
    parser.add_argument(
        '--repo_id',
        type=str,
        required=True,
        help='HuggingFace repository ID (e.g., "username/model-name")'
    )
    
    parser.add_argument(
        '--branch',
        type=str,
        default='main',
        help='Branch name to push to (default: main)'
    )
    
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace token (if not provided, uses HF_TOKEN env var)'
    )
    
    args = parser.parse_args()
    
    # Push the model
    success = push_model_to_hub(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        branch=args.branch,
        token=args.token
    )
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

