import os
import subprocess
import sys
import time

def safe_print(text):
    if text is None: return
    try:
        # Try to print normally
        print(text)
    except UnicodeEncodeError:
        # Fallback for Windows terminal: replace non-ascii chars
        print(text.encode('ascii', errors='replace').decode('ascii'))

def run_step(name, script_path):
    print(f"\nSTEP: {name}")
    # Run and let it print directly to the console (which handles encoding better sometimes)
    # or capture and print using ascii
    result = subprocess.run([sys.executable, script_path], text=True, errors='replace')
    if result.returncode != 0:
        print(f"FAILED: {name}")
        return False
    print(f"SUCCESS: {name}")
    return True

def main():
    print("=" * 40)
    print("NALAMAI CHENNAI PIPELINE")
    print("=" * 40)
    
    steps = [
        ("Clean", "pipeline/01_clean.py"),
        ("Spatial", "pipeline/02_spatial_join.py"),
        ("Features", "pipeline/03_features.py"),
        ("Model", "pipeline/04_model.py"),
        ("Validate", "pipeline/validate.py"),
        ("Map", "pipeline/05_map_connect.py")
    ]
    
    for name, script in steps:
        if not run_step(name, script):
            sys.exit(1)
            
    print("\nPIPELINE COMPLETE")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
