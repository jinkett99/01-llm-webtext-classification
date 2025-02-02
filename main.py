import os
import subprocess

def run_script(script_name):
    """
    Runs a Python script located in the src folder.
    """
    script_path = os.path.join("src", script_name)
    print(f"Running {script_name}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"{script_name} executed successfully.\n")
    else:
        print(f"Error executing {script_name}:\n{result.stderr}\n")

if __name__ == "__main__":
    scripts = [
        "02_Merge_TTT.py",  # Preprocessing and dataset merging
        "03_Modelling.py",   # Model training and evaluation
        "04_Inference.py"    # Running inference on new data
    ]
    
    for script in scripts:
        run_script(script)
    
    print("All scripts executed successfully!")