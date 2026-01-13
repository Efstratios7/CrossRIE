import subprocess
import sys
import os

def run_command(command, cwd=None):
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        sys.exit(1)

def main():
    print("Creating Conda environment 'crossrie_env'...")
    # Create environment from yml, removing if exists first could be safer, but let's try update or prune
    # Or just try create without force, and handle error?
    # --force is deprecated/removed in some contexts. 
    # Let's try `conda env update --file environment.yml --prune` which is generally idempotent
    run_command("conda env update --file environment.yml --prune")
    
    # Initialize shell for conda (this might be tricky in a subprocess, so we might need users to activate)
    # Instead, we will use the python executable directly from the new environment to install the package
    
    # Assuming standard conda path structure (this can vary by OS/Version, but commonly in envs/)
    # We will try to find where it is.
    
    # Getting conda info to find env path
    try:
        output = subprocess.check_output("conda info --base", shell=True).decode().strip()
        base_path = output
    except:
        print("Could not find conda base path.")
        sys.exit(1)
        
    env_python = os.path.join(base_path, "envs", "crossrie_env", "bin", "python")
    if not os.path.exists(env_python):
        # Try looking in local envs dir if configured
        env_python = os.path.join(os.getcwd(), "envs", "crossrie_env", "bin", "python")
        
    # Fallback: ask conda for the env path
    try:
        envs_output = subprocess.check_output("conda env list", shell=True).decode()
        for line in envs_output.splitlines():
            if "crossrie_env" in line:
                parts = line.split()
                if len(parts) >= 2:
                    env_path = parts[-1]
                    env_python = os.path.join(env_path, "bin", "python")
                    break
    except:
        pass

    print(f"Target Python: {env_python}")
    
    if not os.path.exists(env_python):
        print("Could not locate python executable for crossrie_env. Please activate manualy and run 'pip install -e .'")
        return

    print("Installing package in editable mode...")
    run_command(f"'{env_python}' -m pip install -e .")
    
    print("Registering kernel...")
    run_command(f"'{env_python}' -m ipykernel install --user --name=crossrie_env --display-name 'Python (crossrie_env)'")

    print("\nSetup complete!")
    print("To use: conda activate crossrie_env")

if __name__ == "__main__":
    main()
