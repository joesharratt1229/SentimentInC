import os
import tomli
import shutil
from pathlib import Path

def extract_dependencies(pyproject_path, output_dir=None):
    """
    Extract dependencies from a Poetry pyproject.toml file and store them
    in a Python file in a newly created directory.
    
    Args:
        pyproject_path (str): Path to the pyproject.toml file
        output_dir (str, optional): Directory to store the dependencies. 
                                    If None, creates 'dependencies_output'
    
    Returns:
        str: Path to the created dependencies.py file
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = "dependencies_output"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read and parse the pyproject.toml file
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomli.load(f)
    
    # Extract dependencies
    dependencies = []
    
    # Check if it's a poetry project
    if "tool" in pyproject_data and "poetry" in pyproject_data["tool"]:
        poetry_deps = pyproject_data["tool"]["poetry"].get("dependencies", {})
        
        # Process each dependency
        for package, constraint in poetry_deps.items():
            # Skip python dependency
            if package.lower() == "python":
                continue
                
            if isinstance(constraint, str):
                # Simple version constraint
                dependencies.append(f"{package}=={constraint}")
            elif isinstance(constraint, dict):
                # Complex version constraint
                if "version" in constraint:
                    dependencies.append(f"{package}=={constraint['version']}")
                else:
                    # If no specific version, just include the package name
                    dependencies.append(package)
    
    # Create a Python file with the dependencies list
    output_file = os.path.join(output_dir, "dependencies.py")
    with open(output_file, "w") as f:
        f.write(f"dependencies = {repr(dependencies)}\n")
    
    # Copy the pyproject.toml file to the output directory for reference
    shutil.copy2(pyproject_path, os.path.join(output_dir, "pyproject.toml"))
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract dependencies from a Poetry pyproject.toml file")
    parser.add_argument("pyproject", help="Path to the pyproject.toml file")
    parser.add_argument("--output", "-o", help="Output directory (default: dependencies_output)")
    
    args = parser.parse_args()
    
    result = extract_dependencies(args.pyproject, args.output)
    print(f"Dependencies extracted to {result}")
