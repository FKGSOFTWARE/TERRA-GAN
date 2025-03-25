from pathlib import Path
import yaml
from path_utils import PathManager

def test_path_manager():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize PathManager
    pm = PathManager(config)

    # Test directory creation
    paths = pm.create_output_structure("NJ05")

    # Print created paths
    for key, path in paths.items():
        print(f"{key}: {path}")
        print(f"Exists: {path.exists()}")

if __name__ == "__main__":
    test_path_manager()
