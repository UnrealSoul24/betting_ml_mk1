import sys
import os

print("Current ID:", os.getcwd())
print("Script loc:", __file__)

# Add project root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Project Root:", root)
sys.path.append(root)

print("Sys Path:", sys.path)

try:
    import src
    print("Import src successful:", src)
except Exception as e:
    print("Import src failed:", e)

try:
    from src.feature_engineer import FeatureEngineer
    print("Import FeatureEngineer successful")
except Exception as e:
    print("Import FeatureEngineer failed:", e)
