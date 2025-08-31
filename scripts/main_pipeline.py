# scripts/run_full_pipeline.py
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.preprocessing.connectome_processing import preprocess_connectomes
from src.models.gnm.generator import generate_gnm
from src.models.esn.memory_capacity import evaluate_memory_capacity
