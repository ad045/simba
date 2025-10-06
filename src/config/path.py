from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Path configuration."""
    root_dir: Path = field(default_factory=lambda: Path.cwd().resolve())
    
    def __post_init__(self):
        # Find project root by looking for markers
        current = self.root_dir
        for parent in (current, *current.parents):
            if any((parent / marker).exists() for marker in ("pyproject.toml", "setup.cfg", ".git")):
                self.root_dir = parent
                break
        
        # self.data_dir = self.root_dir / "14_4D_lab_code" / "data/preprocessed/01_first_analysises" # TODO TODO !!! 
        self.data_dir = self.root_dir / "data/preprocessed/01_first_analysises"
        self.output_dir = self.root_dir / "output"
        self.esn_output_dir = self.output_dir / "esn"
        self.gnm_output_dir = self.output_dir / "gnm"
        self.dynamic_gnm_output_dir = self.output_dir / "dynamic_gnm"
