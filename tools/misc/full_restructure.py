#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
QuantoniumOS Full Repository Restructure Script
Reorganizes the entire repository according to professional standards
"""

import os
import shutil
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

class QuantoniumRestructure:
    def __init__(self, source_root: str, dry_run: bool = True):
        self.source_root = Path(source_root)
        self.dry_run = dry_run
        self.moves_executed = []
        self.errors = []
        
        # Define the complete restructure mapping
        self.restructure_map = {
            # Core Algorithms
            "src/core/": "algorithms/rft/core/",
            "src/assembly/": "algorithms/rft/kernels/",
            "validation/crypto_benchmarks/": "algorithms/crypto/",
            
            # Operating System Components
            "src/apps/": "os/apps/",
            "src/frontend/": "os/frontend/",
            "src/engine/": "os/engine/",
            "core/safety/": "os/safety/",
            "ui/": "os/frontend/ui/",
            
            # AI Model Management
            "ai/models/": "ai/models/metadata/",
            "encoded_models/": "ai/models/encoded/",
            "decoded_models/": "ai/models/decoded/",
            "hf_models/": "ai/models/cache/",
            "ai/datasets/": "ai/datasets/",
            "ai/inference/": "ai/inference/",
            "ai/training/": "ai/training/",
            
            # Tools Reorganization
            "tools/": "tools/",  # Will be further reorganized
            "dev/": "tools/development/",
            
            # Testing & Validation
            "validation/": "tests/validation/",
            
            # Documentation
            "docs/": "docs/",  # Will be reorganized internally
            
            # Data Management
            "data/config/": "data/config/",
            "data/ai/": "data/datasets/",
            "logs/": "data/cache/logs/",
            "results/": "data/cache/results/",
        }
        
        # Special tool categorization
        self.tool_categories = {
            "compression": [
                "compression_pipeline.py",
                "rft_encode_model.py", 
                "rft_decode_model.py",
                "rft_hybrid_compress.py",
                "rft_hybrid_decode.py",
                "real_*_compressor.py"
            ],
            "model_management": [
                "real_model_downloader.py",
                "real_hf_model_compressor.py",
                "*_to_brain.py",
                "model_analysis.py"
            ],
            "benchmarking": [
                "compare_python_vs_assembly.py",
                "run_all_tests.py",
                "*_benchmarks.py"
            ],
            "development": [
                "generate_*.py",
                "restructure_*.py",
                "legal_compliance_manager.py"
            ]
        }
        
        # Files to keep at root
        self.root_files = [
            "README.md", "LICENSE.md", "requirements.txt", 
            "quantonium_boot.py", ".gitignore", ".gitattributes",
            "pytest.ini", ".github"
        ]
        
        # Research/report files for docs/research/
        self.research_files = [
            "BELL_VIOLATION_ACHIEVEMENT.md",
            "BENCHMARK_REPORT.md", 
            "HYBRID_CODEC.md",
            "FINAL_AI_MODEL_INVENTORY.md",
            "LOSSLESS_CLARIFICATION.md",
            "TERMINOLOGY_CORRECT.md",
            "REQUIRED_BENCHMARKS.md",
            "FIXES_APPLIED.md",
            "enhancement_report.md",
            "203837_19169399_08-13-2025_PEFR.PDF"
        ]

    def analyze_directory(self, path: Path) -> Dict:
        """Analyze directory contents"""
        analysis = {
            "total_files": 0,
            "total_dirs": 0,
            "python_files": 0,
            "subdirs": []
        }
        
        if not path.exists():
            return analysis
            
        for item in path.rglob("*"):
            if item.is_file():
                analysis["total_files"] += 1
                if item.suffix == ".py":
                    analysis["python_files"] += 1
            elif item.is_dir():
                analysis["total_dirs"] += 1
                
        return analysis

    def categorize_tool(self, filename: str) -> str:
        """Categorize a tool file"""
        import fnmatch
        
        for category, patterns in self.tool_categories.items():
            for pattern in patterns:
                if fnmatch.fnmatch(filename, pattern):
                    return category
        return "misc"

    def create_directory_structure(self):
        """Create the new directory structure"""
        new_dirs = [
            "algorithms/rft/core",
            "algorithms/rft/kernels", 
            "algorithms/compression/hybrid",
            "algorithms/compression/vertex",
            "algorithms/crypto",
            "os/apps/symbolic_simulator",
            "os/apps/q_vault",
            "os/apps/q_notes", 
            "os/apps/visualizers",
            "os/frontend/ui",
            "os/engine",
            "os/safety",
            "ai/models/metadata",
            "ai/models/encoded",
            "ai/models/decoded",
            "ai/models/cache",
            "ai/datasets",
            "ai/inference", 
            "ai/training",
            "tools/compression",
            "tools/model_management",
            "tools/benchmarking",
            "tools/development",
            "tests/algorithms",
            "tests/integration", 
            "tests/validation",
            "tests/benchmarks",
            "docs/technical",
            "docs/api",
            "docs/research",
            "docs/user",
            "data/config",
            "data/datasets",
            "data/cache/logs",
            "data/cache/results",
            "deployment/docker",
            "deployment/scripts",
            "deployment/ci"
        ]
        
        for dir_path in new_dirs:
            full_path = self.source_root / dir_path
            if self.dry_run:
                print(f"[DRY RUN] Would create directory: {full_path}")
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {full_path}")

    def move_file_or_directory(self, src: Path, dst: Path):
        """Move a file or directory"""
        try:
            if self.dry_run:
                print(f"[DRY RUN] Would move: {src} → {dst}")
                return True
            
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if src.exists():
                shutil.move(str(src), str(dst))
                self.moves_executed.append((str(src), str(dst)))
                print(f"Moved: {src} → {dst}")
                return True
            else:
                print(f"Warning: Source does not exist: {src}")
                return False
                
        except Exception as e:
            error_msg = f"Error moving {src} to {dst}: {e}"
            self.errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            return False

    def reorganize_tools(self):
        """Reorganize tools by category"""
        tools_dir = self.source_root / "tools"
        if not tools_dir.exists():
            return
            
        for tool_file in tools_dir.glob("*.py"):
            category = self.categorize_tool(tool_file.name)
            new_path = self.source_root / f"tools/{category}" / tool_file.name
            self.move_file_or_directory(tool_file, new_path)

    def reorganize_apps(self):
        """Reorganize apps into logical groups"""
        apps_dir = self.source_root / "src/apps"
        if not apps_dir.exists():
            return
            
        app_groups = {
            "symbolic_simulator": ["symbolic_simulator.py", "quantum_inspired_crypto.py", "quantum_parameter_3d_visualizer.py"],
            "q_vault": ["q_vault.py", "launch_q_vault.py"],
            "q_notes": ["q_notes.py", "launch_q_notes.py"],
            "visualizers": ["rft_visualizer*.py", "rft_validation_visualizer.py"],
            "system": ["qshll_*.py", "real_time_chat_monitor.py", "compressed_model_router.py"],
            "crypto": ["enhanced_rft_crypto.py"],
            "engines": ["baremetal_engine_3d.py"]
        }
        
        for group, files in app_groups.items():
            group_dir = self.source_root / f"os/apps/{group}"
            
            for file_pattern in files:
                import fnmatch
                for app_file in apps_dir.glob("*"):
                    if fnmatch.fnmatch(app_file.name, file_pattern):
                        new_path = group_dir / app_file.name
                        self.move_file_or_directory(app_file, new_path)

    def move_research_files(self):
        """Move research files to docs/research/"""
        research_dir = self.source_root / "docs/research"
        
        for filename in self.research_files:
            src_path = self.source_root / filename
            if src_path.exists():
                dst_path = research_dir / filename
                self.move_file_or_directory(src_path, dst_path)

    def execute_restructure(self):
        """Execute the full restructure"""
        print("=== QuantoniumOS Full Repository Restructure ===")
        print(f"Source: {self.source_root}")
        print(f"Dry Run: {self.dry_run}")
        print()
        
        # Create new directory structure
        print("Creating directory structure...")
        self.create_directory_structure()
        print()
        
        # Execute primary moves
        print("Executing primary directory moves...")
        for src_pattern, dst_pattern in self.restructure_map.items():
            src_path = self.source_root / src_pattern
            dst_path = self.source_root / dst_pattern
            
            if src_path.exists():
                self.move_file_or_directory(src_path, dst_path)
        print()
        
        # Reorganize tools by category  
        print("Reorganizing tools by category...")
        self.reorganize_tools()
        print()
        
        # Reorganize apps into logical groups
        print("Reorganizing apps into logical groups...")
        self.reorganize_apps()
        print()
        
        # Move research files
        print("Moving research files...")
        self.move_research_files()
        print()
        
        # Summary
        print("=== Restructure Summary ===")
        print(f"Moves executed: {len(self.moves_executed)}")
        print(f"Errors encountered: {len(self.errors)}")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")
        
        return len(self.errors) == 0

def main():
    parser = argparse.ArgumentParser(description="QuantoniumOS Full Repository Restructure")
    parser.add_argument("--source", default="/workspaces/quantoniumos", 
                       help="Source repository path")
    parser.add_argument("--execute", action="store_true", 
                       help="Execute moves (default is dry-run)")
    
    args = parser.parse_args()
    
    restructure = QuantoniumRestructure(
        source_root=args.source,
        dry_run=not args.execute
    )
    
    success = restructure.execute_restructure()
    
    if success:
        print("\n✅ Restructure completed successfully!")
    else:
        print("\n❌ Restructure completed with errors.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())