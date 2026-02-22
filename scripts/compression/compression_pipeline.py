#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
HuggingFace Model Compression Pipeline
=====================================
Automated pipeline for downloading, compressing, validating, and storing HuggingFace models.
This scales the proven DialoGPT-small process to handle multiple models efficiently.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add QuantoniumOS paths (relative to this file's location)
_this_dir = Path(__file__).resolve().parent
_project_root = _this_dir.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "tools"))

from real_hf_model_compressor import HuggingFaceRFTCompressor
from validate_compressed_model import CompressedModelValidator

class CompressionPipeline:
    """Automated compression pipeline for multiple HuggingFace models"""
    
    def __init__(self):
        self.compressor = HuggingFaceRFTCompressor()
        self.validator = CompressedModelValidator()
        self.results = []
        self.base_dir = _project_root
        
        # Pipeline configuration
        self.config = {
            "max_concurrent_downloads": 1,  # Avoid overwhelming the system
            "compression_timeout": 300,     # 5 minutes per model
            "validation_enabled": True,
            "auto_cleanup": False,          # Keep original downloads
            "storage_location": self.base_dir / "data/parameters/quantum_models"
        }
    
    def get_priority_models(self) -> List[Dict]:
        """Get prioritized list of models to compress next"""
        
        # Models from the original database, prioritized by size/importance
        priority_models = [
            {
                "id": "microsoft/DialoGPT-medium",
                "params": "345M",
                "size": "~1.4GB",
                "target_ratio": "1000:1",
                "priority": 1,
                "reason": "Next size up from DialoGPT-small"
            },
            {
                "id": "EleutherAI/gpt-neo-125M", 
                "params": "125M",
                "size": "~500MB",
                "target_ratio": "1000:1",
                "priority": 2,
                "reason": "Small, fast compression test"
            },
            {
                "id": "microsoft/CodeBERT-base",
                "params": "125M", 
                "size": "~500MB",
                "target_ratio": "1000:1",
                "priority": 3,
                "reason": "Code-focused model (different domain)"
            },
            {
                "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "params": "1.1B",
                "size": "~2.2GB", 
                "target_ratio": "1000:1",
                "priority": 4,
                "reason": "Larger model test (1B+ parameters)"
            },
            {
                "id": "EleutherAI/gpt-neo-1.3B",
                "params": "1.3B",
                "size": "~5GB",
                "target_ratio": "1000:1", 
                "priority": 5,
                "reason": "Large model validation"
            }
        ]
        
        return priority_models
    
    def download_model(self, model_info: Dict) -> Dict:
        """Download a HuggingFace model"""
        
        model_id = model_info["id"]
        print(f"\n🔄 Downloading {model_id}...")
        
        try:
            from huggingface_hub import snapshot_download
            
            # Create download directory
            download_dir = self.base_dir / "hf_models/downloaded" / model_id.replace("/", "_")
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model
            start_time = time.time()
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(download_dir)
            )
            download_time = time.time() - start_time
            
            # Check download size
            total_size = 0
            file_count = 0
            for file_path in download_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            result = {
                "status": "success",
                "local_path": str(download_dir),
                "download_time": download_time,
                "file_count": file_count,
                "total_size_mb": total_size / 1024 / 1024,
                "model_id": model_id
            }
            
            print(f"✅ Downloaded {model_id}")
            print(f"   📁 Path: {download_dir}")
            print(f"   📊 Size: {result['total_size_mb']:.1f} MB")
            print(f"   ⏱️ Time: {download_time:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Download failed for {model_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_id": model_id
            }
    
    def compress_model(self, model_info: Dict, download_result: Dict) -> Dict:
        """Compress a downloaded model"""
        
        if download_result["status"] != "success":
            return {"status": "skipped", "reason": "Download failed"}
        
        model_id = model_info["id"]
        model_path = download_result["local_path"]
        
        print(f"\n🔄 Compressing {model_id}...")
        
        try:
            # Apply compression
            start_time = time.time()
            compression_result = self.compressor.compress_huggingface_model(model_path, model_id)
            compression_time = time.time() - start_time
            
            # Save compressed model
            safe_name = model_id.replace("/", "_").replace("-", "_")
            output_path = self.config["storage_location"] / f"{safe_name}_compressed.pkl.gz"
            saved_path = self.compressor.save_compressed_model(compression_result, str(output_path))
            
            # Add timing and storage info
            compression_result["compression_time"] = compression_time
            compression_result["compressed_file_path"] = saved_path
            compression_result["pipeline_processed"] = True
            
            print(f"✅ Compressed {model_id}")
            print(f"   📊 Ratio: {compression_result['compression_ratio']}")
            print(f"   💾 Saved: {Path(saved_path).name}")
            print(f"   ⏱️ Time: {compression_time:.1f}s")
            
            return compression_result
            
        except Exception as e:
            print(f"❌ Compression failed for {model_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_id": model_id
            }
    
    def validate_compressed_model(self, model_info: Dict, compression_result: Dict) -> Dict:
        """Validate compressed model quality"""
        
        if not self.config["validation_enabled"]:
            return {"status": "skipped", "reason": "Validation disabled"}
        
        if compression_result.get("status") == "error":
            return {"status": "skipped", "reason": "Compression failed"}
        
        model_id = model_info["id"]
        compressed_path = compression_result.get("compressed_file_path")
        
        if not compressed_path or not Path(compressed_path).exists():
            return {"status": "error", "reason": "Compressed file not found"}
        
        print(f"\n🧪 Validating {model_id}...")
        
        try:
            # Load and analyze compressed model
            compressed_data = self.validator.load_compressed_model(compressed_path)
            
            # Simulate performance analysis
            performance_result = self.validator.simulate_compressed_performance(compressed_data)
            
            validation_result = {
                "status": "completed",
                "model_id": model_id,
                "compressed_file": compressed_path,
                "performance": performance_result,
                "validation_time": datetime.now().isoformat()
            }
            
            print(f"✅ Validated {model_id}")
            if performance_result.get("status") == "simulated":
                score = performance_result.get("estimated_performance", 0)
                rating = performance_result.get("quality_rating", "Unknown")
                print(f"   📊 Performance: {score:.3f}")
                print(f"   ⭐ Rating: {rating}")
            
            return validation_result
            
        except Exception as e:
            print(f"❌ Validation failed for {model_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_id": model_id
            }
    
    def process_single_model(self, model_info: Dict) -> Dict:
        """Process a single model through the complete pipeline"""
        
        print(f"\n{'='*60}")
        print(f"🎯 PROCESSING: {model_info['id']}")
        print(f"   📊 Parameters: {model_info['params']}")
        print(f"   📊 Target ratio: {model_info['target_ratio']}")
        print(f"{'='*60}")
        
        pipeline_result = {
            "model_info": model_info,
            "pipeline_start": datetime.now().isoformat(),
            "stages": {}
        }
        
        # Stage 1: Download
        download_result = self.download_model(model_info)
        pipeline_result["stages"]["download"] = download_result
        
        # Stage 2: Compress
        compression_result = self.compress_model(model_info, download_result)
        pipeline_result["stages"]["compression"] = compression_result
        
        # Stage 3: Validate
        validation_result = self.validate_compressed_model(model_info, compression_result)
        pipeline_result["stages"]["validation"] = validation_result
        
        # Overall result
        pipeline_result["pipeline_end"] = datetime.now().isoformat()
        pipeline_result["overall_status"] = self._determine_overall_status(pipeline_result)
        
        return pipeline_result
    
    def _determine_overall_status(self, pipeline_result: Dict) -> str:
        """Determine overall pipeline status"""
        
        stages = pipeline_result["stages"]
        
        if stages.get("download", {}).get("status") != "success":
            return "download_failed"
        elif stages.get("compression", {}).get("status") == "error":
            return "compression_failed"
        elif stages.get("validation", {}).get("status") == "error":
            return "validation_failed"
        elif stages.get("compression", {}).get("compression_ratio"):
            return "success"
        else:
            return "incomplete"
    
    def run_batch_processing(self, max_models: int = 3) -> List[Dict]:
        """Run batch processing of multiple models"""
        
        print("🚀 STARTING BATCH COMPRESSION PIPELINE")
        print("=" * 60)
        
        # Get models to process
        priority_models = self.get_priority_models()
        models_to_process = priority_models[:max_models]
        
        print(f"📋 Processing {len(models_to_process)} models:")
        for i, model in enumerate(models_to_process, 1):
            print(f"   {i}. {model['id']} ({model['params']})")
        
        # Process each model
        batch_results = []
        for model_info in models_to_process:
            try:
                result = self.process_single_model(model_info)
                batch_results.append(result)
                
                # Brief pause between models
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\n⚠️ Pipeline interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Pipeline error for {model_info['id']}: {e}")
                batch_results.append({
                    "model_info": model_info,
                    "overall_status": "pipeline_error",
                    "error": str(e)
                })
        
        # Save batch results
        self._save_batch_results(batch_results)
        
        # Print summary
        self._print_batch_summary(batch_results)
        
        return batch_results
    
    def _save_batch_results(self, batch_results: List[Dict]):
        """Save batch processing results"""
        
        results_file = self.base_dir / "results" / f"batch_compression_results_{int(time.time())}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        batch_summary = {
            "pipeline_run": {
                "start_time": batch_results[0]["pipeline_start"] if batch_results else None,
                "end_time": datetime.now().isoformat(),
                "models_processed": len(batch_results),
                "config": self.config
            },
            "results": batch_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"\n💾 Batch results saved: {results_file}")
    
    def _print_batch_summary(self, batch_results: List[Dict]):
        """Print summary of batch processing"""
        
        print(f"\n🎯 BATCH PROCESSING SUMMARY")
        print("=" * 40)
        
        success_count = 0
        total_original_params = 0
        total_compressed_params = 0
        
        for result in batch_results:
            model_id = result.get("model_info", {}).get("id", "Unknown")
            status = result.get("overall_status", "unknown")
            
            print(f"\n📋 {model_id}")
            print(f"   Status: {status.upper()}")
            
            if status == "success":
                success_count += 1
                compression = result.get("stages", {}).get("compression", {})
                if compression:
                    orig_params = compression.get("original_parameters", 0)
                    comp_params = compression.get("compressed_parameters", 0)
                    ratio = compression.get("compression_ratio", "N/A")
                    
                    total_original_params += orig_params
                    total_compressed_params += comp_params
                    
                    print(f"   Ratio: {ratio}")
                    print(f"   Original: {orig_params:,} params")
                    print(f"   Compressed: {comp_params:,} params")
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   ✅ Successful: {success_count}/{len(batch_results)}")
        print(f"   📊 Total original params: {total_original_params:,}")
        print(f"   📊 Total compressed params: {total_compressed_params:,}")
        
        if total_original_params > 0:
            overall_ratio = total_original_params / total_compressed_params
            print(f"   📊 Overall compression: {overall_ratio:.1f}:1")

def main():
    """Main pipeline execution"""
    
    pipeline = CompressionPipeline()
    
    # Run batch processing for next 3 priority models
    results = pipeline.run_batch_processing(max_models=3)
    
    print(f"\n✅ Pipeline complete! Processed {len(results)} models.")

if __name__ == "__main__":
    main()