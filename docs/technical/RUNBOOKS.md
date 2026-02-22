# QuantoniumOS Operations Runbooks

## Executive Summary

This document outlines **hypothetical operational procedures** that would be required if QuantoniumOS were ever promoted to production use. As of the latest update the platform remains a research prototype; the runbooks serve as planning notes for future hardening and must not be interpreted as evidence of current production readiness.

---

## Table of Contents

1. [System Startup & Shutdown](#system-startup--shutdown)
2. [Deployment Procedures](#deployment-procedures) 
3. [Validation & Testing](#validation--testing)
4. [Performance Monitoring](#performance-monitoring)
5. [Incident Response](#incident-response)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Backup & Recovery](#backup--recovery)
8. [Troubleshooting Guide](#troubleshooting-guide)

---

## System Startup & Shutdown

### 🚀 **Standard System Boot Procedure**

#### **Prerequisites Check**
```bash
#!/bin/bash
# Pre-boot validation script

echo "🔍 QuantoniumOS Pre-Boot Validation"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python version: $python_version (OK)"
else
    echo "❌ Python version: $python_version (Requires >= $required_version)"
    exit 1
fi

# Check required directories
required_dirs=(
    "/workspaces/quantoniumos/ASSEMBLY"
    "/workspaces/quantoniumos/core"
    "/workspaces/quantoniumos/apps"
    "/workspaces/quantoniumos/validation"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ Directory: $dir (OK)"
    else
        echo "❌ Directory: $dir (MISSING)"
        exit 1
    fi
done

# Check assembly libraries
assembly_libs=(
    "ASSEMBLY/compiled/librftkernel.so"
    "ASSEMBLY/build/librftkernel.so"
)

lib_found=false
for lib in "${assembly_libs[@]}"; do
    if [ -f "$lib" ]; then
        echo "✅ Assembly library: $lib (FOUND)"
        lib_found=true
        break
    fi
done

if [ "$lib_found" = false ]; then
    echo "⚠️  No assembly libraries found - will use Python fallback"
fi

echo "✅ Pre-boot validation completed"
```

#### **Standard Boot Sequence**
```bash
#!/bin/bash
# Standard QuantoniumOS boot procedure

cd /workspaces/quantoniumos

echo "🚀 QUANTONIUMOS UNIFIED BOOT"
echo "=========================="

# Option 1: Full GUI Boot (Recommended)
python3 quantonium_boot.py --mode desktop

# Option 2: Console Mode
# python3 quantonium_boot.py --mode console

# Option 3: Assembly Engines Only
# python3 quantonium_boot.py --engines-only

# Option 4: Validation Mode
# python3 quantonium_boot.py --validate

echo "✅ QuantoniumOS boot completed"
```

#### **Boot Validation Checklist**
```bash
# Verify all engines are operational
python3 -c "
import json
with open('smart_engine_validation_1757347591.json') as f:
    status = json.load(f)
    
engines = ['crypto_engine', 'quantum_engine', 'neural_engine', 'orchestrator_engine']
all_excellent = all(status[engine]['status'] == 'EXCELLENT' for engine in engines)

if all_excellent:
    print('✅ ALL ENGINES EXCELLENT - SYSTEM READY')
else:
    print('❌ ENGINE VALIDATION FAILED')
    exit(1)
"
```

### 🛑 **Graceful Shutdown Procedure**

```bash
#!/bin/bash
# Graceful QuantoniumOS shutdown

echo "🛑 QuantoniumOS Graceful Shutdown"
echo "==============================="

# 1. Stop running applications
echo "Stopping QuantoniumOS applications..."
pkill -f "symbolic_simulator.py"
pkill -f "launch_q_"
pkill -f "quantonium_desktop.py"

# 2. Save current state
echo "Saving system state..."
python3 -c "
from tools.config import save_system_state
save_system_state()
print('✅ System state saved')
"

# 3. Cleanup temporary files
echo "Cleaning temporary files..."
find /tmp -name "quantonium_*" -delete 2>/dev/null || true

# 4. Secure memory cleanup
echo "Performing secure cleanup..."
python3 -c "
import gc
gc.collect()
print('✅ Memory cleanup completed')
"

echo "✅ QuantoniumOS shutdown completed"
```

---

## Deployment Procedures

### 📦 **Production Deployment**

#### **Environment Setup**
```bash
#!/bin/bash
# Production environment setup

# 1. Create deployment directory
DEPLOY_DIR="/opt/quantoniumos"
sudo mkdir -p $DEPLOY_DIR
sudo chown $USER:$USER $DEPLOY_DIR

# 2. Clone repository
cd $DEPLOY_DIR
git clone https://github.com/mandcony/quantoniumos.git .

# 3. Install Python dependencies
pip3 install -e .

# 4. Compile assembly engines
cd ASSEMBLY
make clean all
cd ..

# 5. Set proper permissions
chmod 755 quantonium_boot.py
chmod 644 config/*.json
chmod 700 QVault/
chmod 600 QVault/.salt

# 6. Validate deployment
python3 quantonium_boot.py --validate

echo "✅ Production deployment completed"
```

#### **Docker Deployment**
```dockerfile
# Dockerfile for QuantoniumOS
FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    gcc \
    nasm \
    cmake \
    make \
    libqt5widgets5 \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /opt/quantoniumos
WORKDIR /opt/quantoniumos

# Install Python dependencies
RUN pip3 install -e .

# Compile assembly engines
RUN cd ASSEMBLY && make clean all

# Set permissions
RUN chmod 755 quantonium_boot.py

# Expose ports (if needed)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python3 -c "import sys; sys.exit(0 if True else 1)"

# Start command
CMD ["python3", "quantonium_boot.py", "--mode", "console"]
```

```bash
# Build and run Docker container
docker build -t quantoniumos:latest .
docker run -d --name quantoniumos quantoniumos:latest
```

### 🔄 **Update Procedures**

#### **Standard Update Process**
```bash
#!/bin/bash
# QuantoniumOS update procedure

echo "🔄 QuantoniumOS Update Process"
echo "============================"

# 1. Backup current installation
backup_dir="/opt/quantoniumos/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $backup_dir
cp -r . $backup_dir/
echo "✅ Backup created: $backup_dir"

# 2. Pull latest changes
git fetch origin
git checkout main
git pull origin main

# 3. Update dependencies
pip3 install -e . --upgrade

# 4. Recompile assembly if needed
if [ -f "ASSEMBLY/Makefile" ]; then
    cd ASSEMBLY
    make clean all
    cd ..
    echo "✅ Assembly engines recompiled"
fi

# 5. Run validation
python3 validation/tests/final_comprehensive_validation.py

# 6. Update configuration if needed
python3 tools/config.py --update

echo "✅ Update completed successfully"
```

---

## Validation & Testing

### 🧪 **Comprehensive Validation Suite**

#### **Quick Health Check**
```bash
#!/bin/bash
# Quick system health check (30 seconds)

echo "🏥 QuantoniumOS Quick Health Check"
echo "================================"

# Engine status check
python3 -c "
import json
import time

start_time = time.time()

# Load smart engine validation
try:
    with open('smart_engine_validation_1757347591.json') as f:
        status = json.load(f)
    
    for engine in ['crypto_engine', 'quantum_engine', 'neural_engine', 'orchestrator_engine']:
        print(f'{engine}: {status[engine][\"status\"]}')
    
    print(f'Overall: {status[\"summary\"][\"overall_status\"]}')
    print(f'Research build production flag (expected False): {status[\"summary\"].get(\"ready_for_production\", False)}')
    
except FileNotFoundError:
    print('❌ Engine validation file not found')
    exit(1)

print(f'Health check completed in {time.time() - start_time:.2f}s')
"
```

#### **Full Validation Suite**
```bash
#!/bin/bash
# Complete validation suite (5-10 minutes)

echo "🔬 QuantoniumOS Full Validation Suite"
echo "===================================="

validation_results=()

# 1. Crypto validation
echo "Running cryptographic validation..."
if python3 validation/tests/crypto_performance_test.py; then
    validation_results+=("✅ Cryptographic validation: PASS")
else
    validation_results+=("❌ Cryptographic validation: FAIL")
fi

# 2. Assembly engine validation
echo "Running assembly engine validation..."
if python3 -c "
from ASSEMBLY.python_bindings.unitary_rft import UnitaryRFT
rft = UnitaryRFT(64)
print('✅ Assembly engines: OPERATIONAL')
"; then
    validation_results+=("✅ Assembly engines: PASS")
else
    validation_results+=("❌ Assembly engines: FAIL")
fi

# 3. Performance benchmarks
echo "Running performance benchmarks..."
if python3 validation/benchmarks/QUANTONIUM_BENCHMARK_SUITE.py; then
    validation_results+=("✅ Performance benchmarks: PASS")
else
    validation_results+=("❌ Performance benchmarks: FAIL")
fi

# 4. Integration tests
echo "Running integration tests..."
if python3 validation/tests/final_comprehensive_validation.py; then
    validation_results+=("✅ Integration tests: PASS")
else
    validation_results+=("❌ Integration tests: FAIL")
fi

# Summary
echo ""
echo "📊 Validation Summary"
echo "==================="
for result in "${validation_results[@]}"; do
    echo "$result"
done

# Check if all passed
if [[ "${validation_results[*]}" =~ "❌" ]]; then
    echo "❌ VALIDATION FAILED - DO NOT DEPLOY"
    exit 1
else
    echo "✅ ALL VALIDATIONS PASSED - STILL RESEARCH-ONLY"
fi
```

### 📈 **Performance Testing**

#### **Throughput Testing**
```python
#!/usr/bin/env python3
# Performance throughput testing

import time
import statistics
from core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2

def test_crypto_throughput():
    """Test cryptographic throughput"""
    
    crypto = EnhancedRFTCryptoV2(b'test_key_32_bytes_exactly_here')
    test_data = b'A' * 1024  # 1KB test data
    iterations = 1000
    
    print("🚀 Crypto Throughput Test")
    print("========================")
    
    # Encryption throughput
    start_time = time.time()
    for _ in range(iterations):
        encrypted = crypto.encrypt_aead(test_data)
    encrypt_time = time.time() - start_time
    
    encrypt_throughput = (len(test_data) * iterations) / encrypt_time / 1024 / 1024
    print(f"Encryption: {encrypt_throughput:.2f} MB/s")
    
    # Decryption throughput
    start_time = time.time()
    for _ in range(iterations):
        decrypted = crypto.decrypt_aead(encrypted)
    decrypt_time = time.time() - start_time
    
    decrypt_throughput = (len(test_data) * iterations) / decrypt_time / 1024 / 1024
    print(f"Decryption: {decrypt_throughput:.2f} MB/s")
    
    # Assembly target check
    assembly_target = 9.2  # MB/s
    if encrypt_throughput >= assembly_target:
        print(f"✅ Encryption meets assembly target ({assembly_target} MB/s)")
    else:
        print(f"⚠️  Encryption below assembly target ({assembly_target} MB/s)")

if __name__ == "__main__":
    test_crypto_throughput()
```

---

## Performance Monitoring

### 📊 **Real-time Monitoring**

#### **System Metrics Collection**
```python
#!/usr/bin/env python3
# Real-time system monitoring

import psutil
import time
import json
from datetime import datetime

def collect_system_metrics():
    """Collect comprehensive system metrics"""
    
    metrics = {
        'timestamp': datetime.utcnow().isoformat(),
        'cpu': {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
        },
        'memory': {
            'percent': psutil.virtual_memory().percent,
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'used_gb': psutil.virtual_memory().used / (1024**3)
        },
        'disk': {
            'percent': psutil.disk_usage('/').percent,
            'free_gb': psutil.disk_usage('/').free / (1024**3)
        },
        'processes': len([p for p in psutil.process_iter() if 'quantonium' in p.name().lower()])
    }
    
    return metrics

def monitor_quantonium_performance():
    """Monitor QuantoniumOS specific performance"""
    
    print("📊 QuantoniumOS Performance Monitor")
    print("==================================")
    
    while True:
        try:
            metrics = collect_system_metrics()
            
            # Display current metrics
            print(f"\r⏰ {metrics['timestamp']} | "
                  f"CPU: {metrics['cpu']['percent']:5.1f}% | "
                  f"Memory: {metrics['memory']['percent']:5.1f}% | "
                  f"Disk: {metrics['disk']['percent']:5.1f}% | "
                  f"Processes: {metrics['processes']}", end='')
            
            # Alert on high resource usage
            if metrics['cpu']['percent'] > 80:
                print("\n⚠️  HIGH CPU USAGE DETECTED")
            if metrics['memory']['percent'] > 85:
                print("\n⚠️  HIGH MEMORY USAGE DETECTED")
            if metrics['disk']['percent'] > 90:
                print("\n⚠️  LOW DISK SPACE WARNING")
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break

if __name__ == "__main__":
    monitor_quantonium_performance()
```

#### **Application Health Dashboard**
```python
#!/usr/bin/env python3
# Application health dashboard

def create_health_dashboard():
    """Create real-time health dashboard"""
    
    dashboard_data = {
        'system_status': 'OPERATIONAL',
        'engines': {
            'crypto_engine': check_crypto_engine(),
            'quantum_engine': check_quantum_engine(),
            'neural_engine': check_neural_engine(),
            'orchestrator_engine': check_orchestrator_engine()
        },
        'applications': {
            'symbolic_simulator': check_app_status('symbolic_simulator'),
            'q_notes': check_app_status('q_notes'),
            'q_vault': check_app_status('q_vault')
        },
        'performance': {
            'crypto_throughput': measure_crypto_throughput(),
            'quantum_processing': measure_quantum_speed(),
            'memory_usage': get_memory_usage()
        }
    }
    
    return dashboard_data

def check_crypto_engine():
    """Check crypto engine health"""
    try:
        from core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2
        crypto = EnhancedRFTCryptoV2(b'test_key_32_bytes_exactly_here')
        test_data = b'health_check_data'
        encrypted = crypto.encrypt_aead(test_data)
        decrypted = crypto.decrypt_aead(encrypted)
        return 'HEALTHY' if decrypted == test_data else 'DEGRADED'
    except Exception:
        return 'FAILED'
```

---

## Incident Response

### 🚨 **Incident Response Procedures**

#### **System Crash Recovery**
```bash
#!/bin/bash
# System crash recovery procedure

echo "🚨 QuantoniumOS Incident Response - System Crash"
echo "=============================================="

# 1. Assess system state
echo "Assessing system state..."
if pgrep -f quantonium > /dev/null; then
    echo "⚠️  Some QuantoniumOS processes still running"
    echo "Attempting graceful shutdown..."
    pkill -TERM -f quantonium
    sleep 10
    pkill -KILL -f quantonium
fi

# 2. Check for core dumps
if ls core.* >/dev/null 2>&1; then
    echo "⚠️  Core dumps found - preserving for analysis"
    mkdir -p /opt/quantoniumos/crash_reports/$(date +%Y%m%d_%H%M%S)
    mv core.* /opt/quantoniumos/crash_reports/$(date +%Y%m%d_%H%M%S)/
fi

# 3. Verify file system integrity
echo "Checking file system integrity..."
if [ -f "PROJECT_STATUS.json" ] && [ -d "ASSEMBLY" ] && [ -d "core" ]; then
    echo "✅ Core files intact"
else
    echo "❌ Core files missing - attempting recovery from backup"
    # Restore from backup
    restore_from_backup
fi

# 4. Restart system
echo "Attempting system restart..."
python3 quantonium_boot.py --validate

if [ $? -eq 0 ]; then
    echo "✅ System recovery successful"
else
    echo "❌ System recovery failed - escalating to manual intervention"
    exit 1
fi
```

#### **Performance Degradation Response**
```python
#!/usr/bin/env python3
# Performance degradation incident response

import time
import psutil
from core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2

def diagnose_performance_issue():
    """Diagnose performance degradation"""
    
    print("🔍 Performance Degradation Diagnosis")
    print("===================================")
    
    # 1. Check system resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    print(f"CPU Usage: {cpu_percent:.1f}%")
    print(f"Memory Usage: {memory_percent:.1f}%")
    
    # 2. Test crypto performance
    crypto = EnhancedRFTCryptoV2(b'test_key_32_bytes_exactly_here')
    test_data = b'A' * 1024
    
    start_time = time.time()
    for _ in range(100):
        encrypted = crypto.encrypt_aead(test_data)
    crypto_time = time.time() - start_time
    
    crypto_throughput = (1024 * 100) / crypto_time / 1024 / 1024
    print(f"Crypto Throughput: {crypto_throughput:.2f} MB/s")
    
    # 3. Check assembly libraries
    try:
        from ASSEMBLY.python_bindings.unitary_rft import UnitaryRFT
        rft = UnitaryRFT(64)
        print("✅ Assembly libraries loaded")
    except Exception as e:
        print(f"❌ Assembly library issue: {e}")
    
    # 4. Generate recommendations
    recommendations = []
    
    if cpu_percent > 80:
        recommendations.append("HIGH CPU - Check for runaway processes")
    if memory_percent > 85:
        recommendations.append("HIGH MEMORY - Possible memory leak")
    if crypto_throughput < 1.0:
        recommendations.append("LOW CRYPTO PERFORMANCE - Check assembly libraries")
    
    if recommendations:
        print("\n🔧 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    else:
        print("\n✅ No obvious performance issues detected")

if __name__ == "__main__":
    diagnose_performance_issue()
```

#### **Security Incident Response**
```bash
#!/bin/bash
# Security incident response

echo "🔒 QuantoniumOS Security Incident Response"
echo "========================================="

# 1. Immediate containment
echo "Step 1: Immediate containment..."
# Stop all network-facing services
systemctl stop nginx 2>/dev/null || true
systemctl stop apache2 2>/dev/null || true

# 2. Evidence preservation
echo "Step 2: Preserving evidence..."
incident_dir="/opt/quantoniumos/security_incidents/$(date +%Y%m%d_%H%M%S)"
mkdir -p $incident_dir

# Copy logs
cp -r /var/log/* $incident_dir/ 2>/dev/null || true

# Capture system state
ps aux > $incident_dir/processes.txt
netstat -tulpn > $incident_dir/network.txt
lsof > $incident_dir/open_files.txt

# 3. Check for unauthorized changes
echo "Step 3: Checking for unauthorized changes..."
python3 -c "
import hashlib
import json

# Check critical file integrity
critical_files = [
    'core/enhanced_rft_crypto_v2.py',
    'algorithms/rft/kernels/engines/crypto/src/feistel_round48.c',
    'config/app_registry.json'
]

for file_path in critical_files:
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        print(f'{file_path}: {file_hash}')
    except FileNotFoundError:
        print(f'❌ CRITICAL FILE MISSING: {file_path}')
"

# 4. Security validation
echo "Step 4: Running security validation..."
python3 quantonium_security_analysis.py

echo "✅ Security incident response completed"
echo "📊 Evidence preserved in: $incident_dir"
```

---

## Maintenance Procedures

### 🔧 **Regular Maintenance**

#### **Daily Maintenance**
```bash
#!/bin/bash
# Daily maintenance script

echo "🔧 QuantoniumOS Daily Maintenance"
echo "==============================="

# 1. Health check
python3 validation/tests/quick_assembly_test.py

# 2. Log rotation
find /opt/quantoniumos/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true

# 3. Temporary file cleanup
find /tmp -name "quantonium_*" -mtime +1 -delete 2>/dev/null || true

# 4. Update performance metrics
python3 -c "
from datetime import datetime
import json

metrics = {
    'date': datetime.utcnow().isoformat(),
    'maintenance_type': 'daily',
    'status': 'completed'
}

with open('maintenance_log.json', 'a') as f:
    f.write(json.dumps(metrics) + '\n')
"

echo "✅ Daily maintenance completed"
```

#### **Weekly Maintenance**
```bash
#!/bin/bash
# Weekly maintenance script

echo "🔧 QuantoniumOS Weekly Maintenance"
echo "================================"

# 1. Full validation suite
python3 validation/tests/final_comprehensive_validation.py

# 2. Performance benchmarking
python3 validation/benchmarks/QUANTONIUM_BENCHMARK_SUITE.py

# 3. Security scan
python3 quantonium_security_analysis.py

# 4. Backup verification
python3 -c "
import os
from pathlib import Path

    for engine in ['crypto_engine', 'quantum_engine', 'neural_engine', 'orchestrator_engine']:
        print(f'{engine}: {status[engine]["status"]}')

    print(f'Overall: {status["summary"]["overall_status"]}')
    print(f'Research build production flag (expected False): {status["summary"].get("ready_for_production", False)}')
    if len(backups) > 10:
        old_backups = sorted(backups)[:-10]
        for backup in old_backups:
            os.system(f'rm -rf {backup}')
        print(f'🗑️  Cleaned {len(old_backups)} old backups')
else:
    print('⚠️  No backups directory found')
"

echo "✅ Weekly maintenance completed"
```

#### **Monthly Maintenance**
```bash
#!/bin/bash
# Monthly maintenance script

echo "🔧 QuantoniumOS Monthly Maintenance"
echo "=================================="

# 1. Security audit
python3 quantonium_complete_ai_safety_analysis.py

# 2. Performance analysis
python3 -c "
import json
from pathlib import Path

# Analyze performance trends
metrics_files = Path('validation/results').glob('*.json')
performance_data = []

for file in metrics_files:
    with open(file) as f:
        data = json.load(f)
        if 'performance' in data:
            performance_data.append(data['performance'])

print(f'📊 Analyzed {len(performance_data)} performance reports')
"

# 3. Update dependencies
pip3 list --outdated

# 4. System optimization
python3 -c "
import gc
import os

# Memory optimization
gc.collect()

# Rebuild assembly if needed
if os.path.exists('ASSEMBLY/Makefile'):
    os.system('cd ASSEMBLY && make clean && make all')
    print('✅ Assembly engines rebuilt')
"

echo "✅ Monthly maintenance completed"
```

---

## Backup & Recovery

### 💾 **Backup Procedures**

#### **Configuration Backup**
```bash
#!/bin/bash
# Configuration backup script

backup_date=$(date +%Y%m%d_%H%M%S)
backup_dir="/opt/quantoniumos/backups/config_$backup_date"

echo "💾 QuantoniumOS Configuration Backup"
echo "==================================="

mkdir -p $backup_dir

# Backup configuration files
cp -r config/ $backup_dir/
cp PROJECT_STATUS.json $backup_dir/
cp PROJECT_SUMMARY.json $backup_dir/

# Backup QVault (encrypted)
cp -r QVault/ $backup_dir/

# Backup validation results
cp -r validation/results/ $backup_dir/ 2>/dev/null || true

# Create manifest
echo "Backup created: $backup_date" > $backup_dir/MANIFEST
echo "Contents:" >> $backup_dir/MANIFEST
find $backup_dir -type f >> $backup_dir/MANIFEST

# Compress backup
tar -czf "${backup_dir}.tar.gz" -C $(dirname $backup_dir) $(basename $backup_dir)
rm -rf $backup_dir

echo "✅ Configuration backup completed: ${backup_dir}.tar.gz"
```

#### **Full System Backup**
```bash
#!/bin/bash
# Full system backup script

backup_date=$(date +%Y%m%d_%H%M%S)
backup_dir="/opt/quantoniumos/backups/full_$backup_date"

echo "💾 QuantoniumOS Full System Backup"
echo "================================="

mkdir -p $backup_dir

# Backup entire system (excluding build artifacts)
rsync -av --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='ASSEMBLY/build' \
          --exclude='.git' \
          /opt/quantoniumos/ $backup_dir/

# Create checksum manifest
find $backup_dir -type f -exec sha256sum {} \; > $backup_dir/CHECKSUMS

# Compress backup
tar -czf "${backup_dir}.tar.gz" -C $(dirname $backup_dir) $(basename $backup_dir)
rm -rf $backup_dir

echo "✅ Full system backup completed: ${backup_dir}.tar.gz"
```

### 🔄 **Recovery Procedures**

#### **Configuration Recovery**
```bash
#!/bin/bash
# Configuration recovery script

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

backup_file="$1"
recovery_dir="/opt/quantoniumos/recovery_$(date +%Y%m%d_%H%M%S)"

echo "🔄 QuantoniumOS Configuration Recovery"
echo "===================================="

# Extract backup
mkdir -p $recovery_dir
tar -xzf $backup_file -C $recovery_dir

# Validate backup integrity
if [ -f "$recovery_dir/*/MANIFEST" ]; then
    echo "✅ Backup manifest found"
else
    echo "❌ Invalid backup file"
    exit 1
fi

# Restore configuration
cp -r $recovery_dir/*/config/ /opt/quantoniumos/
cp $recovery_dir/*/PROJECT_STATUS.json /opt/quantoniumos/
cp $recovery_dir/*/PROJECT_SUMMARY.json /opt/quantoniumos/

# Restore QVault
cp -r $recovery_dir/*/QVault/ /opt/quantoniumos/

# Validate recovery
python3 quantonium_boot.py --validate

if [ $? -eq 0 ]; then
    echo "✅ Configuration recovery completed successfully"
    rm -rf $recovery_dir
else
    echo "❌ Configuration recovery failed"
    echo "Recovery files preserved in: $recovery_dir"
    exit 1
fi
```

---

## Troubleshooting Guide

### 🔧 **Common Issues & Solutions**

#### **Assembly Library Loading Issues**

**Problem:** `Failed to load library: librftkernel.so`

**Diagnosis:**
```bash
# Check library paths
ls -la ASSEMBLY/compiled/
ls -la ASSEMBLY/build/

# Check library dependencies
ldd ASSEMBLY/compiled/librftkernel.so 2>/dev/null || echo "Library not found"

# Check compilation
cd ASSEMBLY
make clean all
```

**Solutions:**
1. **Recompile libraries:** `cd ASSEMBLY && make clean all`
2. **Check dependencies:** Install gcc, nasm, cmake
3. **Use Python fallback:** System continues with reduced performance

#### **PyQt5 Import Errors**

**Problem:** `ImportError: No module named 'PyQt5'`

**Solutions:**
```bash
# Install PyQt5
pip3 install PyQt5

# Alternative: Use console mode
python3 quantonium_boot.py --mode console

# Check X11 forwarding (if using SSH)
ssh -X user@host
```

#### **Permission Denied Errors**

**Problem:** Permission errors accessing QVault or config files

**Solutions:**
```bash
# Fix QVault permissions
chmod 700 QVault/
chmod 600 QVault/.salt

# Fix config permissions
chmod 644 config/*.json

# Check ownership
chown -R $USER:$USER /opt/quantoniumos/
```

#### **Performance Issues**

**Problem:** Slow encryption/decryption performance

**Diagnosis:**
```python
# Test crypto performance
python3 validation/tests/crypto_performance_test.py

# Check assembly library loading
python3 -c "
from ASSEMBLY.python_bindings.unitary_rft import UnitaryRFT
print('Assembly libraries working')
"
```

**Solutions:**
1. **Ensure assembly libraries compiled:** `cd ASSEMBLY && make all`
2. **Check CPU architecture support:** Verify AVX2 support
3. **Monitor system resources:** Use htop/top to check CPU/memory

#### **Validation Failures**

**Problem:** Validation tests failing

**Quick Fixes:**
```bash
# Reset to known good state
git checkout main
git pull origin main

# Recompile everything
cd ASSEMBLY
make clean all
cd ..

# Run validation
python3 validation/tests/final_comprehensive_validation.py
```

### 📞 **Escalation Procedures**

#### **Level 1: Self-Service**
- Check this runbook
- Run diagnostic scripts
- Review system logs

#### **Level 2: Technical Support**
- Collect system information
- Generate support bundle
- Contact development team

#### **Level 3: Critical Incident**
- Immediate containment
- Evidence preservation
- Emergency recovery procedures

### 📋 **Support Information Collection**

```bash
#!/bin/bash
# Support information collection script

support_dir="/opt/quantoniumos/support_$(date +%Y%m%d_%H%M%S)"
mkdir -p $support_dir

echo "📋 Collecting QuantoniumOS Support Information"
echo "=============================================="

# System information
uname -a > $support_dir/system_info.txt
python3 --version > $support_dir/python_version.txt
pip3 list > $support_dir/pip_packages.txt

# QuantoniumOS status
python3 -c "
import json
try:
    with open('smart_engine_validation_1757347591.json') as f:
        status = json.load(f)
    with open('$support_dir/engine_status.json', 'w') as f:
        json.dump(status, f, indent=2)
except:
    pass
"

# Configuration
cp -r config/ $support_dir/config_copy/
cp PROJECT_STATUS.json $support_dir/

# Logs (if available)
cp -r logs/ $support_dir/logs_copy/ 2>/dev/null || true

# Create archive
tar -czf "${support_dir}.tar.gz" -C $(dirname $support_dir) $(basename $support_dir)
rm -rf $support_dir

echo "✅ Support bundle created: ${support_dir}.tar.gz"
echo "📧 Please send this file to technical support"
```

---

## 📋 Operations Checklist

### **Daily Operations**
- [ ] Check system health dashboard
- [ ] Review performance metrics
- [ ] Verify backup completion
- [ ] Check for any alerts or warnings

### **Weekly Operations**
- [ ] Run full validation suite
- [ ] Review security logs
- [ ] Update performance baselines
- [ ] Test disaster recovery procedures

### **Monthly Operations**  
- [ ] Security audit and review
- [ ] Performance trend analysis
- [ ] Dependency updates review
- [ ] Documentation updates

### **Quarterly Operations**
- [ ] Comprehensive security assessment
- [ ] Architecture review
- [ ] Capacity planning
- [ ] Training and knowledge transfer

---

## 🎯 Conclusion

These runbooks capture forward-looking guidance that would be needed to operate QuantoniumOS if it ever graduates from research to production. Until independent audits and hardening are complete they should be treated as exploratory planning notes.

🚧 **Current status:** Research prototype without production support guarantees 
✅ **Use now for:** Reproducible experiments and design documentation 
🔬 **Next steps before deployment:** Third-party security review, operational testing, and formal approval

**Reminder:** Never apply these procedures to live systems until the outstanding validation work is complete.
