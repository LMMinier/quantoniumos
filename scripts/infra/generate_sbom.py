#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Generate Software Bill of Materials (SBOM) in CycloneDX format.

This script generates an SBOM for the QuantoniumOS project, listing all
dependencies and their versions for supply chain security compliance.

Usage:
    python scripts/generate_sbom.py
    
Output:
    sbom.json - CycloneDX 1.5 SBOM in JSON format
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
import uuid

from atomic_io import atomic_write_json


def get_installed_packages() -> List[Dict[str, str]]:
    """Get list of installed packages via pip."""
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list', '--format=json'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Warning: pip list failed: {result.stderr}")
        return []
    
    return json.loads(result.stdout)


def get_package_info(package_name: str) -> Dict[str, Any]:
    """Get detailed info about a package."""
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'show', package_name],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return {}
    
    info = {}
    for line in result.stdout.split('\n'):
        if ':' in line:
            key, _, value = line.partition(':')
            info[key.strip().lower()] = value.strip()
    
    return info


def generate_cyclonedx_sbom(packages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Generate CycloneDX 1.5 SBOM."""
    
    # Get project info
    repo_root = Path(__file__).parent.parent
    
    # Build components list
    components = []
    for pkg in packages:
        name = pkg.get('name', '')
        version = pkg.get('version', '')
        
        # Get additional info
        info = get_package_info(name)
        
        component = {
            "type": "library",
            "bom-ref": f"pkg:pypi/{name}@{version}",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name}@{version}",
        }
        
        if info.get('license'):
            component["licenses"] = [{"license": {"name": info['license']}}]
        
        if info.get('home-page'):
            component["externalReferences"] = [{
                "type": "website",
                "url": info['home-page']
            }]
        
        if info.get('author'):
            component["author"] = info['author']
        
        components.append(component)
    
    # Build SBOM
    sbom = {
        "$schema": "http://cyclonedx.org/schema/bom-1.5.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": {
                "components": [{
                    "type": "application",
                    "name": "quantoniumos-sbom-generator",
                    "version": "1.0.0"
                }]
            },
            "component": {
                "type": "application",
                "bom-ref": "pkg:github/mandcony/quantoniumos@2.0.0",
                "name": "quantoniumos",
                "version": "2.0.0",
                "description": "Quantum-Inspired Signal Processing Research Platform",
                "licenses": [
                    {"license": {"id": "AGPL-3.0-only"}},
                    {"license": {"name": "LicenseRef-QuantoniumOS-Claims-NC"}}
                ],
                "purl": "pkg:github/mandcony/quantoniumos@2.0.0",
                "externalReferences": [
                    {
                        "type": "vcs",
                        "url": "https://github.com/mandcony/quantoniumos"
                    },
                    {
                        "type": "website", 
                        "url": "https://github.com/mandcony/quantoniumos"
                    }
                ]
            }
        },
        "components": components
    }
    
    return sbom


def main():
    print("Generating SBOM for QuantoniumOS...")
    print()
    
    # Get packages
    packages = get_installed_packages()
    print(f"Found {len(packages)} installed packages")
    
    # Generate SBOM
    sbom = generate_cyclonedx_sbom(packages)
    
    # Write output
    output_path = Path(__file__).parent.parent / 'sbom.json'
    atomic_write_json(output_path, sbom, indent=2)
    
    print(f"Written to: {output_path}")
    print()
    
    # Summary
    print("SBOM Summary:")
    print(f"  Format: CycloneDX 1.5")
    print(f"  Components: {len(sbom['components'])}")
    print(f"  Serial: {sbom['serialNumber']}")
    print()
    print("Next steps:")
    print("  1. Run CVE scan: trivy fs --exit-code 1 .")
    print("  2. Enable Dependabot in GitHub settings")
    print("  3. Add OpenSSF Scorecard badge")


if __name__ == '__main__':
    main()
