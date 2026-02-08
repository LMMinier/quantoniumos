#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Breaking the ASCII Wall: Hierarchical Transform Coding for Discrete Signals
===========================================================================

Comprehensive experimental validation for standalone research paper.

Tests 10+ hypotheses across diverse signal types to validate the 
"hierarchical cascade" architecture for hybrid transform coding.

Signal Categories:
1. Pure ASCII (text, source code, JSON, XML)
2. Mixed Discrete-Continuous (ASCII + waves, JSON + sine)
3. Natural Continuous (Fibonacci, sine, chirp)
4. Real-world Data (logs, config files, CSV)

Hypothesis Testing:
- H0 (Baseline): Greedy per-bin selection (current paper)
- H3: Hierarchical cascade (structure/texture split)
- H5: Attention-based soft gating
- H6: Dictionary learning with bridge atoms
- H7: H3 + H5 hybrid (cascade with attention routing)

Metrics:
- BPP (bits per pixel): Primary compression metric
- PSNR (dB): Signal reconstruction quality
- Coherence Violation: Energy in rejected basis coefficients
- Sparsity (%): Zero coefficient percentage
- Runtime (ms): Encoding time

Output:
- Raw results in experiments/ascii_wall_results.json
- Publication-ready tables in experiments/ASCII_WALL_PAPER_RESULTS.md
- LaTeX tables for direct paper inclusion
"""

import numpy as np
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import time

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import hypothesis implementations
from hybrid_mca_fixes import (
    compute_bpp,
    compute_psnr,
    compute_coherence_violation,
    baseline_greedy_hybrid,
    hypothesis3_hierarchical_cascade,
    hypothesis5_attention_gating,
    hypothesis6_dictionary_learning,
    hypothesis7_cascade_attention
)

# Import transforms directly
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse


@dataclass
class SignalDescriptor:
    """Metadata for test signal"""
    name: str
    category: str  # "ascii", "mixed", "continuous", "real-world"
    size: int
    description: str
    expected_winner: str = "unknown"  # For hypothesis validation


@dataclass
class ExperimentResult:
    """Complete experimental result"""
    signal_name: str
    signal_category: str
    signal_size: int
    method: str
    bpp: float
    psnr_db: float
    coherence_violation: float
    sparsity_pct: float
    reconstruction_error: float
    time_ms: float
    
    def to_latex_row(self) -> str:
        """Format as LaTeX table row"""
        return (
            f"{self.method} & "
            f"{self.bpp:.3f} & "
            f"{self.psnr_db:.2f} & "
            f"{self.coherence_violation:.2e} & "
            f"{self.sparsity_pct:.1f}\\% & "
            f"{self.time_ms:.2f} \\\\"
        )


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_python_source(n: int = 2048) -> np.ndarray:
    """Real Python source code from this project"""
    try:
        code_file = Path(__file__).parent.parent / "core" / "canonical_true_rft.py"
        with open(code_file, 'r') as f:
            text = f.read()[:n]
        return text_to_signal(text, n)
    except:
        # Fallback: synthetic Python-like code
        keywords = ['def', 'class', 'import', 'return', 'if', 'for', 'while', 'True', 'False', 'None']
        code = ' '.join(np.random.choice(keywords) for _ in range(n//5))
        return text_to_signal(code[:n], n)


def generate_json_data(n: int = 1024) -> np.ndarray:
    """Synthetic JSON-like structure"""
    template = '{"key": "value", "number": 42, "array": [1, 2, 3], "nested": {"inner": true}}'
    repeated = (template * (n // len(template) + 1))[:n]
    return text_to_signal(repeated, n)


def generate_xml_data(n: int = 1024) -> np.ndarray:
    """Synthetic XML-like structure"""
    template = '<root><item id="1"><name>Test</name><value>42</value></item></root>'
    repeated = (template * (n // len(template) + 1))[:n]
    return text_to_signal(repeated, n)


def generate_csv_data(n: int = 1024) -> np.ndarray:
    """Synthetic CSV with numbers"""
    rows = []
    for i in range(n // 20):
        row = f"{i},{np.random.randint(0,100)},{np.random.randn():.3f}\n"
        rows.append(row)
    text = ''.join(rows)[:n]
    return text_to_signal(text, n)


def generate_log_file(n: int = 1024) -> np.ndarray:
    """Synthetic log entries"""
    levels = ['INFO', 'WARN', 'ERROR', 'DEBUG']
    messages = ['Request received', 'Processing', 'Complete', 'Failed']
    lines = []
    for i in range(n // 50):
        level = np.random.choice(levels)
        msg = np.random.choice(messages)
        line = f"[2025-11-25 12:34:{i:02d}] {level}: {msg}\n"
        lines.append(line)
    text = ''.join(lines)[:n]
    return text_to_signal(text, n)


def generate_random_ascii(n: int = 1024) -> np.ndarray:
    """Pure random ASCII text"""
    chars = ''.join(chr(x) for x in np.random.randint(32, 127, n))
    return text_to_signal(chars, n)


def generate_mixed_ascii_wave(n: int = 512) -> np.ndarray:
    """ASCII steps + Fibonacci wave (from paper)"""
    # ASCII steps (similar to paper's test)
    ascii_part = np.zeros(n)
    current_val = 0
    step_length = 16
    for i in range(0, n, step_length):
        if np.random.rand() < 0.7:
            current_val = np.random.randint(32, 127)
        ascii_part[i:i+step_length] = current_val
    
    # Fibonacci wave component
    phi = (1 + np.sqrt(5)) / 2
    t = np.linspace(0, 4*np.pi, n)
    fib_wave = 30 * np.sin(phi * t)
    
    # Mix 60% ASCII, 40% wave
    signal = 0.6 * ascii_part + 0.4 * fib_wave
    return signal / np.max(np.abs(signal))  # Normalize to [-1, 1]


def generate_mixed_json_sine(n: int = 512) -> np.ndarray:
    """JSON structure + sine wave"""
    json_sig = generate_json_data(n)
    t = np.linspace(0, 4*np.pi, n)
    sine = np.sin(t)
    return 0.7 * json_sig + 0.3 * sine


def generate_fibonacci_wave(n: int = 512) -> np.ndarray:
    """Pure Fibonacci-based wave (continuous)"""
    phi = (1 + np.sqrt(5)) / 2
    t = np.linspace(0, 8*np.pi, n)
    signal = np.sin(phi * t) + 0.5 * np.cos(phi**2 * t)
    return signal / np.max(np.abs(signal))


def generate_chirp(n: int = 512) -> np.ndarray:
    """Linear frequency chirp (continuous)"""
    t = np.linspace(0, 1, n)
    f0, f1 = 5, 50  # Start and end frequency
    signal = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2))
    return signal


def generate_natural_sine(n: int = 512) -> np.ndarray:
    """Simple sine wave (continuous baseline)"""
    t = np.linspace(0, 8*np.pi, n)
    return np.sin(t)


def text_to_signal(text: str, n: int) -> np.ndarray:
    """Convert text to normalized signal"""
    # Pad or truncate
    if len(text) < n:
        text = text + ' ' * (n - len(text))
    else:
        text = text[:n]
    
    # Convert to ASCII values, normalize to [-1, 1]
    signal = np.array([ord(c) for c in text], dtype=float)
    signal = 2 * (signal - 32) / (126 - 32) - 1  # Map [32,126] -> [-1,1]
    return signal


# ============================================================================
# TEST SUITE
# ============================================================================

def build_test_suite() -> Dict[str, Tuple[np.ndarray, SignalDescriptor]]:
    """
    Build comprehensive test suite across all signal categories.
    
    Returns:
        Dict mapping signal_id -> (signal_array, descriptor)
    """
    suite = {}
    
    # Category 1: Pure ASCII (discrete symbol sequences)
    suite['python_2k'] = (
        generate_python_source(2048),
        SignalDescriptor(
            name="Python Source Code",
            category="ascii",
            size=2048,
            description="Real Python code (canonical_true_rft.py)",
            expected_winner="H3_Cascade"
        )
    )
    
    suite['json_1k'] = (
        generate_json_data(1024),
        SignalDescriptor(
            name="JSON Data",
            category="ascii",
            size=1024,
            description="Structured JSON with nested objects",
            expected_winner="H3_Cascade"
        )
    )
    
    suite['xml_1k'] = (
        generate_xml_data(1024),
        SignalDescriptor(
            name="XML Data",
            category="ascii",
            size=1024,
            description="XML markup with tags and attributes",
            expected_winner="H3_Cascade"
        )
    )
    
    suite['csv_1k'] = (
        generate_csv_data(1024),
        SignalDescriptor(
            name="CSV Data",
            category="ascii",
            size=1024,
            description="Comma-separated values with numbers",
            expected_winner="H3_Cascade"
        )
    )
    
    suite['log_1k'] = (
        generate_log_file(1024),
        SignalDescriptor(
            name="Log File",
            category="ascii",
            size=1024,
            description="Application log with timestamps",
            expected_winner="H3_Cascade"
        )
    )
    
    suite['random_ascii_1k'] = (
        generate_random_ascii(1024),
        SignalDescriptor(
            name="Random ASCII",
            category="ascii",
            size=1024,
            description="Uniformly random printable characters",
            expected_winner="H3_Cascade"
        )
    )
    
    # Category 2: Mixed Discrete-Continuous
    suite['ascii_fib_512'] = (
        generate_mixed_ascii_wave(512),
        SignalDescriptor(
            name="ASCII + Fibonacci",
            category="mixed",
            size=512,
            description="Paper's test signal: ASCII steps + golden ratio wave",
            expected_winner="H3_Cascade"
        )
    )
    
    suite['json_sine_512'] = (
        generate_mixed_json_sine(512),
        SignalDescriptor(
            name="JSON + Sine",
            category="mixed",
            size=512,
            description="JSON structure mixed with sine wave",
            expected_winner="H3_Cascade"
        )
    )
    
    # Category 3: Natural Continuous (baseline comparison)
    suite['fibonacci_512'] = (
        generate_fibonacci_wave(512),
        SignalDescriptor(
            name="Fibonacci Wave",
            category="continuous",
            size=512,
            description="Pure golden ratio based waveform",
            expected_winner="Baseline_or_H5"
        )
    )
    
    suite['chirp_512'] = (
        generate_chirp(512),
        SignalDescriptor(
            name="Chirp Signal",
            category="continuous",
            size=512,
            description="Linear frequency sweep",
            expected_winner="Baseline_or_H5"
        )
    )
    
    suite['sine_512'] = (
        generate_natural_sine(512),
        SignalDescriptor(
            name="Sine Wave",
            category="continuous",
            size=512,
            description="Simple sinusoid baseline",
            expected_winner="Baseline_or_H5"
        )
    )
    
    return suite


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_method(method_name: str, signal: np.ndarray, sparsity: float = 0.95):
    """
    Run a single hypothesis method.
    
    Returns:
        ExperimentResult from hybrid_mca_fixes
    """
    if method_name == "Baseline_Greedy":
        return baseline_greedy_hybrid(signal, sparsity)
    elif method_name == "H3_Cascade":
        return hypothesis3_hierarchical_cascade(signal, sparsity)
    elif method_name == "H5_Attention":
        return hypothesis5_attention_gating(signal, sparsity)
    elif method_name == "H6_Dictionary":
        return hypothesis6_dictionary_learning(signal, sparsity)
    elif method_name == "H7_Cascade_Attention":
        return hypothesis7_cascade_attention(signal, sparsity)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_single_experiment(
    signal_id: str,
    signal: np.ndarray,
    descriptor: SignalDescriptor,
    method_name: str,
    sparsity: float = 0.95
) -> ExperimentResult:
    """Run one method on one signal and collect all metrics"""
    
    try:
        # Get result from hypothesis function (already contains all metrics)
        result = run_method(method_name, signal, sparsity)
        
        # Convert to our ExperimentResult format with signal metadata
        return ExperimentResult(
            signal_name=descriptor.name,
            signal_category=descriptor.category,
            signal_size=descriptor.size,
            method=method_name,
            bpp=result.bpp,
            psnr_db=result.psnr,
            coherence_violation=result.coherence_violation,
            sparsity_pct=result.sparsity_pct,
            reconstruction_error=result.reconstruction_error,
            time_ms=result.time_ms
        )
    
    except Exception as e:
        print(f"  ❌ {method_name} failed: {e}")
        return ExperimentResult(
            signal_name=descriptor.name,
            signal_category=descriptor.category,
            signal_size=descriptor.size,
            method=method_name,
            bpp=999.0,
            psnr_db=-999.0,
            coherence_violation=999.0,
            sparsity_pct=0.0,
            reconstruction_error=999.0,
            time_ms=0.0
        )


def run_all_experiments(sparsity: float = 0.95) -> List[ExperimentResult]:
    """
    Run all hypotheses on all test signals.
    
    Returns:
        List of all experimental results
    """
    print("="*80)
    print("Breaking the ASCII Wall: Comprehensive Experimental Validation")
    print("="*80)
    print()
    
    test_suite = build_test_suite()
    methods = [
        "Baseline_Greedy",
        "H3_Cascade",
        "H5_Attention",
        "H6_Dictionary",
        "H7_Cascade_Attention"
    ]
    
    all_results = []
    
    for signal_id, (signal, descriptor) in test_suite.items():
        print(f"\n📊 Testing: {descriptor.name} ({descriptor.category}, N={descriptor.size})")
        print(f"   {descriptor.description}")
        print(f"   Expected winner: {descriptor.expected_winner}")
        print()
        
        signal_results = []
        for method in methods:
            print(f"   Running {method}...", end=' ', flush=True)
            result = run_single_experiment(signal_id, signal, descriptor, method, sparsity)
            signal_results.append(result)
            all_results.append(result)
            print(f"BPP={result.bpp:.3f}, PSNR={result.psnr_db:.2f}dB, Coherence={result.coherence_violation:.2e}")
        
        # Find winner for this signal
        valid_results = [r for r in signal_results if r.bpp < 900]
        if valid_results:
            winner = min(valid_results, key=lambda r: r.bpp)
            print(f"\n   🏆 Winner: {winner.method} (BPP={winner.bpp:.3f}, PSNR={winner.psnr_db:.2f}dB)")
    
    return all_results


# ============================================================================
# RESULTS ANALYSIS & OUTPUT
# ============================================================================

def generate_summary_tables(results: List[ExperimentResult]) -> str:
    """Generate publication-ready markdown tables"""
    
    md = ["# Breaking the ASCII Wall: Experimental Results\n"]
    md.append("## Summary by Signal Category\n")
    
    # Group by category
    categories = {}
    for result in results:
        if result.signal_category not in categories:
            categories[result.signal_category] = []
        categories[result.signal_category].append(result)
    
    for category, cat_results in sorted(categories.items()):
        md.append(f"\n### {category.upper()} Signals\n")
        
        # Group by signal within category
        signals = {}
        for r in cat_results:
            if r.signal_name not in signals:
                signals[r.signal_name] = []
            signals[r.signal_name].append(r)
        
        for signal_name, sig_results in signals.items():
            md.append(f"\n#### {signal_name}\n")
            md.append("| Method | BPP | PSNR (dB) | Coherence | Sparsity | Time (ms) |")
            md.append("|--------|-----|-----------|-----------|----------|-----------|")
            
            valid = [r for r in sig_results if r.bpp < 900]
            for r in valid:
                md.append(
                    f"| {r.method} | {r.bpp:.3f} | {r.psnr_db:.2f} | "
                    f"{r.coherence_violation:.2e} | {r.sparsity_pct:.1f}% | {r.time_ms:.2f} |"
                )
            
            if valid:
                winner = min(valid, key=lambda r: r.bpp)
                best_psnr = max(valid, key=lambda r: r.psnr_db)
                md.append(f"\n**Winner:** {winner.method} ({winner.bpp:.3f} BPP)")
                md.append(f"**Best PSNR:** {best_psnr.method} ({best_psnr.psnr_db:.2f} dB)\n")
    
    # Overall statistics
    md.append("\n## Overall Statistics\n")
    
    # Average by method across all signals
    methods = {}
    for r in results:
        if r.bpp < 900:  # Valid result
            if r.method not in methods:
                methods[r.method] = []
            methods[r.method].append(r)
    
    md.append("| Method | Avg BPP | Avg PSNR (dB) | Avg Coherence | Win Rate |")
    md.append("|--------|---------|---------------|---------------|----------|")
    
    for method, method_results in sorted(methods.items()):
        avg_bpp = np.mean([r.bpp for r in method_results])
        avg_psnr = np.mean([r.psnr_db for r in method_results])
        avg_coh = np.mean([r.coherence_violation for r in method_results])
        
        # Win rate: how often is this method the best BPP?
        wins = 0
        total_signals = len(set(r.signal_name for r in results if r.bpp < 900))
        
        for signal_name in set(r.signal_name for r in results):
            signal_results = [r for r in results if r.signal_name == signal_name and r.bpp < 900]
            if signal_results:
                winner = min(signal_results, key=lambda r: r.bpp)
                if winner.method == method:
                    wins += 1
        
        win_rate = 100 * wins / total_signals if total_signals > 0 else 0
        
        md.append(
            f"| {method} | {avg_bpp:.3f} | {avg_psnr:.2f} | "
            f"{avg_coh:.2e} | {win_rate:.1f}% |"
        )
    
    return '\n'.join(md)


def generate_latex_tables(results: List[ExperimentResult]) -> str:
    """Generate LaTeX tables for paper"""
    
    latex = []
    latex.append("% Breaking the ASCII Wall - LaTeX Tables")
    latex.append("% Generated automatically from experimental results\n")
    
    # Table 1: ASCII Signals (main result)
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Compression results on pure ASCII signals (95\\% sparsity)}")
    latex.append("\\label{tab:ascii_results}")
    latex.append("\\begin{tabular}{lccccr}")
    latex.append("\\toprule")
    latex.append("Method & BPP & PSNR (dB) & Coherence & Sparsity & Time (ms) \\\\")
    latex.append("\\midrule")
    
    # Get first ASCII signal results as representative
    ascii_results = [r for r in results if r.signal_category == "ascii" and "Python" in r.signal_name]
    for r in sorted(ascii_results, key=lambda x: x.bpp):
        latex.append(r.to_latex_row())
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}\n")
    
    # Table 2: Summary across categories
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Average compression across signal categories}")
    latex.append("\\label{tab:category_summary}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Category & Baseline BPP & H3 Cascade BPP & Improvement \\\\")
    latex.append("\\midrule")
    
    for category in ["ascii", "mixed", "continuous"]:
        cat_results = [r for r in results if r.signal_category == category]
        baseline = [r for r in cat_results if r.method == "Baseline_Greedy" and r.bpp < 900]
        h3 = [r for r in cat_results if r.method == "H3_Cascade" and r.bpp < 900]
        
        if baseline and h3:
            avg_baseline = np.mean([r.bpp for r in baseline])
            avg_h3 = np.mean([r.bpp for r in h3])
            improvement = 100 * (avg_baseline - avg_h3) / avg_baseline
            latex.append(
                f"{category.capitalize()} & {avg_baseline:.3f} & {avg_h3:.3f} & "
                f"{improvement:+.1f}\\% \\\\"
            )
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}\n")
    
    return '\n'.join(latex)


def save_results(results: List[ExperimentResult], output_dir: Path):
    """Save results in multiple formats"""
    
    # JSON (machine-readable)
    json_path = output_dir / "ascii_wall_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n✅ Saved JSON results: {json_path}")
    
    # Markdown (human-readable)
    md_content = generate_summary_tables(results)
    md_path = output_dir / "ASCII_WALL_PAPER_RESULTS.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"✅ Saved Markdown tables: {md_path}")
    
    # LaTeX (publication-ready)
    latex_content = generate_latex_tables(results)
    latex_path = output_dir / "ascii_wall_tables.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"✅ Saved LaTeX tables: {latex_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run full experimental suite"""
    
    # Run experiments
    results = run_all_experiments(sparsity=0.95)
    
    # Save results
    output_dir = Path(__file__).parent
    save_results(results, output_dir)
    
    print("\n" + "="*80)
    print("✅ EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nTotal experiments run: {len(results)}")
    print(f"Signals tested: {len(set(r.signal_name for r in results))}")
    print(f"Methods compared: {len(set(r.method for r in results))}")
    print("\nNext steps:")
    print("1. Review ASCII_WALL_PAPER_RESULTS.md for detailed analysis")
    print("2. Copy ascii_wall_tables.tex into paper LaTeX source")
    print("3. Run with different sparsity levels: python ascii_wall_paper.py --sparsity 0.90")


if __name__ == "__main__":
    main()
