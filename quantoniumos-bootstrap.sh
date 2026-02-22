#!/usr/bin/env bash
################################################################################
# QuantoniumOS Complete Bootstrap Script
# ========================================
# 
# This script performs a complete setup of the QuantoniumOS environment:
# - System dependencies (compilers, tools, libraries)
# - Python environment setup
# - Native module compilation (C++/Assembly RFT kernels)
# - Hardware verification tools
# - Benchmark suite validation
# - Project structure verification
#
# Usage:
#   ./quantoniumos-bootstrap.sh              # Full setup
#   ./quantoniumos-bootstrap.sh --minimal    # Core only (no AI/ML)
#   ./quantoniumos-bootstrap.sh --dev        # Development environment
#   ./quantoniumos-bootstrap.sh --hardware   # Hardware synthesis tools
#
# Author: Luis M. Minier / mandcony
# License: See LICENSE.md
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_MIN_VERSION="3.9"
BUILD_DIR="${PROJECT_ROOT}/src/rftmw_native/build"
VENV_DIR="${PROJECT_ROOT}/.venv"
LOG_FILE="${PROJECT_ROOT}/bootstrap.log"

# Parse command line arguments
MODE="full"
SKIP_VENV=false
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            MODE="minimal"
            shift
            ;;
        --dev)
            MODE="dev"
            shift
            ;;
        --hardware)
            MODE="hardware"
            shift
            ;;
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --help)
            echo "QuantoniumOS Bootstrap Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --minimal       Install core dependencies only"
            echo "  --dev           Install development tools and test suites"
            echo "  --hardware      Setup hardware synthesis environment"
            echo "  --skip-venv     Don't create/activate virtual environment"
            echo "  --force-rebuild Force rebuild of native modules"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Utility Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

section_header() {
    echo "" | tee -a "$LOG_FILE"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN} $1${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 found: $(command -v $1)"
        return 0
    else
        log_warning "$1 not found"
        return 1
    fi
}

################################################################################
# System Information
################################################################################

print_system_info() {
    section_header "System Information"
    log_info "OS: $(uname -s)"
    log_info "Architecture: $(uname -m)"
    log_info "Kernel: $(uname -r)"
    log_info "Python: $(python3 --version 2>&1 || echo 'Not found')"
    log_info "Project Root: $PROJECT_ROOT"
    log_info "Installation Mode: $MODE"
}

################################################################################
# Check System Dependencies
################################################################################

check_system_dependencies() {
    section_header "Checking System Dependencies"
    
    local missing_deps=()
    
    # Core build tools
    check_command "gcc" || missing_deps+=("gcc")
    check_command "g++" || missing_deps+=("g++")
    check_command "make" || missing_deps+=("make")
    check_command "cmake" || missing_deps+=("cmake")
    check_command "git" || missing_deps+=("git")
    check_command "python3" || missing_deps+=("python3")
    check_command "pip3" || missing_deps+=("python3-pip")
    
    # Assembly tools
    check_command "nasm" || missing_deps+=("nasm")
    
    # Optional but recommended
    check_command "ninja" || log_warning "ninja-build not found (optional speedup)"
    check_command "ccache" || log_warning "ccache not found (optional build cache)"
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install them using your package manager:"
        log_info "  Ubuntu/Debian: sudo apt-get install ${missing_deps[*]}"
        log_info "  Fedora/RHEL:   sudo dnf install ${missing_deps[*]}"
        log_info "  macOS:         brew install ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "All required system dependencies found"
}

################################################################################
# Python Environment Setup
################################################################################

setup_python_environment() {
    section_header "Setting up Python Environment"
    
    # Check Python version
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_info "Python version: $python_version"
    
    if [ "$SKIP_VENV" = false ]; then
        # Create virtual environment if it doesn't exist
        if [ ! -d "$VENV_DIR" ]; then
            log_info "Creating virtual environment at $VENV_DIR"
            python3 -m venv "$VENV_DIR"
            log_success "Virtual environment created"
        else
            log_info "Virtual environment already exists"
        fi
        
        # Activate virtual environment
        log_info "Activating virtual environment"
        source "$VENV_DIR/bin/activate"
        log_success "Virtual environment activated"
    fi
    
    # Upgrade pip
    log_info "Upgrading pip, setuptools, and wheel"
    python3 -m pip install --upgrade pip setuptools wheel
    
    log_success "Python environment ready"
}

################################################################################
# Install Python Dependencies
################################################################################

install_python_dependencies() {
    section_header "Installing Python Dependencies"
    
    cd "$PROJECT_ROOT"
    
    case $MODE in
        minimal)
            log_info "Installing minimal dependencies"
            pip install -e ".[dev]" --no-deps
            pip install numpy scipy matplotlib pandas sympy
            ;;
        dev)
            log_info "Installing development dependencies"
            pip install -e ".[dev,ai,image]"
            ;;
        hardware)
            log_info "Installing core + hardware verification tools"
            pip install -e ".[dev]"
            pip install cocotb pytest-xdist
            ;;
        full|*)
            log_info "Installing full dependencies"
            pip install -e ".[dev,ai,image]"
            ;;
    esac
    
    log_success "Python dependencies installed"
}

################################################################################
# Build Native Modules
################################################################################

build_native_modules() {
    section_header "Building Native RFT Modules"
    
    local native_dir="${PROJECT_ROOT}/src/rftmw_native"
    
    if [ ! -d "$native_dir" ]; then
        log_warning "Native module directory not found at $native_dir"
        return 0
    fi
    
    cd "$native_dir"
    
    # Clean build if force rebuild
    if [ "$FORCE_REBUILD" = true ] && [ -d "$BUILD_DIR" ]; then
        log_info "Force rebuild: cleaning existing build directory"
        rm -rf "$BUILD_DIR"
    fi
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure with CMake
    log_info "Configuring native modules with CMake"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_LTO=ON \
        -DRFTMW_ENABLE_ASM=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF
    
    # Build
    log_info "Building native modules (this may take a few minutes)"
    local num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    make -j"$num_cores"
    
    # Verify build
    if [ -f "rftmw_native"*.so ]; then
        log_success "Native modules built successfully"
        log_info "Module location: $BUILD_DIR"
    else
        log_error "Native module build failed"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
}

################################################################################
# Setup Hardware Tools
################################################################################

setup_hardware_tools() {
    section_header "Setting up Hardware Tools"
    
    if [ "$MODE" != "hardware" ] && [ "$MODE" != "full" ]; then
        log_info "Skipping hardware tools (not in hardware/full mode)"
        return 0
    fi
    
    local hw_dir="${PROJECT_ROOT}/hardware"
    cd "$hw_dir"
    
    # Check for Verilog tools
    if check_command "iverilog"; then
        log_info "Running hardware verification tests"
        if [ -f "verify_fixes.sh" ]; then
            bash verify_fixes.sh || log_warning "Hardware verification reported issues"
        fi
    else
        log_warning "iverilog not found - hardware simulation unavailable"
        log_info "To enable: sudo apt-get install iverilog gtkwave"
    fi
    
    # Make scripts executable
    chmod +x generate_all_figures.sh 2>/dev/null || true
    chmod +x verify_fixes.sh 2>/dev/null || true
    
    cd "$PROJECT_ROOT"
    log_success "Hardware tools configured"
}

################################################################################
# Verify Installation
################################################################################

verify_installation() {
    section_header "Verifying Installation"
    
    cd "$PROJECT_ROOT"
    
    # Test Python imports
    log_info "Testing core imports..."
    python3 -c "import numpy; print(f'NumPy {numpy.__version__}')" || log_error "NumPy import failed"
    python3 -c "import scipy; print(f'SciPy {scipy.__version__}')" || log_error "SciPy import failed"
    python3 -c "import algorithms; print('QuantoniumOS algorithms loaded')" || log_error "Algorithms import failed"
    
    # Test native module
    log_info "Testing native RFT module..."
    python3 -c "
import sys
sys.path.insert(0, 'src/rftmw_native/build')
try:
    import rftmw_native
    print('Native RFT module loaded successfully')
except ImportError as e:
    print(f'Native module not available: {e}')
" || log_warning "Native module test failed (fallback to pure Python)"
    
    # Test RFT functionality
    log_info "Testing RFT core functionality..."
    python3 -c "
import numpy as np
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT

rft = CanonicalTrueRFT(128)
x = np.random.randn(128)
y = rft.transform(x)
error = rft.get_unitarity_error()
print(f'RFT unitarity error: {error:.2e}')
assert error < 1e-10, 'Unitarity test failed'
print('✓ RFT core functionality verified')
" || log_error "RFT functionality test failed"
    
    log_success "Installation verification complete"
}

################################################################################
# Run Quick Validation
################################################################################

run_quick_validation() {
    section_header "Running Quick Validation Suite"
    
    if [ "$MODE" = "minimal" ]; then
        log_info "Skipping validation in minimal mode"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    if [ -f "scripts/validation/validate_system.py" ]; then
        log_info "Running system validation..."
        python3 scripts/validation/validate_system.py || log_warning "Some validation tests failed"
    fi
    
    log_success "Validation complete"
}

################################################################################
# Generate Summary
################################################################################

generate_summary() {
    section_header "Installation Summary"
    
    cat << EOF | tee -a "$LOG_FILE"
${GREEN}✓ QuantoniumOS Bootstrap Complete${NC}

Installation Details:
  • Mode:           $MODE
  • Project Root:   $PROJECT_ROOT
  • Python:         $(python3 --version 2>&1)
  • Virtual Env:    $([ "$SKIP_VENV" = false ] && echo "$VENV_DIR" || echo "Disabled")
  • Native Module:  $([ -f "$BUILD_DIR/rftmw_native"*.so ] && echo "Built" || echo "Not built")
  • Log File:       $LOG_FILE

Quick Start Commands:
  • Activate venv:  source $VENV_DIR/bin/activate
  • Run tests:      pytest tests/ -v
  • Run benchmarks: python benchmarks/run_all_benchmarks.py
    • Validate:       python scripts/validation/validate_system.py
  • Hardware sim:   cd hardware && bash verify_fixes.sh

Next Steps:
  1. Review the documentation: docs/INDEX.md
    2. Run the benchmark suite: python benchmarks/run_all_benchmarks.py
    3. Explore examples: experiments/README.md
    4. Read validation reports: docs/validation/RFT_THEOREMS.md

For issues, see: docs/technical/guides/TROUBLESHOOTING.md

EOF
}

################################################################################
# Main Installation Flow
################################################################################

main() {
    clear
    
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              QuantoniumOS Bootstrap Installer                 ║
║           Physics-Inspired Symbolic Computing OS              ║
║                                                               ║
║  Complete environment setup for development and deployment    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

EOF
    
    # Initialize log
    echo "QuantoniumOS Bootstrap Log - $(date)" > "$LOG_FILE"
    
    # Run installation steps
    print_system_info
    check_system_dependencies
    setup_python_environment
    install_python_dependencies
    build_native_modules
    setup_hardware_tools
    verify_installation
    run_quick_validation
    generate_summary
    
    log_success "QuantoniumOS is ready to use!"
    
    if [ "$SKIP_VENV" = false ]; then
        echo ""
        echo -e "${YELLOW}Don't forget to activate the virtual environment:${NC}"
        echo -e "${CYAN}  source $VENV_DIR/bin/activate${NC}"
        echo ""
    fi
}

# Run main installation
main "$@"
