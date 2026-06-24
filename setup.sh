#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# AI Code Review Agent — Local Setup Script
# Usage:  bash setup.sh
# ─────────────────────────────────────────────────────────────

set -e  # Exit immediately on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║   AI Code Review Agent — Setup           ║${RESET}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${RESET}"
echo ""

# ── 1. Check Python version ──────────────────────────────────
info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED="3.11"
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"; then
    success "Python $PYTHON_VERSION — OK"
else
    error "Python 3.11+ is required. Found: $PYTHON_VERSION"
fi

# ── 2. Create virtual environment ────────────────────────────
if [ ! -d "venv" ]; then
    info "Creating virtual environment..."
    python3 -m venv venv
    success "Virtual environment created at ./venv"
else
    warn "Virtual environment already exists — skipping creation."
fi

# ── 3. Activate venv ─────────────────────────────────────────
info "Activating virtual environment..."
# shellcheck disable=SC1091
source venv/bin/activate
success "Virtual environment activated"

# ── 4. Upgrade pip ───────────────────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip --quiet
success "pip upgraded"

# ── 5. Install dependencies ──────────────────────────────────
info "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet
success "All dependencies installed"

# ── 6. Create .env from template ─────────────────────────────
if [ ! -f ".env" ]; then
    info "Creating .env from template..."
    cp .env.example .env
    warn ".env created — please fill in your API keys before running."
else
    warn ".env already exists — skipping."
fi

# ── 7. Create reports directory ──────────────────────────────
mkdir -p reports
success "Reports directory ready"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║   Setup Complete! 🎉                     ║${RESET}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${RESET}"
echo ""
echo "  Next steps:"
echo "  1. Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)"
echo "  2. Activate venv:  source venv/bin/activate"
echo "  3. Run the app:    streamlit run app.py"
echo ""
