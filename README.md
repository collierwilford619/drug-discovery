# NOVA - SN68

## High-throughput ML-driven drug screening.

NOVA harnesses global compute and collective intelligence to navigate huge unexplored chemical spaces, uncovering breakthrough compounds at a fraction of the cost and time.

## System Requirements

- Ubuntu 24.04 LTS (recommended)
- Python 3.12
- CUDA 12.6 (for GPU support)
- Sufficient RAM for ML model operations
- Internet connection for network participation

## Installation and Running

1. Clone the repository:

```bash
git clone <repository-url>
cd nova
```

2. Install dependencies:

   - For CPU:

   ```bash
   ./install_deps_cpu.sh
   ```

   - For CUDA 12.6:

   ```bash
   ./install_deps_cu126.sh
   ```

3. Run:

```bash
# Activate your virtual environment:
source .venv/bin/activate

# Run your script:
# miner:
python neurons/miner.py --wallet.name novawallet --wallet.hotkey novawallet-hotkey --logging.info
```
