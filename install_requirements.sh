#!/usr/bin/env bash
# install_requirements.sh
# Handles cleaning up broken GPU packages and installing requirements.txt safely

echo "============================================================"
echo "🧹 Cleaning up pip cache..."
pip cache purge

echo "============================================================"
echo "🗑️  Uninstalling conflicting cudf and cuda packages..."
# We ignore errors here in case they are already uninstalled
pip uninstall -y cudf-cu12 cuda-python cuda-bindings cudf_cu12 rmm-cu12 cupy-cuda12x pylibraft-cu12 pylibcugraph-cu12 raft-dask-cu12 cugraph-cu12 2>/dev/null || true

echo "============================================================"
echo "📦 Installing clean requirements.txt..."
pip install -r requirements.txt

echo "============================================================"
echo "✅ Installation complete!"
