# Lean + Python, add what you need via pip/conda
FROM jupyter/minimal-notebook:latest

# (Optional) preinstall common libs
RUN pip install --no-cache-dir numpy pandas matplotlib

