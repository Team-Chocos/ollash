#!/bin/bash

set -e  # Exit on error

echo "🧹 Cleaning previous builds..."
rm -rf build dist *.egg-info

echo "📦 Building source and wheel distribution..."
python setup.py sdist bdist_wheel

echo "🚀 Uploading to PyPI..."
twine upload dist/*

echo "✅ Build and publish completed successfully!"
