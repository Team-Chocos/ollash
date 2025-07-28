#!/bin/bash

set -e  # Exit on error

# WARNING: Don't commit this to git with your real token!
PYPI_API_TOKEN="pypi-your-api-token-here"

echo "Cleaning previous builds..."
rm -rf build dist *.egg-info

echo "Building source and wheel distribution..."
python setup.py sdist bdist_wheel

echo "Uploading to PyPI..."
twine upload dist/* --username __token__ --password "$PYPI_API_TOKEN"

echo "Build and publish completed successfully!"