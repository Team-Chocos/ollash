#!/bin/bash
# install_menu_deps.sh - Install enhanced menu dependencies for Ollash

echo "🚀 Installing Enhanced Menu Dependencies for Ollash"
echo "=================================================="

# Function to check if package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null && echo "✓ $1 already installed" || return 1
}

# Install packages
packages=("prompt_toolkit:prompt-toolkit" "inquirer:inquirer" "rich:rich" "pyfzf:pyfzf")

echo "📦 Installing Python packages..."
echo

for package in "${packages[@]}"; do
    IFS=':' read -r import_name pip_name <<< "$package"
    
    if check_package "$import_name"; then
        continue
    fi
    
    echo "📥 Installing $pip_name..."
    pip install "$pip_name"
    
    if check_package "$import_name"; then
        echo "✅ $pip_name installed successfully"
    else
        echo "❌ Failed to install $pip_name"
    fi
    echo
done

echo "🎯 Installation Summary:"
echo "======================"

# Check final status
echo -n "Prompt Toolkit (Best): "
check_package "prompt_toolkit" && echo "✅ Available" || echo "❌ Not available"

echo -n "Inquirer (Good): "
check_package "inquirer" && echo "✅ Available" || echo "❌ Not available"

echo -n "Rich (Beautiful): "
check_package "rich" && echo "✅ Available" || echo "❌ Not available"

echo -n "PyFZF (FZF wrapper): "
check_package "pyfzf" && echo "✅ Available" || echo "❌ Not available"

echo
echo "🎉 Setup complete! Your Ollash dropdown menus are now enhanced!"
echo "💡 Restart your terminal and try: ollash shell"