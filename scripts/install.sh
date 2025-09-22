set -e

echo "==================================="
echo "KLARNET Voice Assistant Installation"
echo "==================================="

# Check system requirements
check_requirements() {
    echo "Checking system requirements..."

    # Check for Rust
    if ! command -v cargo &> /dev/null; then
        echo "Rust not found. Installing..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi

    # Check for Python
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi

    # Check for CUDA (optional)
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA detected"
        export KLARNET_MODE="gpu"
    else
        echo "No CUDA detected, using CPU mode"
        export KLARNET_MODE="cpu"
    fi
}

# Install dependencies
install_dependencies() {
    echo "Installing dependencies..."

    # System packages
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            libasound2-dev \
            portaudio19-dev \
            libssl-dev \
            pkg-config
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install portaudio
    fi

    # Python dependencies
    pip3 install --user -r requirements.txt
}

# Download models
download_models() {
    echo "Downloading models..."

    mkdir -p models
    python3 scripts/download_model.py --size medium --output models/

    if [ "$KLARNET_MODE" == "cpu" ]; then
        python3 scripts/download_model.py --size small --output models/
    fi
}

# Build project
build_project() {
    echo "Building KLARNET..."

    if [ "$KLARNET_MODE" == "gpu" ]; then
        cargo build --release --features gpu
    else
        cargo build --release
    fi
}

# Setup configuration
setup_config() {
    echo "Setting up configuration..."

    mkdir -p config cache logs

    if [ ! -f config/klarnet.toml ]; then
        cp config/klarnet.example.toml config/klarnet.toml
    fi

    # Set appropriate mode in config
    if [ "$KLARNET_MODE" == "cpu" ]; then
        sed -i 's/device = "cuda"/device = "cpu"/' config/klarnet.toml
        sed -i 's/compute_type = "int8_float16"/compute_type = "int8"/' config/klarnet.toml
    fi
}

# Create systemd service
install_service() {
    echo "Installing systemd service..."

    sudo cp deploy/klarnet.service /etc/systemd/system/
    sudo systemctl daemon-reload

    echo "Service installed. To start:"
    echo "  sudo systemctl start klarnet"
    echo "  sudo systemctl enable klarnet  # For autostart"
}

# Main installation
main() {
    check_requirements
    install_dependencies
    download_models
    build_project
    setup_config

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        read -p "Install as systemd service? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_service
        fi
    fi

    echo ""
    echo "==================================="
    echo "Installation complete!"
    echo "Run './target/release/klarnet' to start"
    echo "==================================="
}

main