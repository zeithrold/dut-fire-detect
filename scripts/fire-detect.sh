# fire-detect.sh [backend] [ref]
# backend: {cpu, cuda11, cuda12}, optional, default is cpu
# ref: {main, v1.0.0, ...}, optional, default is main
# DirectML is not available in Linux/macOS.

backend="cpu"
if [ $# -eq 1 ]; then
    backend=$1
fi

ref="main"
if [ $# -eq 2 ]; then
    ref=$2
fi

GITHUB_ENDPOINT="${GITHUB_ENDPOINT:-github.com}"

WEIGHT_URL="https://${GITHUB_ENDPOINT}/zeithrold/dut-fire-detect/releases/download/v1.1.0/model.onnx"

function check_python {
    if [ ! -x "$(command -v python)" ]; then
        echo "Python is not installed or not in PATH"
        exit 1
    fi
}

function check_module {
    activate_venv
    if [ ! -x "$(command -v dut-fire-detect)" ]; then
        echo "fire-detect is not installed..."
        return 1
    fi
    deactivate
    return 0
}

function create_venv {
    python -m venv venv
}

function check_venv {
    if [ ! -d "venv" ]; then
        echo "Virtual environment is not created..."
        return 1
    fi
    return 0
}

function activate_venv {
    source venv/bin/activate
}

function run {
    dut-fire-detect
}

function check_already_installed {
    if check_venv; then
        if check_module; then
            return 0
        else
            install_packages
        fi
    else
        create_venv
        install_packages
    fi
}

function check_weights {
    # Check if weights/model.onnx exists
    if [ ! -f "weights/model.onnx" ]; then
        echo "Downloading weights..."
        mkdir -p weights
        wget -O weights/model.onnx $WEIGHT_URL
    fi
}

function install_packages {
    echo "Installing fire-detect..."
    activate_venv
    # Install required packages
    pip install "dut-fire-detect[${backend}] @ https://github.com/zeithrold/dut-fire-detect/archive/${ref}.zip"
    deactivate
}

check_python
check_already_installed
check_weights
run
