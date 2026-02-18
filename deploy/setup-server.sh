#!/usr/bin/env bash
# =============================================================================
# One-time server setup script for Oracle Cloud ARM A1 instance (Ubuntu 22.04)
# Usage: ssh ubuntu@<VM_IP> 'bash -s' < deploy/setup-server.sh
# =============================================================================
set -euo pipefail

echo "=========================================="
echo " RAG AI - Oracle Cloud Server Setup"
echo "=========================================="

# --- System updates ---
echo "[1/6] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# --- Install Docker ---
echo "[2/6] Installing Docker Engine..."
if ! command -v docker &> /dev/null; then
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to log out and back in for group changes."
else
    echo "Docker already installed, skipping."
fi

# --- Install Git ---
echo "[3/6] Installing Git..."
sudo apt-get install -y git

# --- Configure firewall (iptables) ---
echo "[4/6] Configuring iptables for ports 80, 443..."
# Oracle Cloud Ubuntu images use iptables rules that block traffic by default.
# We need to open ports 80 and 443 in the OS firewall (in addition to the VCN security list).
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
# Persist iptables rules across reboots
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save

# --- Create app directory ---
echo "[5/6] Setting up application directory..."
sudo mkdir -p /opt/rag-ai
sudo chown "$USER":"$USER" /opt/rag-ai

# --- Clone repository ---
echo "[6/6] Cloning repository..."
if [ ! -d "/opt/rag-ai/.git" ]; then
    git clone https://github.com/GauravKumarYadav/rag-ai.git /opt/rag-ai
else
    echo "Repository already cloned, pulling latest..."
    cd /opt/rag-ai && git pull
fi

# --- Create data directories ---
mkdir -p /opt/rag-ai/data/{chroma,bm25,raw}

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. cd /opt/rag-ai"
echo "  2. cp .env.production.example .env"
echo "  3. Edit .env with your API keys (nano .env)"
echo "  4. bash deploy/deploy.sh"
echo ""
echo "NOTE: Log out and back in for Docker group to take effect:"
echo "  exit && ssh ubuntu@<VM_IP>"
