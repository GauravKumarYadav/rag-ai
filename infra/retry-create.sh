#!/usr/bin/env bash
# =============================================================================
# Retry Oracle Cloud ARM A1 instance creation until capacity is available
# Prerequisites: OCI CLI installed and configured
#   brew install oci-cli
#   oci setup config
# Usage: bash infra/retry-create.sh
# =============================================================================
set -uo pipefail

# --- Configuration (from your main.tf) ---
COMPARTMENT_ID="ocid1.tenancy.oc1..aaaaaaaab3fn2mfeejtfbvwoeh2rfq2mepuy3mrrzbqwnj6pgcrdkrs7vzkq"
AVAILABILITY_DOMAIN="oVIz:AP-HYDERABAD-1-AD-1"
SUBNET_ID="ocid1.subnet.oc1.ap-hyderabad-1.aaaaaaaakprgnssu5m7mysi45zaoi5zbn4vmtgcrt4kuyq6fjlfj7wsx57ga"
IMAGE_ID="ocid1.image.oc1.ap-hyderabad-1.aaaaaaaape6pxqtgswnbolzxyjok2klghymdar5hqa2qnu3mvr2tsldahxxq"
SHAPE="VM.Standard.A1.Flex"
OCPUS=4
MEMORY_GB=24
BOOT_VOLUME_GB=100
DISPLAY_NAME="rag-ai-server"
SSH_KEY_FILE="keys/ssh-key-2026-02-18.key.pub"

# Retry settings
RETRY_INTERVAL_SECONDS=300  # 5 minutes between retries
MAX_RETRIES=288             # 288 x 5min = 24 hours

echo "=========================================="
echo " Oracle Cloud ARM A1 - Auto Retry"
echo "=========================================="
echo " Shape: $SHAPE ($OCPUS OCPUs, ${MEMORY_GB}GB RAM)"
echo " Region: AP-HYDERABAD-1"
echo " Retry every: ${RETRY_INTERVAL_SECONDS}s"
echo " Max retries: $MAX_RETRIES (~24 hours)"
echo "=========================================="
echo ""

# Read SSH public key
if [ ! -f "$SSH_KEY_FILE" ]; then
    echo "ERROR: SSH public key not found at $SSH_KEY_FILE"
    exit 1
fi
SSH_KEY=$(cat "$SSH_KEY_FILE")

for i in $(seq 1 $MAX_RETRIES); do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Attempt $i/$MAX_RETRIES — Creating instance..."

    RESULT=$(oci compute instance launch \
        --compartment-id "$COMPARTMENT_ID" \
        --availability-domain "$AVAILABILITY_DOMAIN" \
        --shape "$SHAPE" \
        --shape-config "{\"ocpus\": $OCPUS, \"memoryInGBs\": $MEMORY_GB}" \
        --display-name "$DISPLAY_NAME" \
        --image-id "$IMAGE_ID" \
        --subnet-id "$SUBNET_ID" \
        --assign-public-ip true \
        --boot-volume-size-in-gbs "$BOOT_VOLUME_GB" \
        --metadata "{\"ssh_authorized_keys\": \"$SSH_KEY\"}" \
        --is-pv-encryption-in-transit-enabled true \
        2>&1)

    if echo "$RESULT" | grep -q '"lifecycle-state"'; then
        echo ""
        echo "=========================================="
        echo " SUCCESS! Instance created!"
        echo "=========================================="
        echo "$RESULT" | grep -E '"display-name"|"id"|"lifecycle-state"|"public-ip"'
        echo ""
        echo "Wait 2-3 minutes for it to boot, then get the public IP:"
        echo "  oci compute instance list-vnics --instance-id <INSTANCE_OCID> | grep 'public-ip'"
        echo ""

        # macOS notification
        osascript -e 'display notification "ARM A1 instance created!" with title "Oracle Cloud" sound name "Glass"' 2>/dev/null || true

        exit 0
    fi

    if echo "$RESULT" | grep -qi "out of capacity\|out of host capacity\|capacity"; then
        echo "  → Out of capacity. Retrying in ${RETRY_INTERVAL_SECONDS}s..."
    else
        echo "  → Error: $(echo "$RESULT" | head -5)"
        echo "  → Retrying in ${RETRY_INTERVAL_SECONDS}s..."
    fi

    sleep "$RETRY_INTERVAL_SECONDS"
done

echo "Max retries reached. Try again later or try a different availability domain."
exit 1
