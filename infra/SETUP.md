# Oracle Cloud VM — Automated Retry Setup

Automate ARM A1 instance creation on your personal laptop using OCI CLI.

---

## 1. Install OCI CLI

```bash
# macOS
brew install oci-cli

# Or universal installer (any OS)
bash -c "$(curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh)"
```

Verify:
```bash
oci --version
```

## 2. Configure OCI CLI

```bash
oci setup config
```

You'll be prompted for:

| Prompt | Where to find it |
|---|---|
| **User OCID** | Oracle Console → Profile (top-right) → My Profile → OCID |
| **Tenancy OCID** | Oracle Console → Profile → Tenancy → OCID |
| **Region** | `ap-hyderabad-1` |
| **Generate API key?** | Say **Yes** — it creates `~/.oci/oci_api_key.pem` |

After setup, **upload the public key** to Oracle:
1. Oracle Console → Profile → My Profile → **API Keys** → Add API Key
2. Choose **Paste Public Key**
3. Paste contents of `~/.oci/oci_api_key_public.pem`

Verify:
```bash
oci iam region list --output table
```

## 3. Clone the Repo

```bash
git clone https://github.com/GauravKumarYadav/rag-ai.git
cd rag-ai
```

Make sure the SSH key pair is in `keys/`:
```bash
ls keys/
# Should show: ssh-key-2026-02-18.key  ssh-key-2026-02-18.key.pub
```

If not, copy them from your office laptop.

## 4. Run the Retry Script

```bash
bash infra/retry-create.sh
```

This will:
- Attempt to create the ARM A1 instance every **5 minutes**
- Retry up to **288 times** (~24 hours)
- Send a **macOS notification** when it succeeds
- Print the instance details on success

You can leave it running overnight. Best success rates are **3-6 AM IST**.

### Customize retry interval:
Edit `infra/retry-create.sh` and change:
```bash
RETRY_INTERVAL_SECONDS=300  # Change to 120 for every 2 minutes
```

## 5. After VM is Created

Once the script succeeds, get the public IP:
```bash
# List your instances
oci compute instance list --compartment-id "ocid1.tenancy.oc1..aaaaaaaab3fn2mfeejtfbvwoeh2rfq2mepuy3mrrzbqwnj6pgcrdkrs7vzkq" --output table

# Get public IP (use the instance OCID from above)
oci compute instance list-vnics --instance-id <INSTANCE_OCID> --query 'data[0]."public-ip"' --raw-output
```

Then follow these steps:

### a. SSH into the VM
```bash
ssh -i keys/ssh-key-2026-02-18.key ubuntu@<PUBLIC_IP>
```

### b. Run the server setup
```bash
# From your local machine
ssh -i keys/ssh-key-2026-02-18.key ubuntu@<PUBLIC_IP> 'bash -s' < deploy/setup-server.sh
```

### c. Configure .env on the server
```bash
ssh -i keys/ssh-key-2026-02-18.key ubuntu@<PUBLIC_IP>
cd /opt/rag-ai
cp .env.production.example .env
nano .env
# Set: LLM__GROQ__API_KEY=your-key
# Set: JWT__SECRET_KEY=$(openssl rand -hex 32)
```

### d. Open port 80 in Oracle Security List
1. Oracle Console → Networking → VCN → rag-ai-vcn
2. Security Lists → Default Security List → Add Ingress Rules
3. Source: `0.0.0.0/0`, Protocol: TCP, Destination Port: **80**

### e. Add GitHub Secret
- `github.com/GauravKumarYadav/rag-ai` → Settings → Secrets → Actions
- Add `OCI_HOST` = `<PUBLIC_IP>`

### f. First deploy
```bash
ssh -i keys/ssh-key-2026-02-18.key ubuntu@<PUBLIC_IP>
cd /opt/rag-ai && bash deploy/deploy.sh
```

### g. Verify
```bash
curl http://<PUBLIC_IP>/health
# Should return: healthy
```

After this, every `git push origin main` auto-deploys via GitHub Actions CI/CD.
