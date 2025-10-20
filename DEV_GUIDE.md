# Development Build Guide

## 🎯 Build UI riêng (không conflict với upstream)

### Build UI:

```bash
cd ui/litellm-dashboard
./build_ui_dev.sh
```

UI sẽ được build vào: `litellm/proxy/_my_experimental/out/`

**Không động vào:** `litellm/proxy/_experimental/out/` (để tránh conflict khi merge từ upstream)

---

## 🚀 Run Proxy với UI riêng

### Option 1: Sử dụng UI build riêng

Tạo config file `config_dev.yaml`:

```yaml
model_list:
  - model_name: agentrouter/claude-3-5-sonnet-20241022
    litellm_params:
      model: agentrouter/claude-3-5-sonnet-20241022
      api_key: os.environ/AGENTROUTER_API_KEY

general_settings:
  ui_access_mode: admin
  ui_path: "litellm/proxy/_my_experimental/out"  # UI riêng của bạn
```

```bash
# Run proxy với UI riêng
litellm --config config_dev.yaml
```

### Option 2: Run proxy với UI mặc định (từ upstream)

```bash
# Không set ui_path, proxy sẽ dùng UI trong litellm/proxy/_experimental/out/
litellm --config your_config.yaml
```

---

## 🔧 Test AgentRouter

### Test với code Python:

```bash
cd /home/long/Workspace/litellm

# Activate virtual environment hiện tại
source .venv/bin/activate

# Test với Python
python -c "
import litellm
import os

os.environ['AGENTROUTER_API_KEY'] = 'your-key-here'

response = litellm.completion(
    model='agentrouter/claude-3-5-sonnet-20241022',
    messages=[{'role': 'user', 'content': 'Hello!'}],
    max_tokens=10
)
print(response)
"
```

### Run unit tests:

```bash
pytest tests/test_litellm/test_agentrouter.py -v
```
 
---

## 📦 Build và commit workflow

### Khi có thay đổi:

```bash
# 1. Làm việc như bình thường
git add .
git commit -m "feat: your changes"

# 2. Push lên fork
git push origin feature/add-agentrouter-provider

# 3. Nếu có thay đổi UI, build riêng để test
cd ui/litellm-dashboard
./build_ui_dev.sh

# 4. KHÔNG commit build output
# (ui/litellm-dashboard/out/ và litellm/proxy/_experimental/out/ đều trong .gitignore)
```

### Sync với upstream:

```bash
# Pull changes từ upstream
git fetch upstream
git checkout main
git merge upstream/main

# Rebase feature branch của bạn
git checkout feature/add-agentrouter-provider
git rebase main

# Resolve conflicts nếu có, rồi push
git push origin feature/add-agentrouter-provider --force-with-lease
```

---

## 📂 Cấu trúc thư mục

```
/home/long/Workspace/litellm/          # Fork repository
├── .venv/                             # Virtual env (gitignored)
├── litellm/
│   ├── llms/agentrouter/              # ✅ Code của bạn
│   └── proxy/
│       ├── _experimental/out/         # ❌ KHÔNG commit (upstream UI build)
│       └── _my_experimental/out/      # ✅ UI build riêng của bạn (gitignored)
├── ui/litellm-dashboard/
│   ├── build_ui.sh                    # Script gốc (build vào _experimental/)
│   ├── build_ui_dev.sh                # ✅ Script của bạn (build vào _my_experimental/)
│   └── out/                           # Temp build output (gitignored)
└── tests/test_litellm/
    └── test_agentrouter.py            # ✅ Tests của bạn
```

---

## ⚙️ Environment Variables

```bash
# AgentRouter API Key
export AGENTROUTER_API_KEY="your-key"
export AR_API_KEY="your-key"  # Alternative

# Debug mode
export LITELLM_LOG=DEBUG
```

**Note:** UI path được config trong `config_dev.yaml`, không cần environment variable.

---

## 🎉 Quick Start

```bash
# 1. Đảm bảo đang ở branch feature
git checkout feature/add-agentrouter-provider

# 2. Activate venv
source .venv/bin/activate

# 3. Test code
pytest tests/test_litellm/test_agentrouter.py -v

# 4. Build UI nếu cần
cd ui/litellm-dashboard && ./build_ui_dev.sh

# 5. Commit và push
git add .
git commit -m "feat: your changes"
git push origin feature/add-agentrouter-provider
```
