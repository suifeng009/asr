# Kaggle API (KGAT) 使用完全指南

Kaggle 最近更新了认证机制，推出了全新的 **Kaggle API Token**（即以 `KGAT_` 开头的 Token）。与老版本需要配置 `kaggle.json`（包含 username 和 key）不同，新版 Token 的使用方式更倾向于 **环境变量** 和 **Bearer Token** 机制，变得更加简洁。

为了避免下次再踩坑，我为你整理了一份备忘录。

---

## 1. 最推荐的配置方式：环境变量配置

当你拿到一串类似 `KGAT_15421a278e5df4aa25cc0b423d17748b` 的 Token 时，**你不再需要写 `kaggle.json` 文件**，也不需要提供 Username。只需要将它设置为环境变量 `KAGGLE_API_TOKEN` 即可。

### Windows (PowerShell)
每次在终端执行 Kaggle 命令前，或者写在自动化脚本里：
```powershell
$env:KAGGLE_API_TOKEN="KGAT_你的完整Token"
kaggle datasets list
```

### Linux / macOS (Bash/Zsh)
在终端或 bash 脚本里：
```bash
export KAGGLE_API_TOKEN="KGAT_你的完整Token"
kaggle kernels push -p ./kaggle_kernel
```

> [!TIP]
> 只要终端里有了这个环境变量，你直接运行所有的 `kaggle` 命令（如下载、提交等）就都会自动以该 Token 的所有者身份进行。

---

## 2. 常用 Kaggle CLI 命令备忘

配置好环境变量后，你可以使用以下高频命令进行交互：

### 操作 Notebook (Kernel)
- **推送并运行 Kernel** (最常用)：
  ```bash
  kaggle kernels push -p /path/to/folder
  ```
  *(注：文件夹里必须有 `kernel-metadata.json` 和对应的代码文件)*
- **查看 Kernel 运行状态**：
  ```bash
  kaggle kernels status 你的用户名/你的kernel名字
  ```

### 操作数据集 (Datasets)
- **查看自己名下/私有数据集的文件列表**：
  ```bash
  kaggle datasets files 你的用户名/数据集名字
  ```
- **下载数据集**：
  ```bash
  kaggle datasets download 你的用户名/数据集名字 --unzip
  ```

---

## 3. 在 Python 代码中直接使用 (API 调用)

有时候我们需要用 Python 脚本自动化拉取信息，你可以这么做：

### 方法 A：利用 Kaggle 官方库
在导入 kaggle 库前设定好环境变量，官方库会自动识别：
```python
import os
# 设置环境变量后，再导入 kaggle，它会自动用该 Token 认证
os.environ['KAGGLE_API_TOKEN'] = "KGAT_你的完整Token"

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate() # 认证成功！

# 打印数据集文件
files = api.dataset_list_files('shadiao/asp800')
for f in files.files:
    print(f.name)
```

### 方法 B：直接发 HTTP 请求 (Requests)
Kaggle 的新 Token 兼容标准的 `Bearer` 认证，不需要额外的库就能调接口。
```python
import requests

token = "KGAT_你的完整Token"
headers = {
    "Authorization": f"Bearer {token}"
}

# 例如查询数据集的信息
url = "https://www.kaggle.com/api/v1/datasets/view/shadiao/asp800"
response = requests.get(url, headers=headers)
print(response.json())
```

> [!IMPORTANT]
> **关于遗留的 `kaggle.json`**：虽然 Kaggle 现在依然兼容把 `KGAT_...` 填入 `~/.kaggle/kaggle.json` 中 `"key"` 的字段（需要配合 username），但非常容易配错导致 `403 Forbidden`。建议今后全面拥抱 `KAGGLE_API_TOKEN` 环境变量的用法，省去了维护用户名的麻烦！
