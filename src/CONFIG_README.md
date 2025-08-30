# 配置文件使用说明

## 概述
`agent.py` 现在支持通过 JSON 格式的配置文件来配置 Azure AI Inference 模型参数和工作流设置。

## 配置文件格式

创建一个 JSON 格式的配置文件，例如 `config.json`：

```json
{
  "azure_ai_inference": {
    "endpoint": "https://your-endpoint.cognitiveservices.azure.com/openai/deployments/your-model/",
    "api_key": "your-api-key-here",
    "model_name": "gpt-4.1-mini",
    "temperature": 0.3,
    "top_p": 1.0,
    "max_tokens": 1000,
    "timeout": 20,
    "max_retries": 3
  },
  "workflow": {
    "total_timeout": 120,
    "enable_mcp": false,
    "mcp_servers": {
      "fetch-mcp": {
        "command": "node",
        "args": ["C:\\path\\to\\fetch-mcp\\dist\\index.js"],
        "env": {}
      }
    }
  }
}
```

## 配置参数说明

### azure_ai_inference 部分
- `endpoint`: Azure AI Inference 端点 URL
- `api_key`: API 密钥 (可选，如果未设置将使用 GITHUB_TOKEN 环境变量)
- `model_name`: 模型名称 (默认: "gpt-4.1-mini")
- `temperature`: 温度参数 (默认: 0.3)
- `top_p`: Top-p 参数 (默认: 1.0)
- `max_tokens`: 最大令牌数 (默认: 1000)
- `timeout`: 请求超时时间，秒 (默认: 20)
- `max_retries`: 最大重试次数 (默认: 3)

### workflow 部分
- `total_timeout`: 整个工作流的总超时时间，秒 (默认: 120)
- `enable_mcp`: 是否启用 MCP 服务器 (默认: false)
- `mcp_servers`: MCP 服务器配置

## 使用方法

### 1. 使用配置文件运行
```bash
python src/agent.py --config config.json
```

### 2. 使用默认配置运行
```bash
python src/agent.py
```

程序会按以下顺序查找配置文件：
1. 项目根目录下的 `config.json`
2. `src` 目录下的 `config.json`
3. 当前工作目录下的 `config.json`

### 3. 命令行参数
- `--config, -c`: 指定配置文件路径
- `--message, -m`: 指定要分析的消息内容 (暂未实现)
- `--help, -h`: 显示帮助信息

## 配置优先级

参数的优先级从高到低：
1. 代码中直接传入的 kwargs 参数
2. 配置文件中的参数
3. 默认值

## API 密钥配置

有两种方式配置 API 密钥：

1. **在配置文件中设置** (推荐)：
   ```json
   {
     "azure_ai_inference": {
       "api_key": "your-api-key-here"
     }
   }
   ```

2. **使用环境变量**：
   ```bash
   set GITHUB_TOKEN=your-api-key-here
   ```

## 示例

### 基本配置示例
```json
{
  "azure_ai_inference": {
    "endpoint": "https://zhenjiao-83-resource.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini/",
    "api_key": "your-key-here",
    "temperature": 0.2,
    "max_tokens": 2000
  }
}
```

### 完整配置示例
```json
{
  "azure_ai_inference": {
    "endpoint": "https://zhenjiao-83-resource.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini/",
    "api_key": "your-key-here",
    "model_name": "gpt-4.1-mini",
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 1500,
    "timeout": 30,
    "max_retries": 5
  },
  "workflow": {
    "total_timeout": 180,
    "enable_mcp": true,
    "mcp_servers": {
      "fetch-mcp": {
        "command": "node",
        "args": ["./fetch-mcp/dist/index.js"],
        "env": {
          "NODE_ENV": "production"
        }
      }
    }
  }
}
```

## 注意事项

1. 确保配置文件为有效的 JSON 格式
2. API 密钥应该保密，不要提交到版本控制系统
3. 如果同时设置了配置文件和环境变量，配置文件中的设置会优先使用
4. 路径应使用正确的操作系统格式（Windows 使用反斜杠或双反斜杠）
