module.exports = {
  apps: [
    {
      name: "fast-api-llm",
      script: "cmd.exe", //使用cmd运行
      args: "/c uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8504",
      env: {
        PYTHONUNBUFFERED: "1", // 实时日志输出
      },
      out_file: "./logs/out.log",
      error_file: "./logs/error.log",
      autorestart: true,
      instances: 1, // 单实例（Uvicorn自身多进程由--workers控制）
      max_memory_restart: "1G",
    },
  ],
};
