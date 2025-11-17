# llm-fastapi

bunch of fastapi functions

## Development

start service: `uv run uvicorn src.app.main:app --reload`

### 防止 SQL 注入攻击

- 始终使用参数化查询：对所有的数据值使用 %s 占位符, 而不是直接在 SQL 语句中拼接字符串, 禁止使用动态表名、字段名, aiomysql 库会自动转义。

## Deployment

### Uvicorn

`uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --workers 4`

### PM2

`pm2 start ecosystem.config.js`
