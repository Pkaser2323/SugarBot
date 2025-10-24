# Gunicorn configuration file
import multiprocessing

# Worker 設置
workers = multiprocessing.cpu_count() * 2 + 1  # 建議的 worker 數量
worker_class = 'sync'  # 使用同步 worker
timeout = 300  # 增加 timeout 到 300 秒
keepalive = 5

# 日誌設置
accesslog = '-'  # 輸出到 stdout
errorlog = '-'   # 輸出到 stderr
loglevel = 'info'

# 進程命名
proc_name = 'diabetes-chatbot'

# 優雅的重啟/關閉
graceful_timeout = 120
max_requests = 1000
max_requests_jitter = 50

# 預加載應用
preload_app = True  # 在 fork worker 之前加載應用

def when_ready(server):
    # 當 Gunicorn 準備好接收請求時執行
    server.log.info("Gunicorn 已準備好接收請求！")
