# docker_pull
The script makes it possible to download a docker-image without docker

Required Python 3.7+

## Use
```bash
> git clone https://github.com/myback/docker_pull.git
> cd docker_pull
> chmod +x docker_pull.py
> ./docker_pull.py -h
usage: docker_pull.py [-h] [--save-cache] [--verbose] [--user USER] [--platform PLATFORM] [--password PASSWORD | -P]
                      image [image ...]

positional arguments:
  image

optional arguments:
  -h, --help                        show this help message and exit
  --save-cache, -s                  Do not delete the temp folder after downloading the image
  --verbose, -v                     Enable verbose output
  --user USER, -u USER              Registry login
  --platform PLATFORM               Set platform if server is multi-platform capable
  --password PASSWORD, -p PASSWORD  Registry password
  -P                                Registry password (interactive)
> ./docker_pull.py alpine:3.10
3.10: Pulling from library/alpine
21c83c524219: Pull complete
Digest: sha256:a143f3ba578f79e2c7b3022c488e6e12a35836cd4a6eb9e363d7f3a07d848590
> docker pull alpine:3.10
> docker save alpine:3.10 -o alpine_3.10.tar
> sha256sum *.tar
d59b494721c87e7536ad6b68d9066b82b55b9697d89239adb56a6ba2878a042d  alpine_3.10.tar
d59b494721c87e7536ad6b68d9066b82b55b9697d89239adb56a6ba2878a042d  library_alpine_3.10.tar
```
Fetch multiple images
```bash
> ./docker_pull.py alpine:3.10 ubuntu:18.04 bitnami/redis:5.0
```
Verbose
```bash
> ./docker_pull.py -v alpine  # Same as alpine:latest
```
Fetch image from private registry
```bash
> ./docker_pull.py --user username --password 'P@$$w0rd' private-registry.mydomain.com/my_image:1.2.3
```

## LLM 流式输出性能评测工具
新增 `llm_stream_bench.py`，用于对多个 OpenAI 兼容接口模型进行流式性能评测。

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行
```bash
python llm_stream_bench.py --config benchmark.yaml
```

支持参数：
- `--config`：YAML 配置路径（必填）
- `--output-dir`：输出目录（默认 `benchmark_reports`）
- `--output-prefix`：输出文件名前缀（默认 `llm_stream_benchmark`）

运行后会输出两份报告：

说明：评测流程已取消 warmup 阶段，所有执行均计入统计。
- JSON：结构化明细与聚合结果
- Markdown：按并发分组的可读对比报告
