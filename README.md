# 项目说明

本项目的 Python 环境为 **Python 3.10**。

## 虚拟环境

建议使用虚拟环境来运行本项目。所有的依赖都已定义在 `requirements.txt` 文件中，可以通过以下命令安装：

```bash
pip install -r requirements.txt
```

## 运行项目

可以通过以下命令执行项目：

```bash
python3 main.py
```

所有的参数都通过环境变量进行传递，以下是需要配置的环境变量：

```bash
export CARBON_RUN_MODE="year" # 可选值为 "year" 或 "day"， 分别代表处理一年的数据或者当天的数据
export CARBON_DATA_YEAR="2024" # 当 CARBON_RUN_MODE 为 "year" 时，需要配置该环境变量，指定处理的年份
 # 数据处理完成之后得到的 csv 文件会上传到 OSS，需要配置 OSS 的相关信息。
export CARBON_OSS_ACCESS_KEY_ID="<TBA>" # 阿里云 OSS 的 Access Key ID
export CARBON_OSS_ACCESS_KEY_SECRET="<TBA>" # 阿里云 OSS 的 Access Key Secret
export CARBON_OSS_BUCKET_NAME="<TBA>" # OSS 的 Bucket 名称
export CARBON_OSS_ENDPOINT="<TBA>" # OSS 的 Endpoint
export CARBON_OSS_BASE_PATH="<TBA>" # OSS 中 Bucket 下的基础路径，会在这个路径下创建一个以日期命名的文件夹，存放处理完成的 CSV 数据文件
```

在配置好以上环境变量后，即可运行项目。
