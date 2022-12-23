凹点检测论文对应代码
#### 使用conda 创建环境
```shell
conda create --name picture_deal python=3.7
```

#### 环境安装
```shell
conda activate picture_deal
pip install -r requirements.txt
```

#### 代码运行
```shell
# 运行凹点文章中提出的凹点检测匹配算法
python qie_tu/main.py basic

# 简单凹点匹配
python qie_tu/main.py simple

# 分水岭
python qie_tu/main.py watershed
```


