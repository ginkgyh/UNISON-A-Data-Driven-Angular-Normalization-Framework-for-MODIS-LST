# UNISON MODIS 地表温度角度归一化

本仓库提供三个使用入口：下载训练好的模型进行离线调用、使用作者部署的在线服务，以及了解模型构建方法并训练新的 `.pkl` 模型。建议将本仓库主页作为论文中的统一公开链接。

[English](README.md) · [输入说明](docs/input_contract.md) · [模型下载](models/) · [训练方法](training/README.md)

## 一 离线模型调用

仓库提供六个原始模型包，分别对应 Terra/Aqua 与 Bare、MLVeg、MHVeg 三类地表的组合。每个模型目录还提供字段与单位说明、SHA-256 校验值和输入 CSV 模板。离线调用无需云端账号或在线端点，也无需重新训练。

- Bare：裸地和荒漠。
- MLVeg：研究中的中低纬植被组。
- MHVeg：研究中的高纬植被组。

在 Python 3.13 环境中安装依赖后，运行：

```bash
python -m pip install -r requirements.txt
python offline/predict.py --model models/Bare_Terra/model_bundle.pkl --input examples/Bare_Terra.csv --output result.csv --lst-column modisLST --lst-unit celsius --output-kelvin
```

`examples/` 仅提供合成的格式示例，不代表真实遥感观测或精度验证结果。正式应用时，将其替换为按研究方法准备的输入数据。详细操作见[离线使用说明](offline/README.md)。

**本次发布模型的三个温度输入 `modisLST`、`DSAT`、`DSOIL` 使用摄氏度。** 如果数据为 Kelvin，须在输入模型前减去 273.15。不同模型的字段及顺序不同，代码按包内 `feature_names` 选取，不可统一硬编码。模型包已包含拟合好的输入和目标缩放器，用户不应重复缩放。

```text
预测温差 = 近天底 LST − 观测 LST
归一化 LST（℃）= 观测 LST（℃）+ 预测温差
归一化 LST（K）= 归一化 LST（℃）+ 273.15
```

温差以 K 表示时数值与 ℃ 相同。缺失或无效输入保留为空值，不随意补零。代码保留模型原始预测，包括输入角度为零时的输出，不额外强制校正量为零。

## 二 在线模型调用

**当前状态：待部署，暂未提供可用在线调用链接。** 请先使用离线方式。

在线服务由作者自行部署和维护。部署完成后，在[在线入口说明](online/README.md)及[端点配置](online/endpoint_registry.json)中公布可用链接和服务状态。当前预留配置不会连接到旧的或未经核实的端点。

通过 GEE 的 `ee.Model.fromVertexAi` 直接调用时，使用者需要获得托管项目的 IAM 权限，端点也必须保持可用；还需满足相应云项目计费、Earth Engine 使用资格及配额条件。仅公开脚本网址不能自动授予预测权限。因此，使用条件不宜笼统写成“所有人都要购买 Agent Platform 付费功能”。详细说明及 Google 官方文档见[在线使用条件](online/README.md)。

## 三 模型构建与导出

[训练说明](training/README.md)简述论文中的数据质控、多角度条件均温、近天底参考差值、LOESS 平滑及 LightGBM 学习过程，并提供从已准备好的带标签 CSV 训练并导出模型包的脚本。

训练辅助脚本不会自动完成全部上游 GEE 数据采样和标签生成，也不宣称能逐字节复现现有六个模型。原有训练数据压缩包继续保留，但其中存在重复文件，且与当前训练源代码的字段配置有差异；使用前请阅读[旧数据说明](docs/legacy_data.md)。

## 应用范围与引用链接

验证范围为论文纳入自然地表上的晴空夏季白天 Terra/Aqua MODIS 数据。其他季节、夜间、其他传感器和未纳入地类需要另行验证。Landsat/VIIRS 用于验证参考构建，不是日常归一化的必需输入。

统一公开入口：[UNISON 仓库主页](https://github.com/ginkgyh/UNISON-A-Data-Driven-Angular-Normalization-Framework-for-MODIS-LST)。原有 GEE 链接保存在[历史材料说明](docs/legacy_data.md)，其存在不表示当前在线端点已部署或已授权访问。
