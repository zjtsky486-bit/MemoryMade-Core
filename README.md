# MemoryMade Core · 造忆核心

从 MemoryMade 本地网站中提取的识别、图片转三维、网格检查和 Blender 导出代码。没有网页 UI、入场动画、海报、展示模型或私人上传文件。

## 保留的核心

- Qwen 视觉描述、YOLO-World / Grounding DINO 定位、BiRefNet 前景分割；识别缓存和细节观察。
- TripoSR、Hunyuan3D-2 mini、Hunyuan3D-2.1 Shape / PBR 的本地推理适配。
- 人物前景与几何检查、悬浮部件检查、质量配置、材质识别、VGGT 场景适配。
- Blender 精修与 GLB / STL / OBJ 导出；浮雕、结构先验、雕刻、录音识别和 NFC 编码底层模块。
- 磁盘警戒与任务监控。保留现有质量配置，不以降低分辨率换速度。

## 这是什么

这是源代码仓库，GitHub 页面本身不会运行 GPU 生成任务。仓库不包含模型权重、Blender、CUDA、独立 Python 环境或第三方项目源码。这些资源需要在运行机器上按相应项目的许可另行准备。

当前适配首先针对原有 Windows / NVIDIA CUDA 布局。macOS、Linux 和云端完整部署尚未验证，不能直接把本仓库当成已部署的在线服务。

## 在已有本地资源上使用

主进程依赖见 `requirements.txt`；模型环境必须分开，分别参考 `requirements-vision.txt`、`requirements-reconstruction.txt` 及其他环境清单。CUDA Torch 的 `+cu128` 包需使用 PyTorch 官方 CUDA 12.8 源安装，不能把不同 Transformers 版本装进同一环境。

已有资源布局：

```text
RESOURCE_ROOT/
  assets/models/          # 另行准备的模型权重与清单
  environments/vision/Scripts/python.exe
  environments/reconstruction/Scripts/python.exe
  environments/hunyuan21/Scripts/python.exe
  environments/human/Scripts/python.exe
  tools/                 # Blender、TripoSR、Hunyuan3D、llama 等
```

先检查模型就绪状态，再识别和生成：

```powershell
python core_cli.py --resource-root D:\1223456789 status
python core_cli.py --resource-root D:\1223456789 recognize C:\images\object.jpg
python core_cli.py --resource-root D:\1223456789 --asset-root D:\1223456789\MEMORYMADE-Studio\assets reconstruct C:\images\object.jpg --backend hunyuan21 --width-mm 100
```

可通过 `MEMORYMADE_BLENDER` 指定 Blender 可执行文件。输出默认在本仓库 `runs/data/exports`，日志在 `runs/logs`；不将作品上传至 GitHub。原项目资源目录下的磁盘策略继续生效，触及警戒线会阻止任务。

CLI 单图入口已提供；多视角、材质、人物检查、音频与 NFC 可调用 `memorymade` 包对应接口。缺少资源或推理失败将报错，不使用展示模型冒充生成结果。

## 精度边界

单图无法确认物体背面、底面和遮挡处的真实结构；AI 补全属于推测。人物细绳、飘带和悬空部件可能被几何检查拦截，需要更多视角或人工修模。VGGT 的可见场景点云不等于完整房屋实体。结构先验和浮雕也不能替代高精度完整三维重建。

## 上游项目与许可

- [腾讯 Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)
- [腾讯 Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Ultralytics YOLO-World](https://github.com/ultralytics/ultralytics)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- [VGGT](https://github.com/facebookresearch/vggt)
- [Blender](https://www.blender.org/)

仅保留应用适配代码，不重新分发上游模型或软件；使用、商用和再分发必须分别遵守上游代码与模型许可。本仓库暂未授予统一开源许可。
