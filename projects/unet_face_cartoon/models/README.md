# 本地权重

权重、下载记录和检测缓存不进入 Git。来源与命令见 `../docs/模型准备.md`。
加载前检查 SHA256，PyTorch 使用 `weights_only=True` 和严格参数匹配。
UNet 位于 `unet/model.pth`，生成器位于 `photo2cartoon/photo2cartoon_weights.pt`。
