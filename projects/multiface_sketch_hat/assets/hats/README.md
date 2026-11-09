# 帽子库

初版计划准备 3 款可明确区分的帽子。优先制作原创二维素材，不需要下载帽子识别网络。

每款包含同一画布尺寸的 `front.png`、可选的 `back.png` 和 `metadata.json`。两张图使用透明背景 RGBA；`back` 在人像后方，`front` 在人像前方。当前尚未制作素材。

元数据约定：

| 字段 | 含义 |
| --- | --- |
| `id` / `name` | 稳定标识和显示名称 |
| `canvasSize` | 素材宽高，单位 px |
| `openingCenter` | 帽口中心，素材坐标 |
| `openingLeft` / `openingRight` | 帽口左右锚点，素材坐标 |
| `widthRatio` | 目标帽口宽 / 脸宽 |
| `heightOffsetRatio` | 双眼中点沿脸朝上方向的位移 / 脸宽 |
| `views` | 可用正面、侧面素材及适用范围 |
| `source` / `license` | 原创记录或来源和许可 |

先支持正面、缩放、平移和画面内旋转；帽口参数逐款校准。可按姿态支持范围自动选择默认帽子，保留逐人手动覆盖。仅用五点时不能假装已获得精确 yaw/pitch。
