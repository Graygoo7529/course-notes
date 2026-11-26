# 课程项目与实验索引

本目录按课程专题组织全部课程实践项目与实验代码，同一专题下按作业或讲次顺序使用 `01_` / `01a_` 编号。各项目默认独立维护依赖、文档与测试；本地虚拟环境（`.venv`）、模型权重、数据文件与生成结果默认不纳入 Git。

## 深度学习应用（`deep_learning/`）

| 编号与目录 | 对应作业 | 内容简述 | 核心文档 |
| :--- | :--- | :--- | :--- |
| [01a_cnn_cartoon_portrait](deep_learning/01a_cnn_cartoon_portrait/README.md) | 作业 1（v1） | GrabCut 交互抠图与多尺度逐通道卷积特征漫画化 | [研究方案与设计](deep_learning/01a_cnn_cartoon_portrait/研究方案与设计.md) |
| [01b_cnn_cartoon_portrait_v2](deep_learning/01b_cnn_cartoon_portrait_v2/README.md) | 作业 1（v2） | 五段函数式管道重构与离线交互观察室 | [README](deep_learning/01b_cnn_cartoon_portrait_v2/README.md) |
| [02a_cnn_face_cartoon](deep_learning/02a_cnn_face_cartoon/README.md) | 作业 2（阶段一） | 四种预训练分类 CNN 特征引导人脸漫画化对比 | [技术路线](deep_learning/02a_cnn_face_cartoon/docs/技术路线.md) · [方法说明](deep_learning/02a_cnn_face_cartoon/docs/方法说明.md) |
| [02b_realtime_cartoon_camera](deep_learning/02b_realtime_cartoon_camera/README.md) | 作业 2（阶段二） | Tauri 2 + OpenVINO 桌面端与 Android ONNX Runtime 实时漫画相机 | [技术路线](deep_learning/02b_realtime_cartoon_camera/docs/技术路线.md) · [架构说明](deep_learning/02b_realtime_cartoon_camera/docs/架构说明.md) |
| [03_multiface_sketch_hat](deep_learning/03_multiface_sketch_hat/README.md) | 作业 3 | YOLOv5s-Face 多人素描生成与二维几何虚拟戴帽 Web 系统 | [技术路线](deep_learning/03_multiface_sketch_hat/docs/技术路线.md) · [方案评审](deep_learning/03_multiface_sketch_hat/docs/方案评审.md) |
| [04_unet_face_cartoon](deep_learning/04_unet_face_cartoon/README.md) | 作业 4 | CelebAMask-HQ UNet 语义分割引导的人脸漫画生成 | [技术路线](deep_learning/04_unet_face_cartoon/docs/技术路线.md) · [实现说明](deep_learning/04_unet_face_cartoon/docs/实现说明.md) |

## 大数据与舆论分析导论（`opinion_analysis/`）

| 编号与目录 | 对应讲次 | 内容简述 | 核心文档 |
| :--- | :--- | :--- | :--- |
| [01_iphone18pro](opinion_analysis/01_iphone18pro/README.md) | 第 3 讲（数据收集与存储） | iPhone 18 Pro 国际媒体 GDELT 布尔检索与小样本案例核读 | [课程作业报告](opinion_analysis/01_iphone18pro/课程作业报告.md) |
| [02_youtube_ev_sentiment](opinion_analysis/02_youtube_ev_sentiment/README.md) | 第 5–7 讲（主题与情感分析） | RoBERTa + BERTopic + BART-MNLI + DeBERTa ABSA 评论情感探索 | [技术路线](opinion_analysis/02_youtube_ev_sentiment/docs/技术路线.md) · [预处理说明](opinion_analysis/02_youtube_ev_sentiment/预处理说明.md) |
| [03_nepal_mudslide_factcheck](opinion_analysis/03_nepal_mudslide_factcheck/README.md) | 第 10 讲（谣言传播） | 2026 年中尼边境泥石流灾害事实核查与信息真空双模网络实证 | [事实核查与舆论演化报告](opinion_analysis/03_nepal_mudslide_factcheck/report/事实核查与舆论演化报告.md) |

## 机器学习（`machine_learning/`）

| 编号与目录 | 对应阶段 | 内容简述 | 核心文档 |
| :--- | :--- | :--- | :--- |
| [01_paper_survey_2026](machine_learning/01_paper_survey_2026/README.md) | 期中 / 期末选题 | 2026 年 ICML 五篇候选论文（NT-SSM、AIEC 等）方法梳理与复现评估 | [五篇论文选题调研](machine_learning/01_paper_survey_2026/五篇论文选题调研_2026.md) |
