# 委派给 Grok Bot 的 X（Twitter）数据采集任务单

> 使用说明：可以直接将下方代码块中的提示词复制发送给 Grok。如果单次输出长度受限，可以让它**先输出第 1 批（谣言与早期扩散样本）**，再发一句“继续按相同 JSON 格式输出第 2 批（官方、科学与事实核查样本）”。拿到结果后直接粘贴保存为 `data/raw/grok_x_posts.json`（或发给我帮你整理清洗）。

---

## 复制以下 Prompt 发送给 Grok

```markdown
你现在是一名严谨的开源情报（OSINT）与计算传播学数据采集助手。我正在研究 **2026年8月26日中尼边境（尼泊尔 Rasuwa / Langtang Lirung — 中国西藏吉隆口岸 Gyirong Port / Lhende River）特大山洪泥石流灾害** 发生后 96 小时内（`2026-08-26 00:00 UTC` 至 `2026-08-30 00:00 UTC`）X（Twitter）平台上的舆论传播与事实核查演化。

请你检索 X 平台上的真实历史帖子，分两批为我整理 **40–60 条具有代表性的真实帖子记录**，并严格遵守以下**反幻觉与真实性约束**：
1. **严禁编造或猜测任何账号名、推文链接、数字或文本**。如果某字段无法确认，请填 `null`，绝不要凭印象补全。
2. 优先保留原帖的**逐字英文/原文内容（verbatim text）**，不要自行改写摘要。
3. 时间请尽量精确到分钟（标明时区，统一换算为 UTC 格式 `YYYY-MM-DDTHH:MM:SSZ`；若只能确认日期则注明）。

### 检索范围与两批配比要求

**第 1 批：早期信息真空期内的谣言、误读与跟风放大帖（约 25–35 条，重点覆盖 2026-08-26 至 08-27）**
- 检索关键词组合示例：
  - `("Nepal" OR "Gyirong" OR "Rasuwa" OR "Tibet") ("dam collapse" OR "Chinese dam" OR "dam burst") since:2026-08-26 until:2026-08-30`
  - `("Nepal" OR "Tibet") ("291" OR "105 Indians" OR "38 US") flood since:2026-08-26 until:2026-08-30`
  - `("Nepal" OR "Tibet" OR "Xizang") ("earthquake" OR "M4.4" OR "M5.2") flood since:2026-08-26 until:2026-08-29`
- 重点覆盖：
  1. 首发及核心推动“西藏/中国大坝溃坝论”的账号（例如 `@RealBababanaras`、`@BlacksmithIndie` 等在 8 月 26 日连发的帖子）；
  2. 引用转发（Quote）、回复跟进，或**虽然没有直接转发但搬运了同一段口岸洪水视频、复用了相同数字槽位（如 291 名外国游客、105 名印度人、38 名美国人等）的“伪原创/跟风搬运号”**（这对我们构建“显式转发链 + 隐式槽位共现网络”非常关键！）；
  3. 同期传播“西藏地震引发洪水”或盗用旧视频（如 2025 印度洪灾视频）、AI 合成视频的代表性高互动帖。

**第 2 批：权威通报、科学解释与事实核查澄清帖（约 15–25 条，覆盖 2026-08-26 至 08-30）**
- 检索关键词组合示例：
  - `("Nepal" OR "Gyirong" OR "Rasuwa" OR "Lhende") ("glacier" OR "glacial" OR "ice-rock" OR "avalanche" OR "Langtang") since:2026-08-26 until:2026-08-31`
  - `("Nepal" OR "Gyirong" OR "Rasuwa") ("USGS" OR "ICIMOD" OR "NDRRMA" OR "Tourism Board" OR "fact check" OR "debunk" OR "false") since:2026-08-26 until:2026-09-05`
- 重点覆盖：尼泊尔官方/救援机构（NDRRMA、NTB）、地质与山地研究学者（如解释 USGS 震动波实为滑坡/冰岩崩塌、ICIMOD 遥感分析）、主流通讯社/媒体记者，以及指出“大坝溃坝为假/旧视频移花接木”的 OSINT 与事实核查帖。

### 输出格式要求（严格输出 JSON 数组）

请输出合法的 JSON 数组，每条记录包含以下字段：

```json
[
  {
    "record_id": "X01",
    "batch": "rumor_or_early",
    "post_url": "https://x.com/handle/status/1234567890",
    "author_handle": "@handle",
    "author_category": "influencer | media | official | scientist_osint | general_user | suspected_amplifier",
    "created_at_utc": "2026-08-26T10:41:00Z",
    "post_type": "original | quote | reply | repost_with_media",
    "referenced_handle": "@parent_handle_or_null",
    "text_verbatim": "原帖完整原文...",
    "has_video_or_image": true,
    "media_note": "简述所配视频/图片内容，如：吉隆口岸联检楼前泥石流监控（去水印）/ 2025印度洪灾旧视频 / 卫星图等",
    "claimed_cause": "chinese_dam_collapse | earthquake | glacier_rock_collapse | unattributed_report | debunk_clarification",
    "reused_slots": ["Chinese dam", "Tibet", "291 foreign tourists", "105 Indians", "38 US citizens"],
    "metrics": {
      "likes": 0,
      "reposts": 0,
      "replies": 0,
      "views": null
    }
  }
]
```

如果单次回复放不下全部记录，请先输出 **第 1 批（25–30 条）** 的完整 JSON，并在末尾提示我输入“继续”以输出第 2 批。
```
