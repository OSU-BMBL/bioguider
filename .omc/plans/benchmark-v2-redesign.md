# Benchmark v2 Redesign Plan

**Created:** 2026-04-29 (revised)
**Branch:** refactor/document-generation
**Supersedes:** `benchmark-prose-only-injection.md`, `next-stage-benchmark-execution.md`
**Sources:** Architect audit, GPT analysis, April 24 meeting transcript, E001/E002 results

---

## Benchmark Contract

这个benchmark测的是什么：

> **给定一个原本正确的bioinformatics tutorial，注入一组可恢复的、局部的、受控的corruption。
> 让模型在不破坏受保护区域的前提下修回来。
> 衡量"修回目标错误的能力"和"附带伤害"。**

### 四个定义

**1. 输入是什么**
- 一个合法的 .Rmd/.md 文档（baseline）
- 通过确定性注入器（`force_deterministic=True`）在prose区域注入N个已知错误
- 每个错误有 `(id, category, original_snippet, mutated_snippet)` 记录
- 所有模型/prompt看到完全相同的corrupted文档

**2. 受保护区域（Protected Regions）— 不能碰**
- Fenced code blocks（``` 围起来的内容，包括 ```{r} RMarkdown chunks）
- YAML frontmatter（`---` 围起来的头部）
- Section topology（不能删除/新增/重排章节）
- 代码块内部结构（chunk headers, 缩进）

**3. 什么算"修好了"**
- `original_snippet` 恢复到revised中 → fixed
- `mutated_snippet` 消失且替代文本语义正确 → fixed（允许rewrite）
- 当前 `_check_error_fixed()` 的per-category逻辑

**4. 什么算"附带伤害"（False Positive）**
两层：
- **Hard FP（safety violation）**: 改动落在受保护区域 — code fence内容变了、YAML变了、章节被删/新增
- **Soft FP（collateral damage）**: 改动落在非注入区域的prose中且产生有害变化（如改对了的基因名）

---

## What We Found Wrong

### Evaluator Bugs

| ID | Severity | Problem | Location |
|----|----------|---------|----------|
| **BUG-1** | P0 | `detect_semantic_fp=False` → precision ≡ 1.0 → F1 = recall | `test_single_file_stress.py:509` |
| **BUG-2** | P0 | `link` scorer: 文档里有任何合法链接就算修好 → 3385/3385 | `benchmark_metrics.py:473` |
| **BUG-3** | P1 | Headline F1 vs category breakdown fixed数不一致（40 vs 46） | AGGREGATE_RESULTS.json |
| **BUG-4** | P1 | `comment_typo` 注入匹配 `^#` → 打到markdown标题 | `llm_injector.py:1115` |

### Prompt/实验设计问题

| ID | Severity | Problem |
|----|----------|---------|
| **PROMPT-1** | P0 | BioGuider prompt过度保守（"only demonstrably wrong"） + Generic无约束 → FP=0时Generic天然赢 |
| **PROMPT-2** | P1 | Generic prompt有4条评价维度（含Completeness），但injector不注入"缺失章节"→ 鼓励"增强"而非"修错" |
| **EXP-1** | P1 | skill test默认用 `next(iter(MODELS))` = gpt-5.4，不是E001最优的gpt-4o |

---

## 修复计划：两条轨道

### Track A: 修Evaluator + 重算（不改prompt，不重跑LLM）

用修好的evaluator重新计算E001/E002已有的结果。确认旧结论哪里是evaluator伪影。

#### A.1: Protected Region Safety Metric
- **文件**: `benchmark_metrics.py` 新方法
- **逻辑**: `check_protected_regions(baseline, revised) -> dict`
  - 比较baseline和revised的fenced code blocks是否byte-identical
  - 比较YAML frontmatter是否保留
  - 检查section headers（`^#{1,6} `）数量和内容是否匹配
- **输出**: `{ "code_fence_violations": int, "yaml_violations": int, "section_violations": int }`
- 这是Hard FP的来源

#### A.2: 确定性FP检测器（替代逐行diff）
- **文件**: `benchmark_metrics.py` 新方法
- **逻辑**: `count_collateral_damage(baseline, revised, error_list) -> int`
  - 不是"任何unmatched diff都算FP"
  - 而是: 找到revised相对baseline的所有改动位置
  - 排除: 已知injected error的original/mutated snippet位置（允许rewrite）
  - 排除: 受保护区域（那些单独计入Hard FP）
  - 剩余的prose改动 → 检查是否harmful（对比baseline原文是否被改坏）
  - 简单版: 只计数；精确版: 采样检查
- **集成**: `BenchmarkResult.false_positives = hard_fp + soft_fp`

#### A.3: `link` scorer修复
- **文件**: `benchmark_metrics.py:473`
- **修改**: 检查被注入破坏的那条原始链接是否被还原，而不是"存在任何合法链接"

#### A.4: 查清Headline vs Category不一致
- 调查errors_fixed=40 vs sum(category.fixed)=46的差异来源
- 修复或文档说明

#### A.5: 重算E001/E002
- 用修好的evaluator重新score已有的baseline/corrupted/revised文件
- **不重跑LLM** — 直接读取outputs目录里的.fixed.Rmd文件重算
- 产出: E001-v2 rescore CSV, E002-v2 rescore CSV
- 确认: 模型排序是否变化、precision是否≠1.0

### Track B: Prompt Ablation（新实验）

在Track A验证通过后，用修好的evaluator跑新的prompt比较。

#### B.1: 定义Prompt变体

**Skill 1 — BioGuider (v3)**

保留邵红prompt的结构化价值，去掉过度保守约束，加入评价维度（与benchmark contract对齐）：

```
You are "BioGuider," fixing documentation for biomedical software.

GROUND TRUTH
- Code blocks (``` fences) are the AUTHORITY. If prose contradicts code
  (package version, test name, marker gene, parameter value), fix the
  PROSE to match the CODE.

EVALUATION DIMENSIONS (fix errors in all categories)
1. Scientific accuracy: gene names, species, statistical tests, parameters,
   accession IDs must be correct and consistent with code blocks
2. Markdown formatting: headers, lists, links, inline code, tables,
   image syntax must follow proper markdown
3. Prose-code consistency: prose descriptions must agree with adjacent
   code block contents (versions, function names, parameter values)
4. Structure: section titles, YAML frontmatter must be correct

HOW TO FIX (BioGuider methodology)
- Scan the entire document systematically, dimension by dimension
- Use code blocks as the source of truth for factual claims
- Fix typos, broken links, wrong gene names, incorrect numbers
- Restore proper markdown formatting
- Do NOT add new content or remove existing sections
- Do NOT modify text inside ``` fences
- Output the COMPLETE fixed document as markdown

CORRUPTED DOCUMENT TO FIX:
```

**Skill 2 — Generic (一句话)**

一个普通用户会写的prompt，零领域知识：

```
Fix all errors in this document and output the corrected version:
```

（就是现有的 `SIMPLE_PROMPT`，不泄露任何评价标准）

**对比的核心差异：**

| | Skill 1 (BioGuider) | Skill 2 (一句话) |
|---|---|---|
| 评价维度 | ✅ 4个维度 | ❌ 不知道 |
| 修改方法论 | ✅ 代码作为权威、系统扫描 | ❌ 盲修 |
| Protected regions | ✅ 明确说不碰代码 | ❌ 没约束 |
| 长度 | ~150 tokens | ~10 tokens |

这才是在测BioGuider的真正价值：**"告诉AI该从哪些维度修、怎么修" vs "随便改"**。

#### B.2: 显式指定模型
- `test_skill_comparison`: `model_name="gpt-4o"`（E001 winner）
- `test_skill_matrix`: 同上

#### B.3: 运行实验

| 实验 | 内容 | 预计时间 |
|------|------|---------|
| E002-v2 | Skill comparison, 1 file, gpt-4o, BioGuider v3 vs 一句话 | 5min |
| E003 | Skill matrix, 5 files × 3 levels × 2 prompts, gpt-4o | 30min |

---

## 不改的东西

- 评分体系四大类分数：冻结
- 单Repo选模型：论文注明局限性
- 完整pipeline对比：太花时间，prompt对比是会议决定
- `full_document.txt` 不直接用于benchmark — 那是"基于evaluation report做generation"的prompt，不是corruption-repair的prompt

## 验收标准

验收的是**benchmark本身的质量**，不是BioGuider是否赢：

1. **判分一致**: Headline F1 和 category breakdown fixed数一致
2. **FP生效**: 至少在generic prompt下 precision ≠ 1.0
3. **Protected region**: 有safety violation计数且非全0
4. **link类别**: fix_rate < 1.0（有区分力）
5. **可解释**: 每个模型/prompt的胜负可以归因到具体的category差异
6. **Smoke test通过**: minimal test产出合理的F1值（非全0非全1）

**研究假设**（不是验收条件）：
- BioGuider F1 ≥ Generic F1
- BioGuider duration < Generic duration
- BioGuider protected region violations < Generic violations

## 执行顺序

底层改动大，全部重跑确保数据一致。

### Step 1: 修Evaluator代码（~1.5h）
可并行：
- A.1 Protected region metric（30min）
- A.2 确定性FP检测器（45min，依赖A.1）
- A.3 link scorer（15min）
- A.4 Headline/Category调查（15min）

### Step 2: 改Prompt + 实验参数（~15min）
- B.1 写入BioGuider v3 prompt + Generic改为一句话（SIMPLE_PROMPT）
- B.2 skill test锁定 `model_name="gpt-4o"`

### Step 3: Smoke Test验证（~5min）
- 跑 `test_single_file_stress_minimal` — 1个文件、1个模型、1个级别
- 确认：precision ≠ 1.0，link fix_rate < 1.0，protected region有计数
- **不通过则不进入Step 4**

### Step 4: 重跑E001 — 模型选择（~12h LLM）
- `test_multi_file_full_matrix` — 10文件×9级别×5模型 = 450次
- 全部用修好的evaluator + BioGuider v3 prompt
- 产出新的AGGREGATE_TABLE.csv + heatmap

### Step 5: 跑E002/E003 — Skill比较（~35min LLM）
- E002-v2: 1文件, gpt-4o, BioGuider v3 vs 一句话
- E003: 5文件×3级别×2 prompts, gpt-4o
- 产出SKILL_COMPARISON.csv + SKILL_MATRIX_TABLE.csv

### Step 6: 更新实验日志
- 更新 `docs/EXPERIMENT_LOG.md`
- 标注旧E001/E002为deprecated
- 记录新结果

### Step 7: Publication Figures（R ggplot2）

用R ggplot2制作论文级别的图，基于Step 4/5的结果CSV。

#### Fig 1: Model Selection Heatmap
- 数据源: `AGGREGATE_TABLE.csv`
- 行=模型（5个），列=错误级别（9个），色块=F1 scorable
- 附带注释：每格显示F1值（2位小数）
- 配色：RdYlGn渐变，0→红，1→绿
- 会议要求："一张图就够了，一个heatmap搞定"

#### Fig 2: CONTENT vs HYGIENE分面
- 同样的heatmap但分成两个panel
- Panel A: F1 content（科学准确性）
- Panel B: F1 hygiene（格式/排版）
- 展示"所有模型CONTENT都~0.92，差异在HYGIENE"

#### Fig 3: Skill Comparison
- 数据源: `SKILL_MATRIX_TABLE.csv`
- Grouped bar chart: x=错误级别, 两组bar=BioGuider vs Generic
- y=F1 scorable, error bars=跨文件标准差
- 副图/注释: duration对比（BioGuider应该更快）

#### Fig 4: F1 Degradation Curve
- Line plot: x=错误数量, y=F1 scorable
- 一条线per模型，展示随错误增多F1如何衰减
- 所有模型从~0.8衰减到~0.65，排序稳定

#### 技术要求
- R脚本放在 `scripts/benchmark_figures.R`
- 读取 `outputs/` 下的CSV
- 输出PDF + PNG（300 dpi）到 `outputs/figures/`
- ggplot2 + cowplot/patchwork组合多panel
- 字体: Arial, 8-10pt
- 配色: viridis或自定义学术配色
- 图宽: single column (85mm) 或 double column (170mm)

### Step 8: Benchmark方法学文档

写一个清晰的方法学描述文档，讲清楚benchmark这一块的来龙去脉。
面向论文Methods section + Supplementary，不是技术文档。

#### 内容大纲

**1. 为什么需要这个Benchmark**
- 现有LLM可以直接修文档，但怎么知道它改得对不对？
- 需要一个受控的实验：注入已知错误 → 让模型修 → 量化修复能力
- 类似于软件工程的mutation testing思想

**2. 错误注入设计**
- 36个错误类别，分CONTENT（科学准确性22个）和HYGIENE（格式11个）
- 为什么这样分：CONTENT是"改错了会误导研究者"，HYGIENE是"格式不好但不影响科学理解"
- Prose-only注入：只在文本区域注入，代码块作为ground truth不动
- 确定性注入：同一篇文档注入结果完全一致，确保模型间公平比较

**3. 为什么选这些模型**
- 覆盖闭源（GPT-4o, GPT-5.4, Kimi）和开源（GLM-5, GPT-OSS）
- 都通过同一个LiteLLM proxy路由，消除基础设施差异
- 选择标准：可用性、成本、领域相关性

**4. 评价指标**
- F1 = precision × recall 的调和平均
- Precision：模型做的改动里多少是真的在修错误（vs 乱改）
- Recall：注入的错误里多少被修好了
- Protected region violations：模型有没有碰不该碰的地方（代码块、YAML）
- 为什么不只看accuracy/fix_rate：因为一个什么都改的模型recall高但precision低

**5. Skill对比设计**
- BioGuider prompt：结构化的修改指导（评价维度 + 方法论 + 代码作为权威）
- Generic prompt：一句话"Fix all errors"
- 为什么这样设计：测的是"领域知识指导"的价值，不是"AI能不能修文档"
- 两者用同一个模型（gpt-4o），同一套注入，同一个evaluator

#### 文档位置
- `docs/BENCHMARK_METHODS.md` — 完整版（Supplementary材料）
- 论文Methods section从这里提炼

#### 风格
- 不要技术细节（不提文件名、函数名、行号）
- 写给reviewer看：为什么这样设计、有什么trade-off、局限性是什么
- 中英双语版本（中文先写理清思路，英文用于论文）

## 工作量估计

| Step | 工作量 | 类型 |
|------|--------|------|
| Step 1: 修Evaluator | ~1.5h | 代码 |
| Step 2: 改Prompt | ~15min | 代码 |
| Step 3: Smoke Test | ~5min | LLM |
| Step 4: 重跑E001 | ~12h | LLM（可后台） |
| Step 5: 跑E002/E003 | ~35min | LLM |
| Step 6: 更新日志 | ~15min | 文档 |
| Step 7: Publication Figures | ~2h | R代码 |
| Step 8: Benchmark方法学文档 | ~1.5h | 写作 |
| **合计** | **~5.5h人工 + ~13h LLM** | |

## Changelog

| Date | Change |
|------|--------|
| 2026-04-29 | Created. |
| 2026-04-29 | Revised: 加benchmark contract, 拆v2a/v2b, Generic改一句话, FP改两层定义, 验收条件≠研究假设 |
