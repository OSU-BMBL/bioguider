# Benchmark v2 Redesign Plan

**Created:** 2026-04-29
**Branch:** refactor/document-generation
**Supersedes:** `benchmark-prose-only-injection.md`, `next-stage-benchmark-execution.md`
**Sources:** Architect audit, GPT analysis, April 24 meeting transcript, E001/E002 results

---

## What We Found Wrong

### Evaluator Bugs (must fix before any re-run)

| ID | Severity | Problem | Evidence |
|----|----------|---------|----------|
| **BUG-1** | P0 | `detect_semantic_fp=False` hardcoded → precision ≡ 1.0 → F1 = recall | `test_single_file_stress.py:509`; AGGREGATE_TABLE.csv 450行 precision全是1.0 |
| **BUG-2** | P0 | `link` scorer判定逻辑：文档里有任何合法链接就算修好 → 3385/3385 = 1.0 | `benchmark_metrics.py:473`; aggregate里link fix_rate全是100% |
| **BUG-3** | P1 | Headline F1 vs category breakdown不一致（40 vs 46 fixed） | AGGREGATE_RESULTS.json第一条记录 |
| **BUG-4** | P1 | `comment_typo` 注入器匹配 `^#` 开头行 → 打到markdown标题不是代码注释 | `llm_injector.py:1115`；manifest里实际例子是标题里的 Perform 被改坏 |

### Prompt设计问题 (影响Skill对比)

| ID | Severity | Problem | Evidence |
|----|----------|---------|----------|
| **PROMPT-1** | P0 | BioGuider prompt过度保守（"only demonstrably wrong", "do not rewrite"），Generic无约束 → 在FP=0环境下Generic天然占便宜 | Architect审查；E002 generic F1=0.970 vs bioguider F1=0.911 |
| **PROMPT-2** | P1 | BioGuider prompt不是邵红的真正prompt — 没有评价维度指导、没有结构化修改方法 | `bioguider/generation/prompts/full_document.txt` 才是真正的BioGuider核心 |
| **PROMPT-3** | P1 | Generic prompt泄露了评价维度（4条criteria），BioGuider prompt反而没列 → Generic能针对rubric优化 | `test_single_file_stress.py:278-284` |
| **PROMPT-4** | P2 | 两个prompt长度差异大（~220 vs ~70 tokens） | 更多约束 ≈ 更多限制 |

### 实验设计问题

| ID | Severity | Problem |
|----|----------|---------|
| **EXP-1** | P1 | `test_skill_comparison` 默认用 `next(iter(MODELS))` = gpt-5.4，不是E001最优的gpt-4o |
| **EXP-2** | P2 | E001只用Seurat vignettes（单repo），外推到100-software需注明局限性 |
| **EXP-3** | P2 | E002 n=1，无统计意义 |

---

## 会议的真正目的（重新确认）

### Benchmark 1：选模型 → ✅ E001已完成
> "一张图就够了，一个heatmap搞定"

- 10 vignettes × 9 levels × 5 models = 450 runs
- 结果：gpt-4o > kimi-k2.5 > glm-5 > gpt-5.4 > gpt-oss
- **模型排序方向性可信**（所有模型用同一个prompt，FP偏差是等量的）
- 需要：修好evaluator后看绝对值是否变化，但排序大概率不变

### Benchmark 2：证明BioGuider的Prompt有用
> "我们核心的价值就是上红写的那一套prompt...我们只是想看看这个prompt有没有用"
> "所以它会省token...把token量跟时间也记下来"

会议定义的比较：
- **Skill 1**: 邵红的结构化prompt（评价维度 + 修改方法 + 代码作为权威）
- **Skill 2**: "I want to refine this tutorial + evaluation metrics"（4行）
- **预期BioGuider优势**: 不只是质量，更是**效率**（省token、省时间）

---

## 修复计划

### Phase 1: 修Evaluator（先修工具再用工具）

#### Fix 1.1: 确定性FP检测器
- **文件**: `bioguider/generation/benchmark_metrics.py` 新方法
- **逻辑**: 对比 baseline 和 revised 的每一行diff，如果某行改动不对应任何 injected error 的 original/mutated snippet → 计为1个FP
- **不用LLM**: 纯字符串比较，零额外成本
- **接口**: `count_deterministic_fp(baseline, revised, error_list) -> int`
- **集成**: 在 `BenchmarkEvaluator.evaluate()` 里调用，替代 `SemanticFPDetector`

#### Fix 1.2: `link` scorer修复
- **文件**: `benchmark_metrics.py:473`（`_check_error_fixed` 的 link 分支）
- **当前**: `re.search(r'\[.*\]\(.*\)', revised)` → 任何合法链接就算修好
- **修改为**: 检查被注入破坏的那条原始链接是否被还原
  ```python
  # 当前（错误）
  if any valid link exists in revised → fixed
  
  # 修改后（正确）
  if original_snippet restored in revised → fixed
  OR if mutated_snippet removed AND a valid link at same position → fixed
  ```

#### Fix 1.3: 查清Headline vs Category不一致
- 调查 `errors_fixed=40` vs `sum(category.fixed)=46` 的差异来源
- 可能原因: UNSCORABLE类别在headline中排除但category中没排除
- 如果是统计口径问题 → 文档说明；如果是bug → 统一逻辑

#### Fix 1.4: `comment_typo` 注入regex
- **文件**: `llm_injector.py:1115` 附近
- **当前**: 匹配 `^#` 开头行（markdown标题）
- **修改为**: 只匹配代码块内 `#` 开头的注释行
- **但**: 已经移到UNSCORABLE，不影响scorable F1
- **优先级**: 低，可以后做

### Phase 2: 重新设计Prompt

#### Skill 1 — BioGuider结构化prompt（v3）

核心原则：保留邵红prompt的**结构化指导价值**，去掉**过度保守约束**。

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
4. Structure: section titles, YAML frontmatter, document organization
   must be preserved and correct

HOW TO FIX (BioGuider methodology)
- For each dimension above, scan the entire document systematically
- Use code blocks as the source of truth for factual claims
- Fix typos, broken links, wrong gene names, incorrect numbers
- Restore proper markdown formatting (backticks, headers, lists)
- Do NOT add new content or remove existing sections
- Output the COMPLETE fixed document as markdown

CORRUPTED DOCUMENT TO FIX:
```

**变化vs当前版本:**
- ✅ 保留了代码作为权威（邵红prompt的核心）
- ✅ 保留了评价维度（与Generic持平，不再信息不对等）
- ✅ 加了"HOW TO FIX"结构化方法论（Generic没有的）
- ❌ 去掉了"only demonstrably wrong"、"do not rewrite"等保守约束
- ❌ 去掉了"PRESERVE YAML/headers"的重复约束（简化为一句）

#### Skill 2 — Generic prompt（保持不变）

```
I want to refine this bioinformatics documentation. Here are the
evaluation criteria I will use to judge the result:

1. Scientific accuracy (gene names, species, statistical tests, parameters)
2. Markdown formatting (headers, lists, links, inline code, tables)
3. Consistency between prose descriptions and code block contents
4. Completeness of required sections (installation, usage, examples)

Please improve this document based on these criteria. Fix any errors you
find. Output the complete corrected document as markdown:
```

**两个prompt的区别（公平比较）:**

| 维度 | Skill 1 (BioGuider) | Skill 2 (Generic) |
|------|-------|---------|
| 评价维度 | ✅ 列出4个（持平） | ✅ 列出4个 |
| 行为约束 | "Fix errors" + "do not add/remove" | "Fix any errors" |
| 方法论 | ✅ 代码作为权威、系统扫描、分维度修复 | ❌ 无 |
| 长度 | ~150 tokens | ~70 tokens |
| 核心差异 | **HOW to fix（方法论）** | 只有WHAT to fix |

### Phase 3: 修实验参数

#### Fix 3.1: 显式指定模型
- `test_skill_comparison`: 改为 `model_name="gpt-4o"`
- `test_skill_matrix`: 同上

#### Fix 3.2: 记录token和时间
- token tracking已经加好（E001显示0是LiteLLM proxy问题）
- duration_s已在CSV中
- 如果token一直是0，用duration作为cost proxy

### Phase 4: 重跑实验

| 实验 | 内容 | 预计时间 | 依赖 |
|------|------|---------|------|
| E002-v2 | Skill comparison, 1 file, gpt-4o, 2 prompts | 5分钟 | Phase 1+2+3 |
| E003 | Skill matrix, 5 files × 3 levels × 2 prompts, gpt-4o | 30分钟 | E002-v2验证通过 |
| E001-v2 | （可选）用修好的evaluator重算E001 CSV | 0分钟（重算不重跑） | Phase 1 |

---

## 不改的东西

- E001模型排序：方向性可信，不重跑
- 评分体系（ReadMe/Installation/UserGuide/Tutorial 四大类分数）：冻结
- 单Repo选模型：论文注明局限性
- 完整pipeline对比：太花时间，prompt对比是会议的决定

## 验收标准

1. FP检测生效：precision ≠ 1.0（至少在generic prompt下）
2. link类别：fix_rate < 1.0（有区分力）
3. BioGuider F1 ≥ Generic F1（如果结构化指导真的有用）
4. BioGuider duration < Generic duration（效率优势）
5. 所有单元测试通过

## 工作量估计

| Phase | 工作量 | 阻塞 |
|-------|--------|------|
| Phase 1 (修Evaluator) | 1.5小时代码 | 无 |
| Phase 2 (改Prompt) | 15分钟 | 无 |
| Phase 3 (改实验参数) | 5分钟 | 无 |
| Phase 4 (重跑) | 35分钟LLM时间 | Phase 1-3 |
| **合计** | **~2.5小时** | |
