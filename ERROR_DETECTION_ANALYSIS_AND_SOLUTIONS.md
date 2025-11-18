# Error Detection Analysis & Solutions

## 📊 Current Status

### Test Results Summary:

| Attempt | Errors Detected | Detection Rate | What Changed |
|---------|----------------|----------------|--------------|
| Baseline | 5/27 | 18.5% | Original prompt |
| After explicit instructions | 5/27 | 18.5% | Added "report all instances" |
| After few-shot examples | 5/27 | 18.5% | Added ✅/❌ examples, still same |

### Detected Errors Breakdown:

**Current Detection (5 errors):**
1. ✅ TYPO: `Sys.Dat()` → `Sys.Date()` (1 of 5 function errors)
2. ✅ TYPO: `exampl` → `example` (1 of 2 unique typos)
3. ✅ TYPO: `analysi` → `analysis` (**1 of 4 instances**)  ← ISSUE!
4. ✅ LINK: `https//www.nature.com/articles/nbt.4042` → fixed (1 of 5)
5. ✅ LINK: `https//github.com/satijalab/seurat-data` → fixed (**2 of 5**) ← PARTIAL!

**Still Missing (22 errors):**
- ❌ 3 more instances of "analysi"
- ❌ 3 more malformed URLs
- ❌ 2 bio_term errors: "single sell", "genomis" 
- ❌ 4 more function errors (Dat/Datx mutations)
- ❌ 5 markdown_structure errors (broken code fences)
- ❌ 5 inline_code errors (missing backticks)

## 🔍 Root Cause Analysis

### Why Only 5 Errors Detected?

1. **LLM Optimization for Conciseness**
   - Despite explicit instructions, LLMs are trained to be concise
   - Groups similar errors to save tokens
   - Prioritizes "high-level summary" over "exhaustive list"

2. **Token/Response Length Limits**
   - May hit token limit before listing all errors
   - Conservative behavior to stay within bounds

3. **Schema Design Issue**
   - `readability_suggestions: list[str]` might not enforce individual listing
   - LLM sees it as "list of suggestion types" not "list of error instances"

4. **Error Categories Not Explicitly Searched**
   - bio_term: Completely missed (0/2)
   - inline_code: Completely missed (0/5)
   - markdown_structure: Partially detected but grouped

## 💡 Comprehensive Solutions

### Solution 1: **Restructure Data Schema** ⭐ RECOMMENDED

Change from generic list to structured error counting:

```python
# Current schema (in constants.py):
class TutorialEvaluationResult(BaseModel):
    readability_suggestions: list[str]  # ← Too generic!

# Proposed schema:
class ErrorInstance(BaseModel):
    error_type: str  # "typo", "link", "bio_term", etc.
    original: str
    corrected: str
    location: str  # Context or line reference
    
class TutorialEvaluationResult(BaseModel):
    readability_score: int
    errors_found: list[ErrorInstance]  # ← Structured!
    error_count_by_type: dict[str, int]  # e.g., {"typo": 5, "link": 5}
    general_suggestions: list[str]  # Non-error improvements
```

**Advantages:**
- Forces LLM to output each error as separate object
- Can validate count matches expected
- Easier to analyze and compare

**Implementation:**
1. Update `constants.py` with new schema
2. Update prompt to request new format
3. Update report generation to use structured errors

### Solution 2: **Two-Pass Evaluation** ⭐ GOOD FOR THOROUGHNESS

Split error detection into multiple focused passes:

```python
# Pass 1: Error Counting
prompt_1 = """
Count how many times each error appears:
- "analysi": __ times
- "exampl": __ times
- Malformed URLs: __ links
- Bio term errors: __ terms
...
"""

# Pass 2: List Each Instance
prompt_2 = """
Now list EVERY individual instance you found:
1. TYPO: 'analysi' → 'analysis' (occurrence 1/4)
2. TYPO: 'analysi' → 'analysis' (occurrence 2/4)
...
"""
```

**Advantages:**
- First pass commits to count
- Second pass must match the count
- More thorough

**Disadvantages:**
- 2x LLM calls = 2x cost
- More complex implementation

### Solution 3: **Increase Max Tokens Significantly**

```python
# In evaluation_tutorial_task.py, _evaluate_individual_tutorial():
agent = CommonAgentTwoSteps(
    llm=self.llm.with_config(max_completion_tokens=8192)  # 2x increase
)
```

**Test this next!**

### Solution 4: **Template with Required Sections**

Force LLM to report on each category explicitly:

```
**Typo Errors Found:**
1. [List all typos]
2. ...

**Link Errors Found:**
1. [List all link errors]
2. ...

**Bio Term Errors Found:**
1. [List all bio term errors]
2. ...

... (continue for all categories)

**TOTAL ERROR COUNT**: [sum of all above]
```

### Solution 5: **Use Claude/GPT with Higher Instruction Following**

Some models follow detailed instructions better:
- Claude 3.5 Sonnet: Excellent instruction following
- GPT-4 Turbo: Good but can be concise
- Consider model switch for error detection specifically

## 🧪 Recommended Testing Sequence

### Step 1: Increase Max Tokens (Quick Test)
```python
# Modify evaluation_tutorial_task.py
max_completion_tokens=8192  # or even 16384
```
**Expected**: Should get more errors listed (maybe 10-15/27)

### Step 2: Add Error Count Requirement
```python
# Add to prompt:
"At the very end, report: 'TOTAL ERRORS FOUND: X'"
```
**Expected**: Forces LLM to be accountable for count

### Step 3: Implement Structured Schema
- Update `TutorialEvaluationResult` with `ErrorInstance` list
- Update prompts to request structured format
**Expected**: Should get closer to 20-25/27

### Step 4: Two-Pass if Still Low
- Implement counting pass + listing pass
**Expected**: Should achieve 25-27/27

## 📈 Expected Outcomes by Solution

| Solution | Expected Detection Rate | Complexity | Cost Impact |
|----------|------------------------|------------|-------------|
| **Increase tokens** | 40-60% (11-16/27) | Low | +token cost |
| **Structured schema** | 60-80% (16-22/27) | Medium | Same |
| **Template sections** | 70-85% (19-23/27) | Medium | Same |
| **Two-pass** | 85-100% (23-27/27) | High | 2x cost |
| **All combined** | 95-100% (26-27/27) | High | 2-3x cost |

## 🎯 Action Plan

### Immediate Actions (Low Effort):

1. ✅ **DONE**: Enhanced prompts with few-shot examples
2. ⏭️ **NEXT**: Increase `max_completion_tokens` to 8192
3. ⏭️ **NEXT**: Add error count validation to prompt

### Short-term (Medium Effort):

4. Implement structured `ErrorInstance` schema
5. Add template with required sections for each category
6. Test with different LLM models

### Long-term (if needed):

7. Implement two-pass evaluation
8. Build error detection dashboard
9. A/B test different prompting strategies

## 📝 Files to Modify

### For Max Tokens:
- `bioguider/agents/evaluation_tutorial_task.py` (line ~173)

### For Structured Schema:
- `bioguider/utils/constants.py` (add `ErrorInstance` model)
- `bioguider/agents/evaluation_tutorial_task.py` (update schema usage)
- `bioguider/agents/evaluation_tutorial_task_prompts.py` (update output format)
- `system_tests/test_evaluation_tutorial_corrupted_task.py` (update report generation)

### For Two-Pass:
- `bioguider/agents/evaluation_tutorial_task.py` (add second evaluation pass)
- Create new prompts for counting vs listing

---

## 🔍 Next Immediate Test

**Modify:** `/Users/wang.13246/Documents/GitHub/bioguider/bioguider/agents/evaluation_tutorial_task.py`

```python
# Line ~173, change:
agent = CommonAgent TwoSteps(llm=self.llm)

# To:
from langchain_core.runnables import RunnableConfig
agent = CommonAgentTwoSteps(
    llm=self.llm.with_config(
        RunnableConfig(max_completion_tokens=8192)
    )
)
```

**Then run test again to see if token limit was the bottleneck!**

---

*Created: 2025-11-12*
*Purpose: Comprehensive analysis of error detection issues and solutions*



