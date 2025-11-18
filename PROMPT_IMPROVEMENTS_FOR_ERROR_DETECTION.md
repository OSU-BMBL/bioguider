# Prompt Improvements for Error Detection

## 📊 Problem Analysis

### Issue: Low Detection Rate (5/27 = 18.5%)

**Root Causes Identified:**

1. **Duplicate Errors Reported Once**
   - Injected: "analysi" appears 4 times
   - Detected: LLM reports "analysi" only once
   - **Lost**: 3 error instances

2. **Similar Errors Grouped**
   - Injected: 5 malformed URLs (all missing `:` in `https//`)
   - Detected: LLM reports "malformed URLs" as one issue
   - **Lost**: 4 error instances

3. **Missing Category**
   - Injected: 2 bio_term errors ("single sell", "genomis")
   - Detected: 0 bio_term errors
   - **Lost**: 2 error instances

4. **No Count Enforcement**
   - Prompt said "For EACH error" but LLM interpreted as "each error TYPE"
   - No explicit instruction to count occurrences

### Error Distribution Analysis:

| Category | Injected | Unique | Duplicate | Pattern |
|----------|----------|--------|-----------|---------|
| typo | 5 | 2 | 4x "analysi" | Multiple instances of same error |
| link | 5 | 5 | Same pattern | All missing `:` in `https//` |
| function | 5 | 2 | Chain mutation | Dat ↔ Datx repeated |
| markdown_structure | 5 | 1 | Same error | Broken code fence repeated |
| inline_code | 5 | 5 | Different locations | Each unique but similar |
| bio_term | 2 | 2 | None | Different errors, both missed |

**Total**: 27 injected = 17 unique error types + 10 duplicate instances

## ✅ Solutions Implemented

### 1. **Explicit Instance Counting**

**Before:**
```
* **CRITICAL - Error Detection**: You MUST scan for and identify ALL errors and anomalies:
  - **Typos and spelling errors**: Misspelled words...
* **For EACH error found**, provide specific text snippets...
```

**After:**
```
* **CRITICAL - Error Detection**: You MUST scan for and identify ALL error INSTANCES (not just types):
  - **Typos and spelling errors**: Misspelled words...
    * If the SAME typo appears multiple times, LIST EACH OCCURRENCE separately
* **IMPORTANT**: Report EVERY INDIVIDUAL ERROR INSTANCE
  - If "analysi" appears 4 times, report it 4 times (with line references if possible)
  - If 5 URLs are malformed, report all 5 individually
  - Do NOT group similar errors - LIST EACH ONE SEPARATELY
```

### 2. **Category-Specific Instructions**

Added explicit instructions for each error type:

```
- **Malformed links**: URLs missing colons (e.g., "https//..." should be "https://...")
  * Check EVERY link/URL in the document
  
- **Markdown/RMarkdown syntax errors**: 
  * Check ALL code blocks and headers
  
- **Bio/domain term errors**: Wrong scientific terms (e.g., "single sell" → "single cell")
  * Pay special attention to biology/bioinformatics terminology
  
- **Function name errors**: Misspelled function/API names (e.g., "Dat()" → "Date()")
  * Check ALL function calls in code blocks
  
- **Inline code formatting**: Missing backticks around code elements
  * Check that all code references use proper backtick formatting
```

### 3. **Enhanced Output Format**

**Before:**
```
* **Readability Improvement Suggestions:** MUST include ALL detected errors:
  - **Typos**: "original misspelled text" → "corrected text"
  - **Links**: "broken link" → "fixed link"
  ...
  Format each error clearly with: "ERROR_TYPE: 'original' → 'corrected' - explanation"
```

**After:**
```
* **Readability Improvement Suggestions:** MUST include ALL detected errors:
  - **CRITICAL**: List EVERY INDIVIDUAL ERROR INSTANCE (not grouped)
  - **Example**: If "analysi" appears 4 times, list 4 separate error entries
  - **Example**: If 5 URLs are malformed, list 5 separate error entries
  
  **Format for each error** (list them ALL individually):
  ...
  Format: "ERROR_TYPE: 'original' → 'corrected' - explanation (occurrence #X if multiple)"
```

### 4. **Updated Grading Scale**

**Before:**
```
- **65-84**: The documentation is clear with only minor errors (1-3 small issues).
- **45-64**: The documentation has noticeable errors (4-10 typos, links, markdown issues).
```

**After:**
```
* **Grade Level** (based on TOTAL error instances, not types):
  - **85-100**: ERROR-FREE (0 errors).
  - **65-84**: Only minor errors (1-5 total error instances).
  - **45-64**: Noticeable errors (6-15 total error instances).
  - **0-44**: Numerous errors (16+ total error instances).
  - **Note**: Count EVERY instance - if "analysi" appears 4 times, that's 4 errors, not 1.
```

## 🧪 Testing Strategy

### Test Execution:
```bash
cd /Users/wang.13246/Documents/GitHub/bioguider
/Users/wang.13246/.local/share/mamba/envs/bioguider-py/bin/pytest \
  system_tests/test_evaluation_tutorial_corrupted_task.py::test_EvaluationTutorialCorruptedTask_DeVignette_Low \
  -v -s
```

### Expected Improvements:

**Before Improvements:**
- Detection Rate: 5/27 = 18.5%
- Issues: Duplicates counted once, bio_terms missed

**Expected After Improvements:**
- Detection Rate: ~20-25/27 = 74-93%
- Improvements:
  - "analysi" reported 4 times (not 1)
  - All 5 malformed URLs reported individually
  - Bio terms detected: "single sell", "genomis"
  - All inline code errors reported
  - All function name errors reported

### Key Metrics to Monitor:

1. **Individual Error Detection Rate**:
   - Target: >70% (at least 19/27 errors)
   
2. **Category Coverage**:
   - Target: 6/6 = 100% (including bio_term)
   
3. **Duplicate Instance Reporting**:
   - "analysi": Should see 4 separate reports
   - Malformed URLs: Should see 5 separate reports
   - Broken code fences: Should see 5 separate reports

## 📈 Expected Impact

### Optimistic Scenario (Best Case):
- LLM follows instructions perfectly
- Reports all 27 instances individually
- Detection rate: **100%** 🎯

### Realistic Scenario (Most Likely):
- LLM reports most duplicates individually
- May still group some very similar errors
- Detection rate: **70-85%** (19-23 errors) ✅

### Conservative Scenario (Worst Case):
- Moderate improvement in instance reporting
- Still some grouping of duplicates
- Detection rate: **40-60%** (11-16 errors) ⚠️

## 🔄 Iterative Improvement Plan

If detection rate is still low after these changes:

### Additional Strategies:

1. **Increase Token Limits**:
   ```python
   # In evaluation_tutorial_task.py
   max_completion_tokens=8192  # Double current limit
   ```

2. **Add Error Count Validation**:
   ```python
   # Add to prompt:
   "At the end, report: 'TOTAL ERRORS FOUND: [count]'"
   ```

3. **Split Error Detection into Phases**:
   - Phase 1: Detect typos and spelling
   - Phase 2: Detect links and markdown
   - Phase 3: Detect bio terms and functions

4. **Use Few-Shot Examples**:
   ```python
   # Add to prompt:
   "Example output:
   1. TYPO: 'analysi' → 'analysis' - line 23
   2. TYPO: 'analysi' → 'analysis' - line 45
   3. TYPO: 'analysi' → 'analysis' - line 67
   ..."
   ```

## 📝 Files Modified

1. **`bioguider/agents/evaluation_tutorial_task_prompts.py`**:
   - Lines 15-35: Enhanced error detection instructions
   - Lines 36-41: Updated grading scale
   - Lines 118-133: Enhanced output format requirements

## 🚀 Next Steps

1. **Run Test**: Execute test to measure improvement
2. **Analyze Results**: Check detection rate and error breakdown
3. **Iterate**: If <70% detection, apply additional strategies
4. **Document**: Update this file with actual results
5. **Apply to README**: Use same improvements for README evaluation prompts

---

*Created: 2025-11-12*
*Purpose: Track prompt engineering improvements for error detection*



