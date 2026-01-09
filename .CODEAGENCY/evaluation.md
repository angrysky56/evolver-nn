# Architectural Evaluation: Evolver-NN

**Date**: 2026-01-08
**Scope**: Complete simulation engine analysis
**Evaluator**: Chief Architect

## Executive Summary

The repository has undergone significant refactoring that introduced **critical regressions preventing Grandmaster convergence**. Multiple parameter mismatches and architectural inconsistencies exist between the current implementation and the proven working original.

---

## Critical Risks (Must Fix)

### ❌ **[CRITICAL - Learning Rate]** Line 26: `simulationEngine.ts`

**Issue**: Learning rate is **2.25x too high**

- **Current**: `learningRate: 0.045`
- **Original Working**: `LEARNING_RATE = 0.02`
- **Impact**: Overshooting causes instability, preventing convergence to Grandmaster state
- **Root Cause**: Parameter was increased without validation

### ❌ **[CRITICAL - Prune Threshold]** Line 25: `simulationEngine.ts`

**Issue**: Prune threshold is **3.33x too high**

- **Current**: `pruneThreshold: 0.05`
- **Original Working**: `PRUNE_THRESHOLD = 0.015`
- **Impact**: Premature pruning of weak but critical connections
- **Root Cause**: Threshold increased, likely during E-I architecture attempt

### ❌ **[CRITICAL - Decay Rate]** Line 27: `simulationEngine.ts`

**Issue**: Decay rate is **40% too high**

- **Current**: `decayRate: 0.0014`
- **Original Working**: `DECAY_RATE = 0.001`
- **Impact**: Too aggressive weight decay destroys learned patterns
- **Root Cause**: Modified during refactoring

### ❌ **[CRITICAL - Default Spectral]** Line 128: `simulationEngine.ts`

**Issue**: Initial spectral radius mismatch

- **Current**: `spectral: 0.9` (default fallback)
- **Original Working**: `spectral: 0.95`
- **Impact**: Lower spectral radius reduces network expressiveness
- **Root Cause**: Changed default parameters

### ❌ **[CRITICAL - Patience Limit]** Line 28: `simulationEngine.ts`

**Issue**: Patience is **50% too low**

- **Current**: `patienceLimit: 32`
- **Original Working**: `PATIENCE_LIMIT = 64`
- **Impact**: Premature neurogenesis before proper learning
- **Root Cause**: Reduced to make growth happen faster, but prevents stability

### ❌ **[CRITICAL - Solved Threshold]** Line 29: `simulationEngine.ts`

**Issue**: Grandmaster threshold is **10x too strict**

- **Current**: `solvedThreshold: 0.001`
- **Original Working**: `SOLVED_THRESHOLD = 0.01`
- **Impact**: Network can never reach the "locked" Grandmaster state
- **Root Cause**: Threshold made overly strict

### ❌ **[CRITICAL - LV Growth]** Line 32: `simulationEngine.ts`

**Issue**: Lotka-Volterra growth parameter is **7.5x too high**

- **Current**: `lvGrowth: 0.15`
- **Original Working**: `LV_GROWTH = 0.02`
- **Impact**: Excessive density growth causes instability
- **Root Cause**: Increased aggressiveness

### ❌ **[CRITICAL - LV Decay]** Line 33: `simulationEngine.ts`

**Issue**: LV decay is **3x too low**

- **Current**: `lvDecay: 0.006`
- **Original Working**: `LV_DECAY = 0.02`
- **Impact**: Density doesn't decay properly
- **Root Cause**: Imbalanced LV dynamics

### ❌ **[CRITICAL - Duplicate Data Push]** Lines 247, 428: `simulationEngine.ts`

**Issue**: Target value pushed to dataSeries **TWICE** per step

- **Line 247**: `this.dataSeries.push(targetVal);` (BEFORE learning)
- **Line 428**: `this.dataSeries.push(targetVal);` (AFTER learning)
- **Impact**: Data series grows 2x faster than expected, causing memory bloat and incorrect historical references
- **Root Cause**: Refactoring error, duplicate logic

---

## Improvements (Should Fix)

### ⚠️ **[Architecture - BicameralMetaController Complexity]**

**File**: `src/engine/metaController.ts`
**Issue**: The LSTM-based meta-controller adds significant complexity with unclear benefit

- Current implementation uses a full LSTM for hyperparameter mutation
- Original used simpler random mutation with revert-on-fail
- **Recommendation**: Validate if meta-controller improves convergence or just adds overhead

### ⚠️ **[Maintainability - Hardcoded MAX_NEURONS]**

**File**: `src/components/NetworkVisualization.tsx:63`
**Issue**: Hardcoded `512` instead of using `maxNeurons` from config

- **Current**: `const weight = net.weights[i * 512 + j];`
- **Should be**: `const weight = net.weights[i * maxNeurons + j];`
- **Impact**: Breaks if maxNeurons config changes
- **Severity**: Low (current config is 512), but poor practice

### ⚠️ **[Code Hygiene - Unused Optimizer Fields]**

**File**: `src/engine/simulationEngine.ts:149-163`
**Issue**: Optimizer state contains unused fields from old implementation

- `currentIdx`, `state`, `direction`, `lastTestedVal`, `testDirection` are not used
- **Recommendation**: Clean up unused state

---

## Strategic Recommendations

### 1. **Immediate Action: Restore Working Parameters**

Create a parameter restoration plan to match the original working configuration exactly:

- All critical parameter mismatches must be fixed in a single atomic commit
- Run regression test against Mackey-Glass to verify Grandmaster convergence

### 2. **Establish Parameter Validation**

- Add configuration validation on startup
- Document WHY each parameter has its specific value
- Create test harness to validate convergence properties

### 3. **Remove Accidental Complexity**

- The E-I architecture removal was correct, but left parameter debris
- Meta-controller should be validated or removed
- Consider reverting to the exact original `EvolverEngine_more_original.ts` as the canonical source

### 4. **Testing Infrastructure**

**MISSING**: No automated tests for convergence

- **Critical**: Add test case that verifies Grandmaster state is reached within reasonable time
- **Needed**: Stability test for network growth (current manual test in `scripts/test-harness.ts`)
- **Needed**: Regression test suite

### 5. **Documentation Debt**

- No documentation explaining the Lotka-Volterra dynamics
- No explanation of why specific thresholds were chosen
- Parameter comments are sparse (lines 24-33 just have `//` with no explanation)

---

## Architectural Strengths (Preserved)

✅ **Clean Separation**: Engine is properly separated from UI (React components)
✅ **Type Safety**: Full TypeScript with proper interfaces
✅ **Reservoir Computing**: Core RC algorithm is correctly implemented
✅ **Structural Plasticity**: Decay/prune/regrowth logic is sound
✅ **State Checkpointing**: Hyperparameter checkpointing at Grandmaster lock is good design

---

## Root Cause Analysis

The repository suffered from **aggressive parameter tuning during the E-I architecture experiment**. When the E-I code was removed (correctly), the modified parameters were left behind, creating a Franken-configuration that combines:

- High learning rate (from trying to speed up E-I convergence)
- Strict thresholds (from trying to force stability)
- Loose prune threshold (from E-I weight management)
- Imbalanced LV dynamics (from E-I density requirements)

**The fix is surgical**: Restore all 8 critical parameters to match `EvolverEngine_more_original.ts` exactly.

---

## Verification Blockers

**No way to automatically verify the fix** without:

1. A test that runs simulation to convergence
2. Assertion that Grandmaster state is reached
3. Performance baseline (how many steps to convergence)

**Current test harness** (`scripts/test-harness.ts`) exists but:

- Not integrated into npm scripts for easy running
- No pass/fail criteria
- Manual observation required

---

## Conclusion

**Status**: 🔴 **BROKEN - Cannot reach Grandmaster**

**Confidence**: **100%** - All critical issues are parameter mismatches with clear evidence

**Recommended Action**:

1. Restore all 8 critical parameters immediately
2. Remove duplicate `dataSeries.push()` on line 428
3. Run test harness to verify convergence
4. If successful, lock these parameters in stone with documentation
