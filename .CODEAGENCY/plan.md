# Implementation Plan: Restore Working Parameters

**Objective**: Fix 9 critical parameter bugs preventing Grandmaster convergence
**Based on**: `.CODEAGENCY/evaluation.md`
**Target**: Return simulation to proven working state from `EvolverEngine_more_original.ts`

---

## Phase 1: Preparation & Baseline

- [ ] **Verify current broken state**

  - Run: `npm run test:quick`
  - **Expected**: Should NOT reach Grandmaster (avgLoss stuck > 0.001)
  - **Purpose**: Establish broken baseline

- [ ] **Create backup of current config**
  - Backup: `src/engine/simulationEngine.ts` lines 22-34
  - **Purpose**: Easy rollback if needed

---

## Phase 2: Atomic Parameter Fix

All parameters must be fixed in a SINGLE commit to avoid partial broken states.

- [ ] **Fix DEFAULT_CONFIG parameters** (`src/engine/simulationEngine.ts:22-34`)

  - Change `pruneThreshold: 0.05` → `0.015` (3.33x reduction)
  - Change `learningRate: 0.045` → `0.02` (2.25x reduction)
  - Change `decayRate: 0.0014` → `0.001` (1.4x reduction)
  - Change `patienceLimit: 32` → `64` (2x increase)
  - Change `solvedThreshold: 0.001` → `0.01` (10x relaxation)
  - Change `lvGrowth: 0.15` → `0.02` (7.5x reduction)
  - Change `lvDecay: 0.006` → `0.02` (3.33x increase)

- [ ] **Fix default spectral radius** (`src/engine/simulationEngine.ts:128`)

  - Change: `{ leak: 0.8, spectral: 0.9, inputScale: 1.0 }`
  - To: `{ leak: 0.8, spectral: 0.95, inputScale: 1.0 }`

- [ ] **Remove duplicate dataSeries.push()** (`src/engine/simulationEngine.ts:428`)
  - Delete line: `this.dataSeries.push(targetVal);`
  - **Reason**: Already pushed on line 247, causes 2x growth

---

## Phase 3: Verification

### Test 1: TypeScript Compilation

- [ ] **Run**: `npx tsc --noEmit`
- [ ] **Expected**: Zero errors
- [ ] **Blocker**: If fails, fix syntax before proceeding

### Test 2: Quick Convergence Test

- [ ] **Run**: `npm run test:quick`
- [ ] **Expected**:
  - Simulation reaches Grandmaster (avgLoss < 0.01)
  - Status shows "LOCKED"
  - No NaN values in neuron count or loss
  - Network grows beyond initial 8 neurons
- [ ] **Pass Criteria**: "✓ Grandmaster state achieved" in output

### Test 3: Stability Test (Extended)

- [ ] **Run**: `npm run test:stability`
- [ ] **Expected**:
  - Sustained Grandmaster lock for 1000+ steps
  - avgLoss remains < 0.015
  - No crashes from gain explosion
  - Network size stabilizes (not constantly growing)
- [ ] **Pass Criteria**: Completes without errors

### Test 4: Manual UI Verification

- [ ] **Run**: `npm run dev`
- [ ] **Open**: http://localhost:5173 in browser
- [ ] **Observe**:
  - Network visualization shows active connections (cyan/red lines)
  - Neuron count increases from 8
  - Loss graph trends downward
  - Status eventually shows "LOCKED" (purple)
  - avgLoss drops below 0.01
- [ ] **Duration**: Run for 2-3 minutes
- [ ] **Pass Criteria**: Reaches and maintains LOCKED state

---

## Phase 4: Documentation

- [ ] **Add parameter documentation** in `src/engine/simulationEngine.ts:22-34`

  - Add comments explaining WHY each value was chosen
  - Reference that these values are from validated working config
  - Warn against modification without extensive testing

- [ ] **Update evaluation.md**
  - Mark all critical issues as [RESOLVED]
  - Add timestamp of fix
  - Document verification results

---

## Rollback Plan

If verification fails:

1. Restore backup of `simulationEngine.ts`
2. Check for conflicts with other recent changes
3. Compare byte-for-byte with `docs/EvolverEngine_more_original.ts`
4. Report failure to user with diagnostic data

---

## Success Criteria

✅ All 9 critical bugs fixed
✅ TypeScript compiles
✅ Quick test passes (reaches Grandmaster)
✅ Stability test passes (sustains Grandmaster)
✅ Manual UI test shows LOCKED state
✅ Parameters documented

---

## Risk Mitigation

**Risk**: Parameters were changed for a reason we don't know
**Mitigation**: Have user review evaluation.md before proceeding

**Risk**: Other code depends on new parameter values
**Mitigation**: Full test suite verifies end-to-end behavior

**Risk**: Fix doesn't work
**Mitigation**: Rollback plan + user consultation

---

## Notes for Senior Engineer

- Fix is **surgical**: 8 number changes + 1 line deletion
- **Do not** modify any logic, only parameter values
- **Do not** refactor while fixing
- **Verify** each change against evaluation.md
- **Test** immediately after Phase 2 completion
