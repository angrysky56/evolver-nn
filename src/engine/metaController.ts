/**
 * Bicameral Meta-Controller
 *
 * A dual Short-Term / Long-Term LSTM architecture for hyperparameter tuning.
 * Uses REINFORCE-style reward signal based on loss improvement.
 *
 * Architecture:
 * - Short-Term LSTM: Reacts to error spikes (exploitation/fine-tune).
 * - Long-Term LSTM: Tracks slow trends (exploration/resets).
 * - Soft Gate: Sigmoid-gated blend of both outputs.
 */

// ==================== LSTM CELL ====================
// Minimal LSTM implementation in pure TypeScript
// Based on: h_t, c_t = LSTM(x_t, h_{t-1}, c_{t-1})

interface LSTMWeights {
    // Gates: input, forget, cell, output (4 * hiddenSize weights each)
    Wi: Float32Array; // Input gate: [inputSize, hiddenSize]
    Wf: Float32Array; // Forget gate
    Wc: Float32Array; // Cell gate
    Wo: Float32Array; // Output gate

    Ui: Float32Array; // Recurrent input gate: [hiddenSize, hiddenSize]
    Uf: Float32Array;
    Uc: Float32Array;
    Uo: Float32Array;

    bi: Float32Array; // Biases
    bf: Float32Array;
    bc: Float32Array;
    bo: Float32Array;
}

interface LSTMState {
    hidden: Float32Array;
    cell: Float32Array;
}

function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

function tanh(x: number): number {
    return Math.tanh(x);
}

function softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expValues = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    return expValues.map(e => e / sumExp);
}

function initLSTMWeights(inputSize: number, hiddenSize: number, scale: number = 0.1): LSTMWeights {
    const randArr = (len: number) => {
        const arr = new Float32Array(len);
        for (let i = 0; i < len; i++) arr[i] = (Math.random() * 2 - 1) * scale;
        return arr;
    };

    return {
        Wi: randArr(inputSize * hiddenSize),
        Wf: randArr(inputSize * hiddenSize),
        Wc: randArr(inputSize * hiddenSize),
        Wo: randArr(inputSize * hiddenSize),
        Ui: randArr(hiddenSize * hiddenSize),
        Uf: randArr(hiddenSize * hiddenSize),
        Uc: randArr(hiddenSize * hiddenSize),
        Uo: randArr(hiddenSize * hiddenSize),
        bi: new Float32Array(hiddenSize).fill(0),
        bf: new Float32Array(hiddenSize).fill(1), // Forget gate bias = 1 (remember by default)
        bc: new Float32Array(hiddenSize).fill(0),
        bo: new Float32Array(hiddenSize).fill(0),
    };
}

function lstmStep(
    input: Float32Array,
    prevState: LSTMState,
    weights: LSTMWeights,
    hiddenSize: number
): LSTMState {
    const inputSize = input.length;
    const h = prevState.hidden;
    const c = prevState.cell;

    const newHidden = new Float32Array(hiddenSize);
    const newCell = new Float32Array(hiddenSize);

    for (let j = 0; j < hiddenSize; j++) {
        // Compute gate pre-activations
        let i_pre = weights.bi[j];
        let f_pre = weights.bf[j];
        let c_pre = weights.bc[j];
        let o_pre = weights.bo[j];

        // Input contribution: W * x
        for (let k = 0; k < inputSize; k++) {
            const idx = k * hiddenSize + j;
            i_pre += weights.Wi[idx] * input[k];
            f_pre += weights.Wf[idx] * input[k];
            c_pre += weights.Wc[idx] * input[k];
            o_pre += weights.Wo[idx] * input[k];
        }

        // Recurrent contribution: U * h_{t-1}
        for (let k = 0; k < hiddenSize; k++) {
            const idx = k * hiddenSize + j;
            i_pre += weights.Ui[idx] * h[k];
            f_pre += weights.Uf[idx] * h[k];
            c_pre += weights.Uc[idx] * h[k];
            o_pre += weights.Uo[idx] * h[k];
        }

        // NaN Guard: Clamp pre-activations to prevent overflow
        i_pre = Math.max(-50, Math.min(50, i_pre));
        f_pre = Math.max(-50, Math.min(50, f_pre));
        c_pre = Math.max(-50, Math.min(50, c_pre));
        o_pre = Math.max(-50, Math.min(50, o_pre));

        // Gate activations
        const i_gate = sigmoid(i_pre);
        const f_gate = sigmoid(f_pre);
        const c_gate = tanh(c_pre);
        const o_gate = sigmoid(o_pre);

        // New cell state and hidden state
        let cellVal = f_gate * c[j] + i_gate * c_gate;

        // CRITICAL: Clamp cell state to prevent explosion
        // LSTM cells can accumulate unbounded values over many steps
        cellVal = Math.max(-10, Math.min(10, cellVal));

        // NaN check
        if (isNaN(cellVal)) cellVal = 0;

        newCell[j] = cellVal;
        newHidden[j] = o_gate * tanh(newCell[j]);

        // NaN check for hidden
        if (isNaN(newHidden[j])) newHidden[j] = 0;
    }

    return { hidden: newHidden, cell: newCell };
}

// The "Gears" for the Network
// The "Gears" for the Network
// Updated to match narrowed PARAM_RANGES
export const REGIMES = {
    CRUISE: {
        leak: 0.3,            // Low end of [0.2, 0.8]
        spectral: 0.7,        // Low end of [0.6, 1.0]
        inputScale: 0.7,      // Low end of [0.6, 1.2]
        learningRate: 0.005,  // Low end of [0.001, 0.05] - Stable
        lvGrowth: 0.03,       // Low end of [0.02, 0.1]
        lvDecay: 0.008,       // Low end of [0.005, 0.02]
        smoothingFactor: 0.08, // High end of [0.01, 0.1] - More smoothing
        outputGain: 0.8,      // Low end of [0.6, 1.4]
        name: "CRUISE"
    },
    SPORT: {
        leak: 0.5,            // Midpoint
        spectral: 0.85,       // Midpoint-high
        inputScale: 0.95,     // Midpoint
        learningRate: 0.025,  // Midpoint - Active
        lvGrowth: 0.06,       // Midpoint
        lvDecay: 0.0125,      // Midpoint
        smoothingFactor: 0.05, // Midpoint
        outputGain: 1.0,      // Midpoint
        name: "SPORT"
    },
    OFFROAD: {
        leak: 0.7,            // High end
        spectral: 0.95,       // High end (still under 1.0)
        inputScale: 1.1,      // High end
        learningRate: 0.04,   // High end - Aggressive but within bounds
        lvGrowth: 0.08,       // High end
        lvDecay: 0.015,       // High end
        smoothingFactor: 0.02, // Low end - faster adaptation
        outputGain: 1.2,      // High end
        name: "OFFROAD"
    }
};

export type RegimeName = keyof typeof REGIMES;

/**
 * DISCRETE STRATEGIES for LSTM Strategy Selector
 * The LSTM learns WHEN to use each strategy, not HOW to tune parameters.
 * Each strategy is a complete, pre-tuned parameter set for a specific purpose.
 */
export const STRATEGIES = {
    // 0: Bold experimentation - high variance, trying new things
    EXPLORE: {
        leak: 0.5,
        spectral: 0.85,
        inputScale: 1.0,
        learningRate: 0.02,      // Reduced from 0.03 - safer exploration
        lvGrowth: 0.08,          // High - grow aggressively
        lvDecay: 0.005,          // Low - keep connections
        outputGain: 1.1,
        smoothingFactor: 0.02,   // Low - fast adaptation
        name: "EXPLORE"
    },
    // 1: Refine what works - low variance, conservative
    EXPLOIT: {
        leak: 0.4,
        spectral: 0.8,
        inputScale: 0.9,
        learningRate: 0.008,     // Reduced from 0.01 - safer refinement
        lvGrowth: 0.03,          // Low - minimal growth
        lvDecay: 0.01,           // Moderate pruning
        outputGain: 1.0,
        smoothingFactor: 0.06,   // Moderate smoothing
        name: "EXPLOIT"
    },
    // 2: Stop divergence - emergency stabilization
    STABILIZE: {
        leak: 0.4,               // Balanced leak
        spectral: 0.85,          // RESTORED: Needs energy to oscillate! 0.5 killed it.
        inputScale: 0.9,         // RESTORED: Needs to see the signal.
        learningRate: 0.000,     // ZERO: Stop breaking things, just run.
        lvGrowth: 0.0,           // Zero growth
        lvDecay: 0.005,          // Gentle cleanup
        outputGain: 1.0,         // RESTORED: Signal needs amplitude.
        smoothingFactor: 0.1,    // Filter noise
        name: "STABILIZE"
    },
    // 3: Escape local minima - random perturbation
    RESET: {
        leak: 0.6,               // Different from default
        spectral: 0.9,           // Higher risk
        inputScale: 1.1,
        learningRate: 0.02,
        lvGrowth: 0.1,           // Very high - shake things up
        lvDecay: 0.02,           // Higher - aggressive pruning
        outputGain: 1.2,
        smoothingFactor: 0.01,   // Very low - immediate changes
        name: "RESET"
    },
    // 4: Lock in progress - protect learned weights
    CONSOLIDATE: {
        leak: 0.45,
        spectral: 0.8,
        inputScale: 0.9,
        learningRate: 0.002,     // Minimal - protect what we have
        lvGrowth: 0.0,           // Zero - no structural changes
        lvDecay: 0.002,          // Minimal pruning
        outputGain: 1.0,
        smoothingFactor: 0.1,    // High - very smooth
        name: "CONSOLIDATE"
    }
};

export type StrategyName = keyof typeof STRATEGIES;
export const STRATEGY_LIST = Object.keys(STRATEGIES) as StrategyName[];
export const NUM_STRATEGIES = STRATEGY_LIST.length;

export interface MetaControllerConfig {
    inputWindowSize: number;     // Not used for input anymore, but for internal state
    hiddenSize: number;          // LSTM hidden dimension
    outputSize: number;          // 3 (CRUISE, SPORT, OFFROAD)
    targetOutputSize: number;    // Max neurons for target output (DiscoRL-style) - KEPT for Legacy/Future
    shortTermLR: number;
    longTermLR: number;
    gateLR: number;
    targetLR: number;
    updateInterval: number;
    warmupSteps: number;
    inputFeaturesSize: number;   // 3 (Variance, Slope, Signal)
    contextSize: number;         // Kept for type safety though unused in new logic
    deltaScale: number;          // Kept for type safety
    targetBlend: number;         // Kept for type safety
    strategyLockDuration: number; // Minimum steps to hold a strategy
}

export const DEFAULT_META_CONFIG: MetaControllerConfig = {
    inputWindowSize: 16,
    hiddenSize: 32,              // Reverted to 16 (32 was too chaotic)
    outputSize: 5,
    targetOutputSize: 512,


    shortTermLR: 0.005,          // Reduced from 0.01 for more conservative learning
    longTermLR: 0.001,
    gateLR: 0.005,
    targetLR: 0.001,
    updateInterval: 1,           // Eval every step (Driver is always watching)
    warmupSteps: 50,             // Reduced from 200 to 50 - wake up faster!
    inputFeaturesSize: 10,       // 10 FEATURES: 8 spectral + 2 proprioceptive (LR, Density)
    contextSize: 8,
    deltaScale: 0.005,
    targetBlend: 0.3,
    strategyLockDuration: 100    // Minimum steps to hold a strategy before switching
};

export interface MetaControllerState {
    shortTermState: LSTMState;
    longTermState: LSTMState;
    gateWeights: Float32Array;
    shortTermWeights: LSTMWeights;
    longTermWeights: LSTMWeights;



    // REINFORCE Cache
    lastInput: Float32Array | null;
    lastShortOut: Float32Array | null;
    lastLongOut: Float32Array | null;
    lastGate: number;


    strategyWeights: Float32Array;            // hidden -> 5 strategies
    lastSelectedStrategy: number;             // Index of selected strategy
    lastStrategyChangeStep: number;           // Step when strategy was last changed
    strategyEligibility: Float32Array;        // Eligibility traces for strategy selection
}

/**
 * DDL-Inspired Structural Action
 * Instead of hardcoded thresholds, the LSTM learns when/how to modify structure.
 */
export interface StructuralAction {
    beta: number;           // [0, 2] Magnitude of structural change (0=skip, 1=project/prune, 2=reflect/grow)
    attention: number[];    // [N] Softmax attention over neurons - which to target
    actionBias: number;     // [-1, 1] Prune (-1) vs Grow (+1) preference
    hyperparams: {          // Still output hyperparams for weight learning
        leak: number;
        spectral: number;
        inputScale: number;
        learningRate: number;
        lvGrowth: number;
        lvDecay: number;
        outputGain: number;
        smoothingFactor: number;
    };
}

export class BicameralMetaController {
    private config: MetaControllerConfig;
    private state: MetaControllerState;
    private errorHistory: number[] = [];
    private stepCounter: number = 0;



    // ELIGIBILITY TRACE DECAY
    // 0.8 = ~5-10 step memory, 0.95 = ~50 step memory
    private lambda: number = 0.95;  // Increased for longer-horizon credit assignment

    // REWARD BASELINE (Variance Reduction)
    // Track running average of rewards so we only update on ABOVE-AVERAGE performance
    private rewardBaseline: number = 0;
    private rewardBaselineAlpha: number = 0.01; // Slow update for stable baseline

    constructor(config: Partial<MetaControllerConfig> = {}) {
        this.config = { ...DEFAULT_META_CONFIG, ...config };
        this.state = this.initializeState();
    }

    private initializeState(): MetaControllerState {
        const { inputFeaturesSize, hiddenSize } = this.config;



        return {
            shortTermState: { hidden: new Float32Array(hiddenSize), cell: new Float32Array(hiddenSize) },
            longTermState: { hidden: new Float32Array(hiddenSize), cell: new Float32Array(hiddenSize) },

            gateWeights: (() => {
                const w = new Float32Array(2 * hiddenSize + 1);
                for (let i = 0; i < w.length; i++) w[i] = (Math.random() * 2 - 1) * 0.1;
                return w;
            })(),

            // Input weights now map Features -> Hidden
            shortTermWeights: initLSTMWeights(inputFeaturesSize, hiddenSize, 0.5),
            longTermWeights: initLSTMWeights(inputFeaturesSize, hiddenSize, 0.2),

            lastInput: null,
            lastShortOut: null,
            lastLongOut: null,
            lastGate: 0.5,

            // STRATEGY SELECTION WEIGHTS: hidden -> 5 strategies
            strategyWeights: new Float32Array(hiddenSize * NUM_STRATEGIES).fill(0),
            lastSelectedStrategy: 0,  // Default to EXPLORE
            lastStrategyChangeStep: 0,
            strategyEligibility: new Float32Array(NUM_STRATEGIES).fill(0)
        };
    }


    public getConfig(): MetaControllerConfig {
        return this.config;
    }

    public recordError(error: number): void {
        this.errorHistory.push(error);
        if (this.errorHistory.length > this.config.inputWindowSize) {
            this.errorHistory.shift();
        }
    }

    // [DEAD CODE REMOVED: stepRegime & reward]

    // [DEAD CODE REMOVED: stepRegime, reward, stepIntelligentMutation]


    public getState(): { shortTermActivity: number; longTermActivity: number; gate: number; } {
        const shortH = this.state.shortTermState.hidden;
        const longH = this.state.longTermState.hidden;
        let shortAct = 0, longAct = 0;
        for (let i = 0; i < shortH.length; i++) {
            shortAct += Math.abs(shortH[i]);
            longAct += Math.abs(longH[i]);
        }
        return {
            shortTermActivity: shortAct / shortH.length,
            longTermActivity: longAct / longH.length,
            gate: this.state.lastGate,
        };
    }

    /**
     * STRATEGY SELECTOR
     * Instead of outputting 8 continuous hyperparameters, select one of 5 discrete strategies.
     * The LSTM learns WHEN to use each strategy, not HOW to tune parameters.
     */
    public selectStrategy(features: number[], currentLoss: number): {
        strategyId: number;
        strategyName: StrategyName;
        params: typeof STRATEGIES.EXPLORE;
        shortTermActivity: number;
        longTermActivity: number;
        gate: number;
    } {
        const { hiddenSize, warmupSteps } = this.config;
        this.stepCounter++;

        // Run LSTM to get blended hidden state
        const input = new Float32Array(features);
        const shortHidden = lstmStep(
            input,
            this.state.shortTermState,
            this.state.shortTermWeights,
            hiddenSize
        );
        this.state.shortTermState = shortHidden; // Update state!

        const longHidden = lstmStep(
            input,
            this.state.longTermState,
            this.state.longTermWeights,
            hiddenSize
        );
        this.state.longTermState = longHidden; // Update state!

        // Compute gate (blend between short-term and long-term)
        let gateSum = 0;
        for (let i = 0; i < hiddenSize; i++) {
            gateSum += this.state.gateWeights[i] * shortHidden.hidden[i];
            gateSum += this.state.gateWeights[hiddenSize + i] * longHidden.hidden[i];
        }
        gateSum += this.state.gateWeights[2 * hiddenSize]; // Bias
        const gate = sigmoid(gateSum);
        this.state.lastGate = gate;

        // Blend hidden states
        const blendedHidden = new Float32Array(hiddenSize);
        let shortAct = 0;
        let longAct = 0;
        for (let i = 0; i < hiddenSize; i++) {
            blendedHidden[i] = gate * shortHidden.hidden[i] + (1 - gate) * longHidden.hidden[i];
            shortAct += Math.abs(shortHidden.hidden[i]);
            longAct += Math.abs(longHidden.hidden[i]);
        }
        const shortActVal = shortAct / hiddenSize;
        const longActVal = longAct / hiddenSize;

        // Cache for REINFORCE
        this.state.lastInput = input;
        this.state.lastShortOut = new Float32Array(shortHidden.hidden);
        this.state.lastLongOut = new Float32Array(longHidden.hidden);

        // During warmup: return default EXPLOIT strategy (safe and stable)
        if (this.stepCounter < warmupSteps) {
            return {
                strategyId: 1,
                strategyName: 'EXPLOIT',
                params: STRATEGIES.EXPLOIT,
                shortTermActivity: shortActVal,
                longTermActivity: longActVal,
                gate: this.state.lastGate
            };
        }

        const { strategyLockDuration } = this.config;

        // PANIC OVERRIDE: If loss is dangerously high, force STABILIZE immediately.
        // This bypasses the persistence lock.
        if (currentLoss > 0.5) {
            // Force STABILIZE (Id 2) to stop the bleeding
            this.state.lastSelectedStrategy = 2;
            this.state.lastStrategyChangeStep = this.stepCounter; // Reset lock to this new strategy

            return {
                strategyId: 2,
                strategyName: 'STABILIZE',
                params: STRATEGIES.STABILIZE,
                shortTermActivity: shortActVal,
                longTermActivity: longActVal,
                gate: this.state.lastGate
            };
        }

        // STRATEGY PERSISTENCE: Check if we are locked into the current strategy
        if (this.stepCounter - this.state.lastStrategyChangeStep < strategyLockDuration) {
            const currentId = this.state.lastSelectedStrategy;
            const currentName = STRATEGY_LIST[currentId];

            // Still update traces so learning continues, but ACTION is locked
            for (let s = 0; s < NUM_STRATEGIES; s++) {
                this.state.strategyEligibility[s] *= this.lambda;
                if (s === currentId) {
                    this.state.strategyEligibility[s] += 1.0;
                }
            }

            return {
                strategyId: currentId,
                strategyName: currentName,
                params: STRATEGIES[currentName],
                shortTermActivity: shortActVal,
                longTermActivity: longActVal,
                gate: this.state.lastGate
            };
        }

        // --- TIME TO CHOOSE NEW STRATEGY ---

        // --- TIME TO CHOOSE NEW STRATEGY ---
        // HEURISTIC OVERRIDES: Don't let the LSTM do stupid things
        // If we are converged (loss < 0.02), force CONSOLIDATE or EXPLOIT
        // If we are failing badly (loss > 0.3), favor RESET/EXPLORE

        // We need access to loss here. Since selectStrategy features includes recentMSE (usually feature[9]), we can use it.
        // Or we can rely on the LSTM to learn this. BUT, to be a "Brain Surgeon", we should guide it.

        // Compute logits for each strategy
        const logits: number[] = [];
        for (let s = 0; s < NUM_STRATEGIES; s++) {
            let sum = 0;
            for (let i = 0; i < hiddenSize; i++) {
                sum += this.state.strategyWeights[i * NUM_STRATEGIES + s] * blendedHidden[i];
            }
            logits.push(sum);
        }

        // Softmax with LOWER temperature for more exploitation (less random jumping)
        const temperature = 0.3; // Much lower than 1.0 -> prefers high-logit options
        const probs = softmax(logits.map(l => l / temperature));

        // Sample from distribution
        let selected = 0;
        const rand = Math.random();
        let cumulative = 0;
        for (let s = 0; s < NUM_STRATEGIES; s++) {
            cumulative += probs[s];
            if (rand < cumulative) {
                selected = s;
                break;
            }
        }

        // Update Persistence Tracker
        this.state.lastSelectedStrategy = selected;
        this.state.lastStrategyChangeStep = this.stepCounter;
        for (let s = 0; s < NUM_STRATEGIES; s++) {
            // Decay existing eligibility
            this.state.strategyEligibility[s] *= this.lambda;
            // Add new eligibility for selected action (policy gradient style)
            if (s === selected) {
                this.state.strategyEligibility[s] += 1.0;
            }
        }

        const strategyName = STRATEGY_LIST[selected];
        const params = STRATEGIES[strategyName];

        return {
            strategyId: selected,
            strategyName,
            params,
            shortTermActivity: shortActVal,
            longTermActivity: longActVal,
            gate: this.state.lastGate
        };
    }

    /**
     * REINFORCE for Strategy Selection
     * Credit/blame the selected strategy based on delayed reward
     */
    public rewardStrategy(reward: number): void {
        const { shortTermLR, hiddenSize } = this.config;
        const clampedReward = Math.max(-1.0, Math.min(1.0, reward));

        // Update baseline
        this.rewardBaseline = this.rewardBaseline * (1 - this.rewardBaselineAlpha) + clampedReward * this.rewardBaselineAlpha;
        const advantage = clampedReward - this.rewardBaseline;

        // Skip if advantage is too small
        if (Math.abs(advantage) < 0.02) return;

        // Get cached hidden state
        const shortH = this.state.lastShortOut;
        const longH = this.state.lastLongOut;
        if (!shortH || !longH) return;

        const gate = this.state.lastGate;
        const blendedHidden = new Float32Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            blendedHidden[i] = gate * shortH[i] + (1 - gate) * longH[i];
        }

        // Update strategy weights using policy gradient
        // Increase weight for selected strategy if reward was positive
        const lr = shortTermLR * 0.1; // Slower learning for strategy selection

        for (let s = 0; s < NUM_STRATEGIES; s++) {
            const eligibility = this.state.strategyEligibility[s];
            for (let i = 0; i < hiddenSize; i++) {
                const idx = i * NUM_STRATEGIES + s;
                this.state.strategyWeights[idx] += lr * advantage * eligibility * blendedHidden[i];
                // Clamp
                this.state.strategyWeights[idx] = Math.max(-3, Math.min(3, this.state.strategyWeights[idx]));
            }
        }
    }

    // [DEAD CODE REMOVED: stepStructural & rewardStructural]
    // [ORPHAN CODE REMOVED]

    // [DEAD CODE REMOVED: Legacy Continuous Controller & Duplicates]

    public reset(): void {
        this.state = this.initializeState();
        this.errorHistory = [];
        this.stepCounter = 0;
    }

    /**
     * Resets only the short-term memory (fast weights) of the LSTM.
     * Call this when the environment changes abruptly (e.g., Rewiring/Mitosis)
     * so the Driver doesn't react based on outdated momentum.
     */
    public resetShortTermMemory(): void {
        this.state.shortTermState.hidden.fill(0);
        this.state.shortTermState.cell.fill(0);
        // Also clear last output memory to prevent momentum continuity
        this.state.lastShortOut = new Float32Array(this.config.hiddenSize).fill(0);
        // Clear eligibility traces - the world changed, old responsibilities don't apply
        this.state.strategyEligibility.fill(0);
    }

    /**
     * DEBUG API for UI
     */
    public getLSTMDebugState(): any {
        return {
            shortTermHidden: Array.from(this.state.shortTermState.hidden),
            shortTermCell: Array.from(this.state.shortTermState.cell),
            longTermHidden: Array.from(this.state.longTermState.hidden),
            longTermCell: Array.from(this.state.longTermState.cell),
            gate: this.state.lastGate,
            eligibilityTraces: Array.from(this.state.strategyEligibility), // Maps to UI expectation
            outputWeights: Array.from(this.state.strategyWeights),
            rawOutputs: null,
            hiddenSize: this.config.hiddenSize,
            numOutputs: 5, // NUM_STRATEGIES
            health: {
                nanCount: 0,
                maxTrace: Math.max(...this.state.strategyEligibility),
                maxCell: 0,
                maxWeight: 0
            }
        };
    }
}
