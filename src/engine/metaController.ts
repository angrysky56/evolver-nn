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

        // Gate activations
        const i_gate = sigmoid(i_pre);
        const f_gate = sigmoid(f_pre);
        const c_gate = tanh(c_pre);
        const o_gate = sigmoid(o_pre);

        // New cell state and hidden state
        newCell[j] = f_gate * c[j] + i_gate * c_gate;
        newHidden[j] = o_gate * tanh(newCell[j]);
    }

    return { hidden: newHidden, cell: newCell };
}

// ==================== BICAMERAL CONTROLLER ====================

export interface MetaControllerConfig {
    inputWindowSize: number;     // Number of error samples to consider
    hiddenSize: number;          // LSTM hidden dimension
    outputSize: number;          // 3 (leak, spectral, inputScale)
    targetOutputSize: number;    // Max neurons for target output (DiscoRL-style)
    shortTermLR: number;         // Learning rate for short-term updates
    longTermLR: number;          // Learning rate for long-term updates
    gateLR: number;              // Learning rate for gate
    targetLR: number;            // Learning rate for target output weights
    updateInterval: number;      // Only apply deltas every N steps
    warmupSteps: number;         // Don't apply deltas until this many steps
    deltaScale: number;          // Scale factor for output deltas
    targetBlend: number;         // 0-1: how much to blend target updates vs gradient
    contextSize: number;         // Number of context variables (5 parameters)
}

export const DEFAULT_META_CONFIG: MetaControllerConfig = {
    inputWindowSize: 16,
    hiddenSize: 8,
    outputSize: 9, // 7 Params + Direction + Magnitude
    targetOutputSize: 512,       // Matches maxNeurons
    shortTermLR: 0.005,          // Reduced from 0.01 for smoother adaptation
    longTermLR: 0.0001,
    gateLR: 0.0005,
    targetLR: 0.001,
    updateInterval: 32,
    warmupSteps: 100,
    deltaScale: 0.005,
    targetBlend: 0.3,
    contextSize: 8               // 7 Params + Density
};

export interface MetaControllerState {
    shortTermState: LSTMState;
    longTermState: LSTMState;
    gateWeights: Float32Array;  // [2 * hiddenSize + 1] -> scalar gate
    shortTermWeights: LSTMWeights;
    longTermWeights: LSTMWeights;
    outputWeights: Float32Array; // [hiddenSize, outputSize] -> hyperparameter deltas
    targetOutputWeights: Float32Array; // [hiddenSize, targetOutputSize] -> readout targets

    // For REINFORCE
    lastInput: Float32Array | null;
    lastShortOut: Float32Array | null;
    lastLongOut: Float32Array | null;
    lastGate: number;
    lastDeltas: Float32Array | null;
    lastTargets: Float32Array | null;
}

export class BicameralMetaController {
    private config: MetaControllerConfig;
    private state: MetaControllerState;
    private errorHistory: number[] = [];
    private stepCounter: number = 0;

    constructor(config: Partial<MetaControllerConfig> = {}) {
        this.config = { ...DEFAULT_META_CONFIG, ...config };
        this.state = this.initializeState();
    }

    private initializeState(): MetaControllerState {
        const { inputWindowSize, hiddenSize, outputSize, targetOutputSize } = this.config;

        return {
            shortTermState: {
                hidden: new Float32Array(hiddenSize),
                cell: new Float32Array(hiddenSize),
            },
            longTermState: {
                hidden: new Float32Array(hiddenSize),
                cell: new Float32Array(hiddenSize),
            },
            gateWeights: (() => {
                // 2*hiddenSize inputs (short + long hidden) + 1 bias
                const w = new Float32Array(2 * hiddenSize + 1);
                for (let i = 0; i < w.length; i++) w[i] = (Math.random() * 2 - 1) * 0.1;
                return w;
            })(),
            shortTermWeights: initLSTMWeights(inputWindowSize + this.config.contextSize, hiddenSize, 0.15),
            longTermWeights: initLSTMWeights(inputWindowSize + this.config.contextSize, hiddenSize, 0.08),
            outputWeights: (() => {
                const w = new Float32Array(hiddenSize * outputSize);
                for (let i = 0; i < w.length; i++) w[i] = (Math.random() * 2 - 1) * 0.05;
                return w;
            })(),
            targetOutputWeights: (() => {
                // [hiddenSize, targetOutputSize] -> target readout for each neuron
                const w = new Float32Array(hiddenSize * targetOutputSize);
                for (let i = 0; i < w.length; i++) w[i] = (Math.random() * 2 - 1) * 0.01;
                return w;
            })(),
            lastInput: null,
            lastShortOut: null,
            lastLongOut: null,
            lastGate: 0.5,
            lastDeltas: null,
            lastTargets: null,
        };
    }

    /**
     * Get the current configuration
     */
    public getConfig(): MetaControllerConfig {
        return this.config;
    }

    /**
     * Push a new error sample into the history
     */
    public recordError(error: number): void {
        this.errorHistory.push(error);
        if (this.errorHistory.length > this.config.inputWindowSize) {
            this.errorHistory.shift();
        }
    }

    /**
     * Compute hyperparameter deltas using the Bicameral controller
     * Returns: { leak: Δleak, spectral: Δspectral, inputScale: ΔinputScale }
     */
    public step(): { leak: number; spectral: number; inputScale: number } {
        const { hiddenSize, outputSize, inputWindowSize, updateInterval, warmupSteps, deltaScale } = this.config;

        this.stepCounter++;

        // Prepare input: normalize error history
        const input = new Float32Array(inputWindowSize);
        const histLen = this.errorHistory.length;

        if (histLen === 0) {
            return { leak: 0, spectral: 0, inputScale: 0 };
        }

        // Normalize: z-score the error history
        let mean = 0, std = 0;
        for (let i = 0; i < histLen; i++) mean += this.errorHistory[i];
        mean /= histLen;
        for (let i = 0; i < histLen; i++) std += (this.errorHistory[i] - mean) ** 2;
        std = Math.sqrt(std / histLen) + 1e-8;

        for (let i = 0; i < inputWindowSize; i++) {
            const idx = histLen - inputWindowSize + i;
            if (idx >= 0) {
                input[i] = (this.errorHistory[idx] - mean) / std;
            } else {
                input[i] = 0; // Padding
            }
        }

        // Short-Term LSTM step
        this.state.shortTermState = lstmStep(
            input,
            this.state.shortTermState,
            this.state.shortTermWeights,
            hiddenSize
        );

        // Long-Term LSTM step
        this.state.longTermState = lstmStep(
            input,
            this.state.longTermState,
            this.state.longTermWeights,
            hiddenSize
        );

        const shortHidden = this.state.shortTermState.hidden;
        const longHidden = this.state.longTermState.hidden;

        // Compute soft gate: σ(W_g · [short, long] + b)
        let gateSum = this.state.gateWeights[2 * hiddenSize]; // Bias
        for (let i = 0; i < hiddenSize; i++) {
            gateSum += this.state.gateWeights[i] * shortHidden[i];
            gateSum += this.state.gateWeights[hiddenSize + i] * longHidden[i];
        }
        const gate = sigmoid(gateSum); // 0-1: 1 = use short-term, 0 = use long-term

        // Blend hidden states
        const blendedHidden = new Float32Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            blendedHidden[i] = gate * shortHidden[i] + (1 - gate) * longHidden[i];
        }

        // Output layer: hiddenSize -> outputSize (3 deltas)
        const deltas = new Float32Array(outputSize);

        // Only produce non-zero deltas every updateInterval steps, after warmup
        const shouldApply = this.stepCounter >= warmupSteps && (this.stepCounter % updateInterval === 0);

        if (shouldApply) {
            for (let j = 0; j < outputSize; j++) {
                let sum = 0;
                for (let i = 0; i < hiddenSize; i++) {
                    sum += this.state.outputWeights[i * outputSize + j] * blendedHidden[i];
                }
                deltas[j] = tanh(sum) * deltaScale;
            }
        }

        // Cache for REINFORCE update
        this.state.lastInput = input;
        this.state.lastShortOut = new Float32Array(shortHidden);
        this.state.lastLongOut = new Float32Array(longHidden);
        this.state.lastGate = gate;
        this.state.lastDeltas = deltas;

        return {
            leak: deltas[0],
            spectral: deltas[1],
            inputScale: deltas[2],
        };
    }

    /**
     * Intelligent mutation output.
     * LSTM learns WHICH param to adjust, WHICH direction, and HOW much.
     * Based on error patterns - not random!
     *
     * @returns { paramIndex: 0-2, direction: -1 to 1, magnitude: 0-1, shouldMutate: boolean }
     */
    public stepIntelligentMutation(context: number[] = [0.1, 0.5, 1.0, 0.01, 0.1, 0.2, 0.01, 0.2]): {
        paramIndex: number;
        direction: number;
        magnitude: number;
        shouldMutate: boolean;
    } {
        const { hiddenSize, inputWindowSize, contextSize, warmupSteps } = this.config;

        this.stepCounter++;

        // Default: no mutation
        const defaultResult = { paramIndex: 0, direction: 0, magnitude: 0, shouldMutate: false };

        // Warmup Check: Wait for system to stabilize before tuning
        if (this.stepCounter < warmupSteps) {
            return defaultResult;
        }

        // Prepare normalized error history + Context
        const totalInputSize = inputWindowSize + contextSize;
        const input = new Float32Array(totalInputSize);
        const histLen = this.errorHistory.length;

        if (histLen === 0) {
            return defaultResult;
        }

        let mean = 0, std = 0;
        for (let i = 0; i < histLen; i++) mean += this.errorHistory[i];
        mean /= histLen;
        for (let i = 0; i < histLen; i++) std += (this.errorHistory[i] - mean) ** 2;
        std = Math.sqrt(std / histLen) + 1e-8;

        // Fill History part (0 to 15)
        for (let i = 0; i < inputWindowSize; i++) {
            const idx = histLen - inputWindowSize + i;
            input[i] = idx >= 0 ? (this.errorHistory[idx] - mean) / std : 0;
        }

        // Fill Context part (16 to 23)
        // Normalize context? Leak (0-1), Spectral (0-2), Input (0-5), LR (0-1), Smooth (0-0.5), Growth (0-1), Decay (0-0.1), Density (0-1)
        if (context.length === contextSize) {
            input[inputWindowSize + 0] = context[0]; // Leak
            input[inputWindowSize + 1] = context[1] * 0.5; // Spectral
            input[inputWindowSize + 2] = context[2] * 0.2; // Input
            input[inputWindowSize + 3] = context[3]; // LR
            input[inputWindowSize + 4] = context[4] * 2.0; // Smooth
            input[inputWindowSize + 5] = context[5] * 2.0; // lvGrowth (0.18 -> 0.36)
            input[inputWindowSize + 6] = context[6] * 50.0; // lvDecay (0.006 -> 0.3)
            input[inputWindowSize + 7] = context[7]; // Density
        }

        // Run both LSTMs
        this.state.shortTermState = lstmStep(
            input,
            this.state.shortTermState,
            this.state.shortTermWeights,
            hiddenSize
        );

        this.state.longTermState = lstmStep(
            input,
            this.state.longTermState,
            this.state.longTermWeights,
            hiddenSize
        );

        const shortHidden = this.state.shortTermState.hidden;
        const longHidden = this.state.longTermState.hidden;

        // Gate blend
        let gateSum = this.state.gateWeights[2 * hiddenSize];
        for (let i = 0; i < hiddenSize; i++) {
            gateSum += this.state.gateWeights[i] * shortHidden[i];
            gateSum += this.state.gateWeights[hiddenSize + i] * longHidden[i];
        }
        const gate = sigmoid(gateSum);
        this.state.lastGate = gate;

        // Blend hidden states
        const blendedHidden = new Float32Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            blendedHidden[i] = gate * shortHidden[i] + (1 - gate) * longHidden[i];
        }

        // Compute 3 outputs from output weights:
        // output[0] = param selection logit (0=leak, 1=spectral, 2=inputScale)
        // output[7] = direction
        // output[8] = magnitude
        // Output Map:
        // 0-6: Logits for [leak, spectral, inputScale, learningRate, smoothingFactor, lvGrowth, lvDecay]
        // 7: Direction
        // 8: Magnitude

        const outputs = new Float32Array(9);
        for (let j = 0; j < 9; j++) {
            let sum = 0;
            for (let i = 0; i < hiddenSize; i++) {
                sum += this.state.outputWeights[i * this.config.outputSize + j] * blendedHidden[i];
            }
            outputs[j] = sum;
        }

        // Param selection: Softmax over the first 7 logits
        const paramLogits = [
            outputs[0], outputs[1], outputs[2], outputs[3],
            outputs[4], outputs[5], outputs[6]
        ];
        const paramProbs = softmax(paramLogits);

        // Sample param based on probabilities
        let paramIndex = 0;
        const r = Math.random();
        let cumulative = 0;
        for (let i = 0; i < 7; i++) {
            cumulative += paramProbs[i];
            if (r < cumulative) {
                paramIndex = i;
                break;
            }
        }

        // Direction: Tanh (-1 to 1) from output 7
        const direction = tanh(outputs[7]);

        // Magnitude: Sigmoid (0 to 1) from output 8
        const magnitude = sigmoid(outputs[8]);

        return {
            shouldMutate: true,
            paramIndex,
            direction,
            magnitude
        };

        return {
            shouldMutate: true,
            paramIndex,
            direction,
            magnitude
        };
    }

    // (End of stepIntelligentMutation)

    /**
     * Apply reward signal (REINFORCE-style)
     * @param reward Positive if loss improved, negative if worsened
     */
    public reward(reward: number): void {
        if (!this.state.lastDeltas) return;

        // Skip reward if deltas were zero (outside update interval)
        const hasNonZeroDelta = this.state.lastDeltas.some(d => d !== 0);
        if (!hasNonZeroDelta) {
            this.state.lastInput = null;
            this.state.lastDeltas = null;
            return;
        }

        const { hiddenSize, outputSize, shortTermLR, gateLR } = this.config;
        // Gentler reward scaling - don't multiply by 100
        const clampedReward = Math.max(-0.1, Math.min(0.1, reward * 10));

        // Update output weights: W += lr * reward * h * sign(delta)
        const blendedHidden = new Float32Array(hiddenSize);
        const gate = this.state.lastGate;
        const shortH = this.state.lastShortOut!;
        const longH = this.state.lastLongOut!;

        for (let i = 0; i < hiddenSize; i++) {
            blendedHidden[i] = gate * shortH[i] + (1 - gate) * longH[i];
        }

        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < outputSize; j++) {
                const grad = blendedHidden[i] * Math.sign(this.state.lastDeltas![j]);
                this.state.outputWeights[i * outputSize + j] += shortTermLR * clampedReward * grad;
                // Weight clipping
                this.state.outputWeights[i * outputSize + j] = Math.max(-2, Math.min(2, this.state.outputWeights[i * outputSize + j]));
            }
        }

        // Update gate weights: encourage gate direction that was used if reward positive
        const gateDirection = clampedReward > 0 ? 1 : -1;
        for (let i = 0; i < hiddenSize; i++) {
            this.state.gateWeights[i] += gateLR * clampedReward * gateDirection * shortH[i] * 0.1;
            this.state.gateWeights[hiddenSize + i] += gateLR * clampedReward * gateDirection * longH[i] * 0.1;
            // Weight clipping
            this.state.gateWeights[i] = Math.max(-2, Math.min(2, this.state.gateWeights[i]));
            this.state.gateWeights[hiddenSize + i] = Math.max(-2, Math.min(2, this.state.gateWeights[hiddenSize + i]));
        }

        // Clear cache
        this.state.lastInput = null;
        this.state.lastDeltas = null;
    }

    /**
     * Get current controller state for visualization
     */
    public getState(): {
        shortTermActivity: number;
        longTermActivity: number;
        gate: number;
    } {
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
     * DiscoRL-style: Produce target readout values for the network to move toward.
     * The meta-network "discovers" what the readout weights should be.
     *
     * @param activations Current neuron activations [N]
     * @param currentReadout Current readout weights [N]
     * @param N Number of active neurons
     * @returns Target readout values [N]
     */
    public stepTargets(activations: Float32Array, currentReadout: Float32Array, N: number): Float32Array {
        const { hiddenSize, targetOutputSize } = this.config;

        const targets = new Float32Array(N);

        // Use the blended hidden state from the last step() call
        const gate = this.state.lastGate;
        const shortH = this.state.shortTermState.hidden;
        const longH = this.state.longTermState.hidden;

        // Blend short-term and long-term hidden states
        const blendedHidden = new Float32Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            blendedHidden[i] = gate * shortH[i] + (1 - gate) * longH[i];
        }

        // Compute targets for each neuron
        // target[j] = tanh(W_target[j] · blendedHidden) + currentReadout[j]
        // This means: "here's a delta to apply to the current readout"
        for (let j = 0; j < N && j < targetOutputSize; j++) {
            let sum = 0;
            for (let i = 0; i < hiddenSize; i++) {
                sum += this.state.targetOutputWeights[i * targetOutputSize + j] * blendedHidden[i];
            }
            // Output is a delta: target = current + small adjustment
            // Scale by activation magnitude to focus on active neurons
            const activationScale = Math.abs(activations[j]) + 0.1;
            targets[j] = currentReadout[j] + tanh(sum) * 0.1 * activationScale;
        }

        this.state.lastTargets = targets;
        return targets;
    }

    /**
     * Update target output weights based on whether targets helped
     * @param improvement -1 to 1: how much loss improved after applying targets
     */
    public rewardTargets(improvement: number): void {
        if (!this.state.lastTargets) return;

        const { hiddenSize, targetOutputSize, targetLR } = this.config;
        const clampedImprovement = Math.max(-0.1, Math.min(0.1, improvement));

        const gate = this.state.lastGate;
        const shortH = this.state.shortTermState.hidden;
        const longH = this.state.longTermState.hidden;

        // Reconstruct blended hidden
        const blendedHidden = new Float32Array(hiddenSize);
        for (let i = 0; i < hiddenSize; i++) {
            blendedHidden[i] = gate * shortH[i] + (1 - gate) * longH[i];
        }

        // Update target output weights: reinforce if improvement was positive
        const N = this.state.lastTargets.length;
        for (let j = 0; j < N && j < targetOutputSize; j++) {
            const targetVal = this.state.lastTargets[j];
            for (let i = 0; i < hiddenSize; i++) {
                const idx = i * targetOutputSize + j;
                // REINFORCE: increase weights that led to good targets
                this.state.targetOutputWeights[idx] += targetLR * clampedImprovement * blendedHidden[i] * Math.sign(targetVal);
                // Clip to prevent explosion
                this.state.targetOutputWeights[idx] = Math.max(-1, Math.min(1, this.state.targetOutputWeights[idx]));
            }
        }

        this.state.lastTargets = null;
    }

    /**
     * Reset the controller state
     */
    public reset(): void {
        this.state = this.initializeState();
        this.errorHistory = [];
        this.stepCounter = 0;
    }
}
