/**
 * EvolverEngine - Pure reservoir computing simulation
 * Takes input/target values, returns predictions, handles learning
 * NO React, NO data generation, NO UI
 */

// Network Constants
export const MAX_NEURONS = 512;
export const INITIAL_NEURONS = 8;
const PRUNE_THRESHOLD = 0.015;
const LEARNING_RATE = 0.02;
const DECAY_RATE = 0.001;

// Adaptation Constants
const PATIENCE_LIMIT = 64;
const SOLVED_THRESHOLD = 0.01;
const UNLOCK_THRESHOLD = 0.015;
const LEARNING_SLOPE_THRESHOLD = -0.0005;
const LV_GROWTH = 0.02;
const LV_DECAY = 0.02;

// Optimizer Constants
const OPT_INTERVAL = 16;
const OPT_MUTATION_RATE = 0.05;

export interface SeedParams {
    leak: number;
    spectral: number;
    inputScale: number;
}

export interface SimMetrics {
    loss: number;
    avgLoss: number;
    activeConnections: number;
    targetConnections: number;
    regrown: number;
    prediction: number;
    target: number;
    neuronCount: number;
    patience: number;
    regressionSlope: number;
    adaptationStatus: string;
    dna: SeedParams;
}

export interface NetworkState {
    activations: Float32Array;
    weights: Float32Array;
    inputWeights: Float32Array;
    currentSize: number;
    liveHyperparams: SeedParams;
}

export interface WaveformPoint {
    target: number;
    prediction: number;
}

export class EvolverEngine {
    // Network arrays
    private activations = new Float32Array(MAX_NEURONS);
    private prevActivations = new Float32Array(MAX_NEURONS);
    private weights = new Float32Array(MAX_NEURONS * MAX_NEURONS);
    private readout = new Float32Array(MAX_NEURONS);
    private inputWeights = new Float32Array(MAX_NEURONS);

    // Network state
    private currentSize = INITIAL_NEURONS;
    private totalRegrown = 0;
    private currentTargetDensity = 0.2;
    private patienceCounter = 0;
    private regressionSlope = 0;
    private isLocked = false;
    private lossWindow: number[] = [];

    // Hyperparameters
    private liveHyperparams: SeedParams = { leak: 0.5, spectral: 0.9, inputScale: 1.0 };

    // Optimizer
    private optimizer = {
        timer: 0,
        baselineLoss: 1.0,
        currentLossAcc: 0,
        samples: 0,
        params: {
            leak: { step: 0.05, bestVal: 0.5 },
            spectral: { step: 0.05, bestVal: 0.9 },
            inputScale: { step: 0.1, bestVal: 1.0 }
        },
        lastParam: null as string | null,
        lastDelta: 0
    };

    // Tracking
    private waveformHistory: WaveformPoint[] = [];
    private avgLossBuffer = 1.0;
    private stepCount = 0;

    constructor(seedParams?: SeedParams) {
        this.initialize(seedParams);
    }

    initialize(seedParams?: SeedParams): void {
        const params = seedParams || { leak: 0.8, spectral: 0.95, inputScale: 1.0 };

        this.activations.fill(0);
        this.prevActivations.fill(0);
        this.liveHyperparams = { ...params };
        this.currentSize = INITIAL_NEURONS;
        this.currentTargetDensity = 0.2;
        this.patienceCounter = 0;
        this.regressionSlope = 0;
        this.isLocked = false;
        this.lossWindow = [];

        this.optimizer = {
            timer: 0,
            baselineLoss: 1.0,
            currentLossAcc: 0,
            samples: 0,
            params: {
                leak: { step: 0.05, bestVal: params.leak },
                spectral: { step: 0.05, bestVal: params.spectral },
                inputScale: { step: 0.1, bestVal: params.inputScale }
            },
            lastParam: null,
            lastDelta: 0
        };

        for (let i = 0; i < MAX_NEURONS; i++) {
            this.inputWeights[i] = (Math.random() * 2 - 1) * params.inputScale;
            this.readout[i] = 0;
        }

        this.weights.fill(0);
        for (let i = 0; i < INITIAL_NEURONS; i++) {
            for (let j = 0; j < INITIAL_NEURONS; j++) {
                if (Math.random() < 0.2) {
                    this.weights[i * MAX_NEURONS + j] = (Math.random() * 2 - 1) * params.spectral;
                }
            }
        }

        this.totalRegrown = 0;
        this.avgLossBuffer = 1.0;
        this.stepCount = 0;
        this.waveformHistory = new Array(150).fill({ target: 0, prediction: 0 });
    }

    private countConnections(): number {
        let count = 0;
        for (let i = 0; i < this.currentSize; i++) {
            for (let j = 0; j < this.currentSize; j++) {
                if (this.weights[i * MAX_NEURONS + j] !== 0) count++;
            }
        }
        return count;
    }

    private performMitosis(): boolean {
        if (this.currentSize >= MAX_NEURONS) return false;

        const newIdx = this.currentSize;
        const newSize = this.currentSize + 1;
        const spectral = this.liveHyperparams.spectral;

        for (let i = 0; i < newSize; i++) {
            if (Math.random() < this.currentTargetDensity) {
                this.weights[newIdx * MAX_NEURONS + i] = (Math.random() * 2 - 1) * spectral;
            }
            if (Math.random() < this.currentTargetDensity) {
                this.weights[i * MAX_NEURONS + newIdx] = (Math.random() * 2 - 1) * spectral;
            }
        }
        this.currentSize = newSize;
        return true;
    }

    /**
     * Run one simulation step with provided input and target
     */
    step(inputVal: number, targetVal: number): SimMetrics {
        const N = this.currentSize;
        const dna = this.liveHyperparams;

        // 1. Reservoir Update
        this.prevActivations.set(this.activations);
        const leak = dna.leak;

        for (let i = 0; i < N; i++) {
            let sum = 0;
            for (let j = 0; j < N; j++) {
                const w = this.weights[i * MAX_NEURONS + j];
                if (w !== 0) sum += w * this.prevActivations[j];
            }
            sum += this.inputWeights[i] * inputVal * dna.inputScale;
            const newState = Math.tanh(sum);
            this.activations[i] = (1 - leak) * this.prevActivations[i] + leak * newState;
        }

        // 2. Readout
        let prediction = 0;
        for (let i = 0; i < N; i++) {
            prediction += this.activations[i] * this.readout[i];
        }

        // 3. Learning
        const error = targetVal - prediction;
        for (let i = 0; i < N; i++) {
            this.readout[i] += LEARNING_RATE * error * this.activations[i];
        }
        const absError = Math.abs(error);
        this.avgLossBuffer = this.avgLossBuffer * 0.99 + absError * 0.01;

        // 4. Hyperparameter Optimizer
        const opt = this.optimizer;
        opt.timer++;
        opt.currentLossAcc += absError;
        opt.samples++;

        if (opt.timer >= OPT_INTERVAL) {
            const currentAvg = opt.currentLossAcc / opt.samples;

            if (opt.lastParam) {
                const paramState = (opt.params as any)[opt.lastParam];
                if (currentAvg < opt.baselineLoss) {
                    opt.baselineLoss = currentAvg;
                    paramState.step = Math.min(0.2, paramState.step * 1.1);
                } else {
                    (this.liveHyperparams as any)[opt.lastParam] -= opt.lastDelta;
                    paramState.step *= 0.5;
                }
            } else {
                opt.baselineLoss = currentAvg;
            }

            const paramKeys = ['leak', 'spectral', 'inputScale'];
            const targetParam = paramKeys[Math.floor(Math.random() * paramKeys.length)];
            const direction = Math.random() > 0.5 ? 1 : -1;
            const delta = direction * ((this.liveHyperparams as any)[targetParam] * OPT_MUTATION_RATE);

            (this.liveHyperparams as any)[targetParam] += delta;
            if (targetParam === 'leak') {
                this.liveHyperparams.leak = Math.max(0.01, Math.min(1.0, this.liveHyperparams.leak));
            }

            opt.lastParam = targetParam;
            opt.lastDelta = delta;
            opt.timer = 0;
            opt.currentLossAcc = 0;
            opt.samples = 0;
        }

        // 5. Structural Adaptation
        this.lossWindow.push(absError);
        if (this.lossWindow.length > 50) this.lossWindow.shift();

        let trendSlope = 0;
        if (this.lossWindow.length >= 50) {
            const start = this.lossWindow.slice(0, 25).reduce((a, b) => a + b, 0);
            const end = this.lossWindow.slice(25).reduce((a, b) => a + b, 0);
            trendSlope = start - end;
        }
        this.regressionSlope = trendSlope;

        if (!this.isLocked && this.avgLossBuffer < SOLVED_THRESHOLD) {
            this.isLocked = true;
            this.patienceCounter = 0;
        } else if (this.isLocked && this.avgLossBuffer > UNLOCK_THRESHOLD) {
            this.isLocked = false;
        }

        let status = 'STABLE';
        if (this.isLocked) {
            status = 'LOCKED';
        } else {
            const isImproving = trendSlope < LEARNING_SLOPE_THRESHOLD;

            if (isImproving) {
                status = 'LEARNING';
                this.patienceCounter = Math.max(0, this.patienceCounter - 1);
            } else {
                status = 'STAGNANT';
                this.patienceCounter++;
            }

            const growthPressure = absError * LV_GROWTH;
            const densityChange = growthPressure - (this.currentTargetDensity * LV_DECAY);
            this.currentTargetDensity = Math.max(0.1, Math.min(0.9, this.currentTargetDensity + densityChange));

            if (this.patienceCounter > PATIENCE_LIMIT) {
                if (this.performMitosis()) {
                    status = 'GROWING';
                    this.patienceCounter = 0;
                }
            }
        }

        const targetConns = Math.floor(N * N * this.currentTargetDensity);

        // 6. Structural Plasticity
        let activeCount = 0;
        const currentDecay = this.isLocked ? 0 : DECAY_RATE;
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                const idx = i * MAX_NEURONS + j;
                if (this.weights[idx] !== 0) {
                    if (currentDecay > 0) {
                        if (this.weights[idx] > 0) {
                            this.weights[idx] -= currentDecay * Math.abs(this.weights[idx]);
                        } else {
                            this.weights[idx] += currentDecay * Math.abs(this.weights[idx]);
                        }
                    }
                    if (Math.abs(this.weights[idx]) < PRUNE_THRESHOLD) {
                        this.weights[idx] = 0;
                    } else {
                        activeCount++;
                    }
                }
            }
        }

        const deficit = targetConns - activeCount;
        if (deficit > 0) {
            status = status === 'GROWING' ? 'GROWING' : (this.isLocked ? 'LOCKED' : 'GROWING_SYNAPSES');
            const connectionsToGrow = Math.ceil(deficit * 0.1) + 1;
            let newConnections = 0;
            let attempts = 0;
            while (newConnections < connectionsToGrow && attempts < 100) {
                attempts++;
                const i = Math.floor(Math.random() * N);
                const j = Math.floor(Math.random() * N);
                const idx = i * MAX_NEURONS + j;
                if (this.weights[idx] === 0) {
                    this.weights[idx] = (Math.random() * 2 - 1) * dna.spectral;
                    newConnections++;
                }
            }
            this.totalRegrown += newConnections;
            activeCount += newConnections;
        }

        this.waveformHistory.push({ target: targetVal, prediction });
        this.waveformHistory.shift();
        this.stepCount++;

        return {
            loss: absError,
            avgLoss: this.avgLossBuffer,
            activeConnections: activeCount,
            targetConnections: targetConns,
            regrown: this.totalRegrown,
            prediction,
            target: targetVal,
            neuronCount: N,
            patience: this.patienceCounter,
            regressionSlope: this.regressionSlope,
            adaptationStatus: status,
            dna: { ...dna }
        };
    }

    getStep(): number {
        return this.stepCount;
    }

    getNetworkState(): NetworkState {
        return {
            activations: this.activations,
            weights: this.weights,
            inputWeights: this.inputWeights,
            currentSize: this.currentSize,
            liveHyperparams: { ...this.liveHyperparams }
        };
    }

    getWaveformHistory(): WaveformPoint[] {
        return this.waveformHistory;
    }

    getInitialMetrics(): SimMetrics {
        return {
            loss: 0,
            avgLoss: 1.0,
            activeConnections: this.countConnections(),
            targetConnections: Math.floor(INITIAL_NEURONS * INITIAL_NEURONS * 0.2),
            regrown: 0,
            prediction: 0,
            target: 0,
            neuronCount: INITIAL_NEURONS,
            patience: 0,
            regressionSlope: 0,
            adaptationStatus: 'STABLE',
            dna: { ...this.liveHyperparams }
        };
    }
}
