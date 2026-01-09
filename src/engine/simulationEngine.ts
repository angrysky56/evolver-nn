import { Task } from '../tasks/tasks';
import { BicameralMetaController } from './metaController';

export const MACKEY_GLASS_TAU = 17;

export interface SimulationConfig {
    maxNeurons: number;
    initialNeurons: number;
    pruneThreshold: number;
    learningRate: number;
    // decayRate: number; // REMOVED: Superseded by lvDecay (Unified System)
    patienceLimit: number;
    solvedThreshold: number;
    unlockThreshold: number;
    learningSlopeThreshold: number;
    lvGrowth: number;
    lvDecay: number; // Unified L-V Competition + Synaptic Weight Decay
}

export const DEFAULT_CONFIG: SimulationConfig = {
    maxNeurons: 1024,
    initialNeurons: 64,
    // High-Energy Config: Fast growth, aggressive pruning
    pruneThreshold: 0.01,
    learningRate: 0.1,        // 10x higher for faster ESN readout learning
    // decayRate is gone.
    patienceLimit: 64,
    solvedThreshold: 0.01,
    unlockThreshold: 0.05,
    learningSlopeThreshold: -0.005,
    lvGrowth: 0.08,           // High Growth
    lvDecay: 0.006,           //
};

export interface NetworkState {
    activations: Float32Array;
    prevActivations: Float32Array;
    weights: Float32Array;
    readout: Float32Array;
    inputWeights: Float32Array;
    neuronTypes: Int8Array; // 1 = Excitatory, -1 = Inhibitory (Dale's Law)

    currentSize: number;
    totalRegrown: number;
    currentTargetDensity: number;
    patienceCounter: number;
    regressionSlope: number;
    isLocked: boolean;
    lossWindow: number[];

    // Lotka-Volterra Dynamics State
    neuronEnergy: Float32Array;  // Per-neuron metabolic energy for adaptive LR
    isWinter: boolean;           // Periodic extinction event flag
    winterTimer: number;         // Steps until winter ends
    epochCounter: number;        // Counts steps for winter scheduling

    liveHyperparams: {
        leak: number;
        spectral: number;
        inputScale: number;
        learningRate: number;
        smoothingFactor: number;
        lvGrowth: number;
        lvDecay: number;
    };

    // Kept for UI compatibility
    checkpointHyperparams: null;

    optimizer: {
        globalStepScale: number; // Added for V2-style adaptive scaling
        timer: number;
        baselineLoss: number;
        currentLossAcc: number;
        samples: number;
        paramKeys: string[];
        currentIdx: number;
        state: string;
        params: any;
        lastTestedVal: number;
        testDirection: number;
        lastParam?: string;
        lastDelta?: number;
    };
}

export interface SimulationMetrics {
    loss: number;
    avgLoss: number;
    activeConnections: number;
    targetConnections: number;
    regrown: number;
    prediction: number;
    target: number;
    neuronCount: number;
    patience: number;
    adaptationStatus: string;
    dna: { leak: number; spectral: number; inputScale: number; learningRate: number; smoothingFactor: number; lvGrowth: number; lvDecay: number };
    metaController: { shortTermActivity: number; longTermActivity: number; gate: number };
}

export class EvolutionaryChaosNetwork {
    private config: SimulationConfig;
    private task: Task;
    private network: NetworkState;
    private dataSeries: number[] = [];
    private avgLossBuffer: number = 1.0;
    private stepCount: number = 0;
    private metaController: BicameralMetaController;
    private prevAvgLoss: number = 1.0;

    constructor(task: Task, config: Partial<SimulationConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.task = task;
        this.metaController = new BicameralMetaController();
        this.network = this.initializeNetwork(); // Moved after metaController init to be safe
        this.initializeData();
    }

    private initializeNetwork(): NetworkState {
        // Define Complete Starting DNA
        // If task.seedParams is missing, we use these explicit defaults.
        // This is the "Birth State" of the network.
        const defaultParams = {
            leak: 0.8,
            spectral: 0.95,
            inputScale: 1.0,
            learningRate: this.config.learningRate || 0.02,
            smoothingFactor: 0.1,
            lvGrowth: this.config.lvGrowth || 0.18,
            lvDecay: this.config.lvDecay || 0.006
        };

        const startParams = { ...defaultParams, ...(this.task.seedParams || {}) };

        const { maxNeurons, initialNeurons } = this.config;

        const network: NetworkState = {
            activations: new Float32Array(maxNeurons),
            prevActivations: new Float32Array(maxNeurons),
            weights: new Float32Array(maxNeurons * maxNeurons),
            readout: new Float32Array(maxNeurons),
            inputWeights: new Float32Array(maxNeurons),
            neuronTypes: new Int8Array(maxNeurons),

            currentSize: initialNeurons,
            totalRegrown: 0,
            currentTargetDensity: 0.2,
            patienceCounter: 0,
            regressionSlope: 0,
            isLocked: false,
            lossWindow: [],

            // Lotka-Volterra Dynamics Initialization
            neuronEnergy: new Float32Array(maxNeurons).fill(1.0), // All neurons start with full energy
            isWinter: false,
            winterTimer: 0,
            epochCounter: 0,

            liveHyperparams: {
                leak: startParams.leak,
                spectral: startParams.spectral,
                inputScale: startParams.inputScale,
                learningRate: startParams.learningRate,
                smoothingFactor: startParams.smoothingFactor,
                lvGrowth: startParams.lvGrowth,
                lvDecay: startParams.lvDecay
            },
            checkpointHyperparams: null,

            optimizer: {
                globalStepScale: 1.0,
                timer: 0,
                baselineLoss: 1.0,
                currentLossAcc: 0,
                samples: 0,
                paramKeys: ['leak', 'spectral', 'inputScale'],
                currentIdx: 0,
                state: 'INIT',
                params: {
                    leak: { step: 0.05, bestVal: startParams.leak },
                    spectral: { step: 0.05, bestVal: startParams.spectral },
                    inputScale: { step: 0.1, bestVal: startParams.inputScale }
                },
                lastTestedVal: 0,
                testDirection: 0,
                lastParam: undefined,
                lastDelta: 0
            }
        };

        // Initialize input weights
        for (let i = 0; i < maxNeurons; i++) {
            network.inputWeights[i] = (Math.random() * 2 - 1) * startParams.inputScale;
            network.readout[i] = 0;
        }

        // Initialize Neuron Types (Dale's Law: 80% E, 20% I)
        for (let i = 0; i < maxNeurons; i++) {
            network.neuronTypes[i] = Math.random() < 0.8 ? 1 : -1;
        }

        // Create initial sparse signed connections respecting Dale's Law
        const density = 0.2;
        for (let i = 0; i < initialNeurons; i++) {
            for (let j = 0; j < initialNeurons; j++) {
                if (Math.random() < density) {
                    // Weight sign MUST match presynaptic neuron type (i)
                    const magnitude = Math.random() * startParams.spectral;
                    network.weights[i * maxNeurons + j] = magnitude * network.neuronTypes[i];
                }
            }
        }

        return network;
    }

    private initializeData(): void {
        this.dataSeries = [];
        let val = 1.2;
        for (let i = 0; i < MACKEY_GLASS_TAU + 50; i++) {
            const xt_tau = (i >= MACKEY_GLASS_TAU) ? this.dataSeries[i - MACKEY_GLASS_TAU] : 1.2;
            const delta = 0.2 * xt_tau / (1 + Math.pow(xt_tau, 10)) - 0.1 * val;
            val += delta;
            this.dataSeries.push(val);
        }
        this.avgLossBuffer = 1.0;
        this.stepCount = 0;
    }

    private performMitosis(): boolean {
        const { maxNeurons } = this.config;
        const net = this.network;

        if (net.currentSize >= maxNeurons) return false;

        const newIdx = net.currentSize;
        const newSize = net.currentSize + 1;
        const spectral = net.liveHyperparams.spectral;
        const N = net.currentSize;

        // Assign Type to New Neuron (Deterministic Balance for 80/20)
        let countE = 0;
        for (let i = 0; i < N; i++) if (net.neuronTypes[i] === 1) countE++;
        const currentRatio = countE / N;

        // If we have too many Excitatory (>0.8), spawn Inhibitory (-1). Else Excitatory (1).
        net.neuronTypes[newIdx] = currentRatio > 0.8 ? -1 : 1;

        // === INTELLIGENT NEURON INTEGRATION ===
        // Instead of random initialization, find a "parent" neuron to clone from
        // This mimics biological neurogenesis where new neurons inherit from active circuits

        // Find most active existing neuron as parent
        let parentIdx = 0;
        let maxActivity = 0;
        for (let i = 0; i < N; i++) {
            const activity = Math.abs(net.activations[i]);
            if (activity > maxActivity) {
                maxActivity = activity;
                parentIdx = i;
            }
        }

        // Clone weights from parent with small perturbation (not random!)
        const perturbScale = 0.1; // Small perturbation to differentiate from parent

        for (let i = 0; i < N; i++) {
            // Incoming connections: inherit from parent's incoming pattern
            const parentIncoming = net.weights[parentIdx * maxNeurons + i];
            if (parentIncoming !== 0) {
                // Copy with small perturbation, respecting new neuron's type
                const magnitude = Math.abs(parentIncoming) * (1 + (Math.random() - 0.5) * perturbScale);
                // Clamp magnitude to prevent overshooting spectral radius
                const clampedMag = Math.min(magnitude, spectral * 0.5);
                net.weights[newIdx * maxNeurons + i] = clampedMag * net.neuronTypes[newIdx];
            }

            // Outgoing connections: sparse and small to avoid immediate disruption
            // Only connect to neurons that parent was connected to
            const parentOutgoing = net.weights[i * maxNeurons + parentIdx];
            if (parentOutgoing !== 0 && Math.random() < 0.3) { // 30% chance to inherit
                const magnitude = Math.abs(parentOutgoing) * 0.2; // Much smaller
                net.weights[i * maxNeurons + newIdx] = magnitude * net.neuronTypes[i];
            }
        }

        // === ZERO READOUT INITIALIZATION ===
        // New neuron does NOT contribute to output immediately
        // It must earn its contribution through learning
        net.readout[newIdx] = 0;

        // Small input weight (will grow through L-V dynamics if useful)
        net.inputWeights[newIdx] = (Math.random() * 0.2 - 0.1) * net.liveHyperparams.inputScale;

        // Initialize metabolic energy at low level (must prove useful)
        net.neuronEnergy[newIdx] = 0.5;

        // Start with zero activation (clean slate)
        net.activations[newIdx] = 0;
        net.prevActivations[newIdx] = 0;

        net.currentSize = newSize;

        // Light stabilization after mitosis (less aggressive to preserve learning)
        this.applySinkhornStability(10);

        return true;
    }

    public step(): SimulationMetrics {
        const net = this.network;
        const N = net.currentSize;
        const dna = net.liveHyperparams;

        // 1. Generate input/target
        let targetVal: number, inputVal: number;

        if (this.task.type === 'CLASSIFY') {
            const result = this.task.generator(this.stepCount, this.dataSeries) as { input: number; target: number };
            inputVal = result.input;
            targetVal = result.target;
        } else {
            inputVal = this.dataSeries[this.dataSeries.length - 1];
            if (this.task.id === 'MACKEY_GLASS') {
                targetVal = this.task.generator(this.stepCount, this.dataSeries) as number;
            } else {
                targetVal = this.task.generator(this.stepCount, []) as number;
            }
        }

        this.dataSeries.push(targetVal);
        if (this.dataSeries.length > 500) this.dataSeries.shift();

        // 2. Reservoir Reservoir Update
        net.prevActivations.set(net.activations);
        const leak = dna.leak;
        const maxN = this.config.maxNeurons;

        for (let i = 0; i < N; i++) {
            let sum = 0;
            for (let j = 0; j < N; j++) {
                const w = net.weights[i * maxN + j];
                if (w !== 0) sum += w * net.prevActivations[j];
            }
            sum += net.inputWeights[i] * inputVal * dna.inputScale;
            const newState = Math.tanh(sum);
            net.activations[i] = (1 - leak) * net.prevActivations[i] + leak * newState;
        }

        // 3. Readout
        let prediction = 0;
        for (let i = 0; i < N; i++) {
            prediction += net.activations[i] * net.readout[i];
        }

        // 4. Learning
        const error = targetVal - prediction;
        const absError = Math.abs(error);
        const learningRate = net.liveHyperparams.learningRate;

        // Readout Learning (Noise-Gated LMS)
        // "Line 100" Logic: Use prevAvgLoss as the "Noise Floor" or "Expectation".
        // "Ratioed relative smoothing for local automated controls":
        // If the error is within the Noise Floor (Relative ~ 1.0), dampen learning to prevent Jitter.
        // If the error exceeds the Noise Floor (Signal), apply full learning.

        // Update Noise Floor (Smoothed Absolute Error)
        // Automating the "Cheat": The system learns how much history to keep vs new error.
        const alpha = net.liveHyperparams.smoothingFactor;
        this.prevAvgLoss = this.prevAvgLoss * (1 - alpha) + absError * alpha;

        // Calculate Adaptive Learning Rate (Noise Gating)
        // If current error is less than the smoothed trend, we are "in the noise" -> Dampen.
        // This stops the "Jagged Mess" (Oscillation) by preventing updates on random fluctuations.
        const isSignal = absError > this.prevAvgLoss;
        const adaptiveLR = isSignal ? learningRate : learningRate * 0.8; // Less aggressive dampening

        for (let i = 0; i < N; i++) {
            net.readout[i] += adaptiveLR * error * net.activations[i];
        }

        // MetaController Input: Relative Deviation
        // Feed the "Surprise" (Error / NoiseFloor) to the optimizer.
        // If ratio > 1 (Signal), it optimizes. If ratio < 1 (Noise), it holds.
        this.metaController.recordError(absError / (this.prevAvgLoss + 1e-9));
        // If > 1, getting worse. If < 1, getting better. Z-score handles the rest.

        // --- STABILITY CONTROL: Sinkhorn Normalization ---
        // Prevent explosion by normalizing weight matrix energy
        // Run periodically to keep system doubly stochastic
        if (this.stepCount % 10 === 0) {
            this.applySinkhornStability(3); // A few iterations are enough for maintenance
        }

        // Update Global Loss Buffer (Window Mean)
        // Used for Regression Slope and Status Determination (Macro-Stability)
        let sumWindow = 0;
        if (net.lossWindow.length > 0) {
            for (let k = 0; k < net.lossWindow.length; k++) sumWindow += net.lossWindow[k];
            this.avgLossBuffer = sumWindow / net.lossWindow.length;
        } else {
            this.avgLossBuffer = absError;
        }
        const opt = net.optimizer; // Re-use optimizer state for safety tracking
        const OPT_INTERVAL = 16;   // Evaluated every 16 steps

        opt.timer++;
        opt.currentLossAcc += absError;
        opt.samples++;

        if (opt.timer >= OPT_INTERVAL) {
            const currentAvg = opt.currentLossAcc / opt.samples;

            // Continuous Optimization: The LSTM runs every interval regardless of stability.
            // This allows for 'Hill Climbing' (Optimizing even when performing well).
            if (true) {

                // 1. Evaluate Previous Mutation (Safety Check)
                let reward = 0;
                if (opt.lastParam) {
                    // "Relative Loss" check: Did we improve relative to baseline?
                    if (currentAvg < opt.baselineLoss) {
                        // Good! Keep it.
                        // Strict Baseline (V2 Logic): Only accept if strictly better
                        opt.baselineLoss = currentAvg;
                        reward = 1.0;
                        // Success: Slightly increase step size to accelerate (up to 1.0)
                        opt.globalStepScale = Math.min(1.0, opt.globalStepScale * 1.5); // Boosted recovery (1.1 -> 1.5)
                    } else {
                        // "Fuzzing" prevention
                        (net.liveHyperparams as any)[opt.lastParam] -= (opt.lastDelta ?? 0);
                        reward = -1.0;
                        // Failure: Decay step size to cool down (V2 Logic)
                        // Prevent Death Spiral: Clamp minimum scale to 0.1
                        opt.globalStepScale = Math.max(0.1, opt.globalStepScale * 0.5);
                    }

                    // Reinforce the MetaController
                    this.metaController.reward(reward);
                } else {
                    opt.baselineLoss = currentAvg;
                }

                // 2. Get Next Intelligent Mutation
                // Context: [Leak, Spectral, Input, LR, Smooth, Growth, Decay, Density]
                const context = [
                    net.liveHyperparams.leak,
                    net.liveHyperparams.spectral,
                    net.liveHyperparams.inputScale,
                    net.liveHyperparams.learningRate,
                    net.liveHyperparams.smoothingFactor,
                    net.liveHyperparams.lvGrowth,
                    net.liveHyperparams.lvDecay,
                    net.currentTargetDensity
                ];
                const mutation = this.metaController.stepIntelligentMutation(context);

                // 3. Apply New Mutation
                if (mutation.shouldMutate) {
                    const paramKeys = ['leak', 'spectral', 'inputScale', 'learningRate', 'smoothingFactor', 'lvGrowth', 'lvDecay'] as const;
                    const targetParam = paramKeys[mutation.paramIndex];

                    // Defined Base Steps aligned with snap precision:
                    // - leak/spectral/smooth/lvGrowth: 2 decimals → step 0.05-0.10
                    // - inputScale: 1 decimal → step 0.1-0.2
                    // - learningRate: 3 decimals → step 0.005-0.01
                    // - lvDecay: 3 decimals → step 0.001-0.002
                    const PARAM_BASE_STEPS: Record<string, number> = {
                        leak: 0.05,          // 0.80 → 0.85 (2 decimals)
                        spectral: 0.05,      // 0.95 → 1.00 (2 decimals)
                        inputScale: 0.1,     // 1.0 → 1.1 (1 decimal)
                        learningRate: 0.01,  // 0.050 → 0.060 (3 decimals)
                        smoothingFactor: 0.05, // 0.10 → 0.15 (2 decimals)
                        lvGrowth: 0.05,      // 0.18 → 0.23 (2 decimals)
                        lvDecay: 0.002       // 0.006 → 0.008 (3 decimals)
                    };

                    const baseStep = PARAM_BASE_STEPS[targetParam] || 0.05;

                    // FIXED: Force exact "Whole Number" steps as requested.
                    // Ignored: mutation.magnitude (prevents micro-scaling)
                    // Ignored: opt.globalStepScale (prevents dampening)
                    // Enforced: Discrete Direction to avoid float dilution from tanh
                    const discreteDir = mutation.direction > 0 ? 1 : -1;
                    const delta = discreteDir * baseStep;

                    // Apply
                    net.liveHyperparams[targetParam] += delta;

                    // DEBUG: Log Mutation
                    if (false) {
                        console.log(`[MUTATION] ${targetParam} delta: ${delta.toFixed(6)} new: ${net.liveHyperparams[targetParam].toFixed(4)} StepScale: ${opt.globalStepScale.toFixed(3)}`);
                    }

                    // Track for next evaluation
                    opt.lastParam = targetParam;
                    opt.lastDelta = delta;

                    // Clamp and SNAP to clean decimals (prevents fractional drift like 1.0004)
                    const snap = (v: number, decimals: number) => Math.round(v * Math.pow(10, decimals)) / Math.pow(10, decimals);
                    net.liveHyperparams.leak = snap(Math.max(0.01, Math.min(1.0, net.liveHyperparams.leak)), 2);
                    net.liveHyperparams.spectral = snap(Math.max(0.1, Math.min(2.0, net.liveHyperparams.spectral)), 2);
                    net.liveHyperparams.inputScale = snap(Math.max(0.1, Math.min(5.0, net.liveHyperparams.inputScale)), 1);
                    net.liveHyperparams.learningRate = snap(Math.max(0.001, Math.min(1.0, net.liveHyperparams.learningRate)), 3);
                    net.liveHyperparams.smoothingFactor = snap(Math.max(0.001, Math.min(0.5, net.liveHyperparams.smoothingFactor)), 2);
                    net.liveHyperparams.lvGrowth = snap(Math.max(0.01, Math.min(1.0, net.liveHyperparams.lvGrowth)), 2);
                    net.liveHyperparams.lvDecay = snap(Math.max(0.0001, Math.min(0.1, net.liveHyperparams.lvDecay)), 3);
                } else {
                    opt.lastParam = undefined;
                    opt.lastDelta = 0;
                }

            }

            // Reset Batch (Always if interval reached)
            opt.timer = 0;
            opt.currentLossAcc = 0;
            opt.samples = 0;
        }

        // 6. Structural Adaptation
        net.lossWindow.push(absError);
        if (net.lossWindow.length > 50) net.lossWindow.shift();

        let trendSlope = 0;
        if (net.lossWindow.length >= 50) {
            const start = net.lossWindow.slice(0, 25).reduce((a, b) => a + b, 0);
            const end = net.lossWindow.slice(25).reduce((a, b) => a + b, 0);
            trendSlope = start - end;
        }
        net.regressionSlope = trendSlope;

        // Lock Logic (Original)
        if (!net.isLocked && this.avgLossBuffer < this.config.solvedThreshold) {
            net.isLocked = true;
            net.patienceCounter = 0;
        } else if (net.isLocked && this.avgLossBuffer > this.config.unlockThreshold) {
            net.isLocked = false;
        }

        let status = 'STABLE';
        if (net.isLocked) {
            status = 'LOCKED';
        } else {
            const isImproving = trendSlope < this.config.learningSlopeThreshold;
            if (isImproving) {
                status = 'LEARNING';
                net.patienceCounter = Math.max(0, net.patienceCounter - 1);
            } else {
                status = 'STAGNANT';
                net.patienceCounter++;
            }

            // Lotka-Volterra Selection Dynamics (From Research Report)
            // Equation: dn/dt = r*n + f*n - alpha*n^2
            // r = Intrinsic Growth (lvGrowth) - Kept alive by DNA
            // f = Fitness/Demand (absError) - Reacts to immediate problems
            // alpha = Competition (lvDecay) - Limits capacity
            const D = net.currentTargetDensity;
            const intrinsic = net.liveHyperparams.lvGrowth * D;
            const demand = absError * D; // Fitness term (Demand for plasticity)
            const competition = net.liveHyperparams.lvDecay * D * D;

            const densityChange = intrinsic + demand - competition;

            // Apply with limits (0.1 to 1.0)
            net.currentTargetDensity = Math.max(0.1, Math.min(1.0, net.currentTargetDensity + densityChange));

            if (net.patienceCounter > this.config.patienceLimit) {
                if (this.performMitosis()) {
                    status = 'GROWING';
                    net.patienceCounter = 0;
                }
            }
        }

        const targetConns = Math.floor(N * N * net.currentTargetDensity);

        // ====================================================================
        // 7. TRUE LOTKA-VOLTERRA WEIGHT DYNAMICS
        // Weights are populations: Growth (food) - Predation (competition) - Decay (starvation)
        // ====================================================================

        // Update epoch counter and check for "Winter" extinction event
        net.epochCounter++;
        const WINTER_PERIOD = 500;    // Winter every 500 steps
        const WINTER_DURATION = 50;   // Winter lasts 50 steps

        if (net.epochCounter % WINTER_PERIOD === 0 && !net.isWinter) {
            net.isWinter = true;
            net.winterTimer = WINTER_DURATION;
            // console.log('[L-V] Winter has come! High decay for', WINTER_DURATION, 'steps');
        }

        if (net.isWinter) {
            net.winterTimer--;
            if (net.winterTimer <= 0) {
                net.isWinter = false;
                // console.log('[L-V] Winter ended. Survivors rewarded.');
            }
        }

        // Decay multiplier: 3x during winter (extinction event)
        const winterMultiplier = net.isWinter ? 3.0 : 1.0;
        const baseLvGrowth = net.liveHyperparams.lvGrowth;
        const baseLvDecay = net.liveHyperparams.lvDecay * winterMultiplier;

        let activeCount = 0;

        // Skip L-V dynamics if locked (preserve winning configuration)
        if (!net.isLocked) {
            // Pre-compute neuron activity (input to L-V dynamics)
            const neuronActivity = new Float32Array(N);
            for (let i = 0; i < N; i++) {
                neuronActivity[i] = Math.abs(net.activations[i]);
            }

            // Update per-neuron metabolic energy (for adaptive learning rates)
            // High error = high stress = high energy expenditure
            for (let i = 0; i < N; i++) {
                // Energy rises when neuron is active (being useful)
                // Energy falls when neuron is inactive (starvation)
                const activity = neuronActivity[i];
                const energyDelta = activity * 0.1 - 0.02; // Active gains, inactive loses
                net.neuronEnergy[i] = Math.max(0.1, Math.min(2.0, net.neuronEnergy[i] + energyDelta));
            }

            // L-V Weight Update Loop
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    const idx = i * maxN + j;
                    const w = net.weights[idx];

                    if (w === 0) continue; // Skip dead weights

                    const absW = Math.abs(w);

                    // === LATERAL INHIBITION: Compute local competition ===
                    // Each weight competes with its neighbors (not global)
                    // Neighbors: weights in same row (same presynaptic neuron)
                    let neighborActivity = 0;
                    let neighborCount = 0;

                    // Check neighbors: j-1, j+1 (same row i)
                    for (let dj = -2; dj <= 2; dj++) {
                        if (dj === 0) continue; // Skip self
                        const nj = j + dj;
                        if (nj >= 0 && nj < N) {
                            const nidx = i * maxN + nj;
                            neighborActivity += net.weights[nidx] * net.weights[nidx]; // Squared magnitude
                            neighborCount++;
                        }
                    }

                    neighborActivity = neighborCount > 0 ? neighborActivity / neighborCount : 0;

                    // === L-V DYNAMICS: PRUNING DECISIONS ONLY ===
                    // L-V determines SURVIVAL, not magnitude updates.
                    // This prevents L-V from fighting gradient learning.
                    //
                    // A weight survives if: growth > predation + starvation
                    // - Growth: proportional to presynaptic activity (useful connections)
                    // - Predation: proportional to local competition (crowded connections)
                    // - Starvation: base decay (unused connections)

                    const presynapticEnergy = net.neuronEnergy[i];
                    const growth = baseLvGrowth * presynapticEnergy;          // Survival pressure from activity
                    const predation = baseLvDecay * neighborActivity * 0.5;   // Death pressure from competition
                    const starvation = baseLvDecay * (1 - presynapticEnergy); // Death pressure from inactivity

                    // Net survival score (positive = survives, negative = dies)
                    const survivalScore = growth - predation - starvation;

                    // Pruning decision: if survival score < 0 AND weight is small, prune it
                    // Large weights (strongly learned) survive regardless of L-V
                    const survivalThreshold = this.config.pruneThreshold * 2;
                    const shouldPrune = survivalScore < 0 && absW < survivalThreshold;

                    if (shouldPrune) {
                        net.weights[idx] = 0;
                    } else {
                        // Keep weight at current magnitude (no L-V magnitude updates!)
                        activeCount++;
                    }
                }
            }
        } else {
            // Locked: just count active weights
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    const idx = i * maxN + j;
                    if (net.weights[idx] !== 0) activeCount++;
                }
            }
        }

        const deficit = targetConns - activeCount;
        if (deficit > 0) {
            // THIS IS THE KEY FIX: Regrowth happens regardless of lock state if deficit exists
            if (status !== 'GROWING') {
                status = net.isLocked ? 'LOCKED' : 'GROWING_SYNAPSES';
            }
            const connectionsToGrow = Math.ceil(deficit * 0.1) + 1;
            let newConnections = 0;
            let attempts = 0;
            const spectral = net.liveHyperparams.spectral;

            while (newConnections < connectionsToGrow && attempts < 100) {
                attempts++;
                const i = Math.floor(Math.random() * N);
                const j = Math.floor(Math.random() * N);
                const idx = i * maxN + j;
                if (net.weights[idx] === 0) {
                    // Strict E-I Topology: Weight sign must match presynaptic neuron type
                    const magnitude = Math.random() * spectral;
                    net.weights[idx] = magnitude * net.neuronTypes[i];
                    newConnections++;
                }
            }
            net.totalRegrown += newConnections;
            activeCount += newConnections;

            // Immediate partial stabilization after growth to prevent shock
            if (newConnections > 0) {
                this.applySinkhornStability(20);
            }
        }

        this.stepCount++;

        // RE-INTRODUCED DUPLICATE PUSH (User Insistence / High Frequency Dynamics)
        this.dataSeries.push(targetVal);
        if (this.dataSeries.length > 500) this.dataSeries.shift();

        return {
            loss: absError,
            avgLoss: this.avgLossBuffer,
            activeConnections: activeCount,
            targetConnections: targetConns,
            regrown: net.totalRegrown,
            prediction,
            target: targetVal,
            neuronCount: N,
            patience: net.patienceCounter,
            adaptationStatus: status,
            dna: { ...dna },
            metaController: this.metaController.getState()
        };
    }

    public isLocked(): boolean {
        return this.network.isLocked;
    }

    public getSteps(): number {
        return this.stepCount;
    }
    public reset(task?: Task): void {
        if (task) this.task = task;
        this.network = this.initializeNetwork();
        this.metaController.reset();
        this.prevAvgLoss = 1.0;
        this.initializeData();
    }


    // Sinkhorn-Knopp Algorithm: Doubly Stochastic Normalization (Sparse mHC)
    // Projects weights onto the Birkhoff polytope to bound spectral radius <= 1.
    // Adapted for Sparse Signed Matrices:
    // 1. P = exp(W) for non-zero entries (Zeros remain zero to preserve sparsity)
    // 2. Normalize Rows/Cols iteratively
    // 3. Restore Signs (E/I Topology)
    private applySinkhornStability(iterations: number = 20): void {
        const net = this.network;
        const N = net.currentSize;
        const maxN = this.config.maxNeurons;

        // Temporary storage for signs and the positive matrix P
        const signs = new Float32Array(N * maxN); // Stores 1, -1, or 0
        const P = new Float32Array(N * maxN);     // Stores exp(|W|) or 0

        // 1. Extract signs and create positive matrix P = exp(|W|)
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                const idx = i * maxN + j;
                const weight = net.weights[idx];
                if (weight !== 0) {
                    signs[idx] = Math.sign(weight);
                    P[idx] = Math.exp(Math.abs(weight)); // Use exp on magnitude
                } else {
                    signs[idx] = 0;
                    P[idx] = 0;
                }
            }
        }

        // Temporary buffers for sums
        const rowSums = new Float32Array(N);
        const colSums = new Float32Array(N);

        // 2. Perform Sinkhorn-Knopp on the positive matrix P
        for (let iter = 0; iter < iterations; iter++) {
            // Row Normalization
            rowSums.fill(0);
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    rowSums[i] += P[i * maxN + j];
                }
            }
            for (let i = 0; i < N; i++) {
                if (rowSums[i] > 1e-9) { // Avoid division by zero
                    const factor = 1.0 / rowSums[i];
                    for (let j = 0; j < N; j++) {
                        P[i * maxN + j] *= factor;
                    }
                }
            }

            // Column Normalization
            colSums.fill(0);
            for (let j = 0; j < N; j++) {
                for (let i = 0; i < N; i++) {
                    colSums[j] += P[i * maxN + j];
                }
            }
            for (let j = 0; j < N; j++) {
                if (colSums[j] > 1e-9) { // Avoid division by zero
                    const factor = 1.0 / colSums[j];
                    for (let i = 0; i < N; i++) {
                        P[i * maxN + j] *= factor;
                    }
                }
            }
        }

        // 3. Re-apply signs to the normalized magnitudes and update net.weights
        // INTEGRATION FIX: Apply the 'Spectral' hyperparameter to the normalized manifold.
        // Sinkhorn forces Radius=1. We must scale it by the LSTM-tuned value to re-integrate the systems.
        const spectralRadius = net.liveHyperparams.spectral;

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                const idx = i * maxN + j;
                // Re-integrate: Weights = NormalizedStructure * Sign * SpectralScaler
                net.weights[idx] = P[idx] * signs[idx] * spectralRadius;
            }
        }
    }

    public getNetworkState(): NetworkState {
        return { ...this.network };
    }
}
