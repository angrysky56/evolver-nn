import { Task } from "../tasks/tasks";

import { BicameralMetaController, REGIMES } from "./metaController";

export interface SimulationConfig {
  // STRUCTURE: These affect network architecture
  maxNeurons: number;
  initialNeurons: number;
  pruneThreshold: number;

  // GOALS: These affect when to stop/grow
  patienceLimit: number;
  solvedThreshold: number;
  unlockThreshold: number;
  learningSlopeThreshold: number;

  // NOTE: learningRate, lvGrowth, lvDecay, leak, spectral, etc.
  // are now 100% controlled by the Regime Driver (REGIMES in metaController.ts)
}

export const DEFAULT_CONFIG: SimulationConfig = {
  maxNeurons: 4096,
  initialNeurons: 512,
  pruneThreshold: 0.01,
  patienceLimit: 256,
  solvedThreshold: 0.01,
  unlockThreshold: 0.02,
  learningSlopeThreshold: -0.0005,
};

export interface NetworkState {
  activations: Float32Array;
  prevActivations: Float32Array;
  weights: Float32Array;
  readout: Float32Array;
  inputWeights: Float32Array;
  feedbackWeights: Float32Array;
  neuronTypes: Int8Array;

  currentSize: number;
  totalRegrown: number;
  currentTargetDensity: number;
  patienceCounter: number;
  regressionSlope: number;
  isLocked: boolean;
  lossWindow: number[];

  // L-V State
  neuronEnergy: Float32Array; // Current energy per neuron
  engramTrace: Float32Array; // Tracks long-term stability of synapses (Plastic Delta-Engrams)

  liveHyperparams: {
    leak: number;
    spectral: number;
    inputScale: number;
    learningRate: number;
    smoothingFactor: number;
    lvGrowth: number;
    lvDecay: number;
    outputGain: number;
  };

  // Spectral Normalization State (Power Iteration)
  spectralU: Float32Array; // Left Singular Vector
  spectralV: Float32Array; // Right Singular Vector
  currentSpectralRadius: number;

  // Simple Optimizer State (No LSTM)
  optimizer: {
    timer: number;
    baselineLoss: number;
    currentLossAcc: number;
    samples: number;
    lastParam: string | null;
    lastDelta: number;
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
  spectralRadius: number;
  adaptationStatus: string;
  // DNA flattened for UI
  dna: {
    leak: number;
    spectral: number;
    inputScale: number;
    learningRate: number;
    smoothingFactor: number;
    lvGrowth: number;
    lvDecay: number;
    outputGain: number;
  };
  // Stubbed for UI compatibility
  metaController: {
    shortTermActivity: number;
    longTermActivity: number;
    gate: number;
  };
  logs: string[];
  step: number; // Added for log sync
}

export class EvolutionaryChaosNetwork {
  private config: SimulationConfig;
  private task: Task;
  private network: NetworkState;
  private metaController: BicameralMetaController;
  private dataSeries: number[] = [];
  private recentMSEWindow: number[] = []; // Short window for honest convergence (20 steps)
  private recentMSE: number = 1.0; // Honest metric (20-step average)
  private prevRecentMSE: number = 1.0; // Previous recentMSE for reward calculation
  private stepCount: number = 0;
  private logQueue: string[] = [];
  // private prevAvgLoss: number = 1.0; // Removed - unused
  private solvedGracePeriod: number = 0; // Counter for sustained low-loss before locking
  private structuralCooldown: number = 0; // Steps to wait before next structural change

  private lastInput: number = 0; // Store last input for Hebbian alignment

  constructor(task: Task, config: Partial<SimulationConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.task = task;
    this.network = this.initializeNetwork();
    // FIX: inputFeaturesSize must match the 10 features we calculate in step()
    // (Spectral 8 + Proprioception 2)
    this.metaController = new BicameralMetaController({
      hiddenSize: 128, // Reduced from 512 (128 is faster/sharper for 10 inputs)
      outputSize: 8,
      inputFeaturesSize: 10, // WAS 8 - THIS CAUSED THE CRASH
    });
    this.initializeData();
  }

  private initializeNetwork(): NetworkState {
    // "Birth State" - Soft Start (CRUISE)
    // Start conservatively to prevent immediate saturation.
    const startParams = {
      ...REGIMES.CRUISE,
    };

    const { maxNeurons, initialNeurons } = this.config;

    const network: NetworkState = {
      activations: new Float32Array(maxNeurons),
      prevActivations: new Float32Array(maxNeurons),
      weights: new Float32Array(maxNeurons * maxNeurons),
      readout: new Float32Array(maxNeurons),
      inputWeights: new Float32Array(maxNeurons),
      feedbackWeights: new Float32Array(maxNeurons),
      neuronTypes: new Int8Array(maxNeurons),

      currentSize: initialNeurons,
      totalRegrown: 0,
      currentTargetDensity: 0.2, // 20% Density Target
      patienceCounter: 0,
      regressionSlope: -0.01, // Assume learning during warmup (prevent false stagnation)
      isLocked: false,
      lossWindow: [],
      neuronEnergy: new Float32Array(maxNeurons).fill(1.0),
      engramTrace: new Float32Array(maxNeurons * maxNeurons).fill(0),
      liveHyperparams: { ...startParams },

      // Spectral State
      spectralU: new Float32Array(maxNeurons)
        .fill(0)
        .map(() => Math.random() - 0.5),
      spectralV: new Float32Array(maxNeurons)
        .fill(0)
        .map(() => Math.random() - 0.5),
      currentSpectralRadius: 1.0,

      // Simplified Optimizer (The one that worked)
      optimizer: {
        timer: 0,
        baselineLoss: 1.0,
        currentLossAcc: 0,
        samples: 0,
        lastParam: null,
        lastDelta: 0,
      },
    };

    // Init Weights
    for (let i = 0; i < maxNeurons; i++) {
      network.inputWeights[i] =
        (Math.random() * 2 - 1) * startParams.inputScale;
      network.readout[i] = 0;
      network.feedbackWeights[i] = Math.random() * 2 - 1; // Fixed random feedback (Feedback Alignment)
      network.neuronTypes[i] = Math.random() < 0.8 ? 1 : -1; // 80% E, 20% I
    }

    // Initial Wiring - Reservoir
    const density = 0.2;
    for (let i = 0; i < initialNeurons; i++) {
      for (let j = 0; j < initialNeurons; j++) {
        if (Math.random() < density) {
          network.weights[i * maxNeurons + j] =
            (Math.random() * 2 - 1) * startParams.spectral;
        }
      }
    }

    // We need to ensure we don't start with a "fried" brain.
    // Force normalization immediately.
    this.applySpectralNormalization(network, 10); // 10 iterations to be sure

    return network;
  }

  private initializeData(): void {
    // Use task-specific seed if available
    if (this.task.seed) {
      this.dataSeries = this.task.seed();
    } else {
      // Generic neutral seed for tasks without specific needs
      this.dataSeries = [];
      for (let i = 0; i < 50; i++) {
        this.dataSeries.push(0);
      }
    }
    this.stepCount = 0;
  }

  // --- Core Math ---

  public step(): SimulationMetrics {
    const net = this.network;
    const N = net.currentSize;
    const dna = net.liveHyperparams;
    const maxN = this.config.maxNeurons;

    // 1. Inputs (Standardized)
    const { input, target } = this.task.generator(
      this.stepCount,
      this.dataSeries,
    );
    const inputVal = input; // Standardized
    const targetVal = target; // Standardized
    this.lastInput = inputVal; // Capture for Hebbian logic

    this.dataSeries.push(targetVal);
    if (this.dataSeries.length > 500) this.dataSeries.shift();

    // 2. Reservoir Update
    net.prevActivations.set(net.activations);
    const leak = dna.leak;
    const spectral = dna.spectral; // Assuming spectral is used here

    // Update reservoir neurons with Tanh non-linearity
    for (let i = 0; i < N; i++) {
      let sum = 0;
      // Recurrent weights (sparse)
      for (let j = 0; j < N; j++) {
        sum += net.weights[i * maxN + j] * net.prevActivations[j];
      }
      // Input weights
      sum += net.inputWeights[i] * inputVal * net.liveHyperparams.inputScale;

      // Leakage integrator
      let state =
        (1 - leak) * net.prevActivations[i] + leak * Math.tanh(sum * spectral);

      // --- SHORT-TERM PLASTICITY (Memristive Fatigue) ---
      // Neurons tire out if overused, forcing signal diversity.
      // 1. Modulate output by current energy
      state *= net.neuronEnergy[i];

      // 2. Deplete energy (Fatigue)
      // Consumption: 0.2% (was 0.5%) - Much gentler
      // Equilibrium at state=1.0 if consumption=recovery
      net.neuronEnergy[i] -= Math.abs(state) * 0.002;

      // 3. Recharge energy (Recovery)
      // Recovery: 0.2% (was 0.1%) - Faster recovery
      net.neuronEnergy[i] += 0.002;

      // 4. Clamp Energy [0, 1]
      net.neuronEnergy[i] = Math.max(0.0, Math.min(1.0, net.neuronEnergy[i]));
      // --------------------------------------------------

      net.activations[i] = state;
    }

    // --- UNSUPERVISED PLASTICITY REMOVED (Replaced by Feedback Alignment after Readout) ---

    // 3. Readout
    let prediction = 0;
    for (let i = 0; i < N; i++) {
      prediction += net.activations[i] * net.readout[i];
    }

    // 4. Learning (Online LMS)
    const error = targetVal - prediction;
    const absError = Math.abs(error);

    // EXPLOSION GUARD
    if (isNaN(absError) || absError > 50) {
      this.logEvent(
        `[SYSTEM] Emergency Stabilization (Loss: ${isNaN(absError) ? "NaN" : absError.toFixed(2)})`,
      );

      // CRITICAL: Punish the Meta-Controller for crashing the system
      // (Handled implicitly by huge loss causing negative reward in next eval)

      // Also force-feed this error into the honest metric so UI/Logic sees it
      const penaltyError = isNaN(absError) ? 10.0 : Math.min(absError, 10.0);
      this.recentMSEWindow.push(penaltyError);
      if (this.recentMSEWindow.length > 20) this.recentMSEWindow.shift();
      this.recentMSE =
        this.recentMSEWindow.reduce((a, b) => a + b, 0) /
        this.recentMSEWindow.length;

      this.recentMSE =
        this.recentMSEWindow.reduce((a, b) => a + b, 0) /
        this.recentMSEWindow.length;

      this.applySpectralNormalization(net, 20); // Force fix
      net.activations.fill(0); // Reset energy
      net.prevActivations.fill(0);
      return this.getMetrics(0, 0, 1.0, "STABILIZING");
    }

    // METRICS: RADICAL HONESTY (No Smoothing)
    // We use a short window (20 steps) to see the spikes.

    // Recent MSE (honest 20-sample average)
    this.recentMSEWindow.push(absError);
    if (this.recentMSEWindow.length > 20) this.recentMSEWindow.shift();
    this.prevRecentMSE = this.recentMSE; // Track previous for reward
    this.recentMSE =
      this.recentMSEWindow.reduce((a, b) => a + b, 0) /
      this.recentMSEWindow.length;

    // Get LR from live hyperparams (controlled by MetaController via delta clamp)
    // Apply local safety cap for THIS step only
    // Get LR from live hyperparams (controlled by MetaController)
    // REMOVED HARD CAP (was 0.005) - Let the Strategy Selector drive!
    // Strategies like EXPLORE use 0.03, which is needed for fast learning.
    let currentLR = dna.learningRate;

    // If SOLVED (using consistent recentMSE metric), stop learning
    if (this.recentMSE < this.config.solvedThreshold) {
      currentLR = 0; // FREEZE WEIGHTS for this step only
    }

    // REMOVED "Cruise Control" (Dampening)
    // It was scaling LR down to 0.001 near convergence, making it impossible to close the final gap.
    // The LSTM's "CONSOLIDATE" strategy (LR 0.002) is the correct way to handle this.

    // NOTE: We do NOT write currentLR back to dna.learningRate here.
    // The MetaController is the sole authority for the "live" LR displayed in UI.

    // Update Readouts
    for (let i = 0; i < N; i++) {
      let delta = currentLR * error * net.activations[i];
      // Safety Clamp for LMS: Prevent runaway readout growth
      delta = Math.max(-0.1, Math.min(0.1, delta));
      net.readout[i] += delta;
    }

    // 5. REGIME DRIVER (LSTM)

    // --- SUPERVISED REFLEXIVE PLASTICITY (Feedback Alignment) ---
    // The "Nudged Phase" - inject error back into the reservoir to align features.
    if (!net.isLocked && currentLR > 0.0001) {
      // Use a smaller LR for internal weights to be stable (10% of readout LR)
      this.applySupervisedHebbian(net, error, currentLR * 0.1);
    }
    // ------------------------------------------------------------

    // A. SPECTRAL FEATURE EXTRACTION (7 features for LSTM)
    // All features NORMALIZED to [-1, 1] range for stable LSTM input

    // CRITICAL FIX: Analyze the ERROR signal, not the target data!
    // The LSTM needs to see error cycles to react to them.
    let errorLowFreq = 0; // Slow error trends
    let errorMidFreq = 0; // Error oscillation pattern (cyclic loss)
    let errorHighFreq = 0; // Error noise/detail

    const errorWindowSize = Math.min(128, this.recentMSEWindow.length);

    if (errorWindowSize >= 32) {
      const errorSnippet = this.recentMSEWindow.slice(-errorWindowSize);

      // Low Freq: Error trend over the window (getting better or worse?)
      const half = Math.floor(errorWindowSize / 2);
      const firstHalfErr =
        errorSnippet.slice(0, half).reduce((a, b) => a + b, 0) / half;
      const secondHalfErr =
        errorSnippet.slice(half).reduce((a, b) => a + b, 0) / half;
      // Positive = error increasing (bad), Negative = error decreasing (good)
      errorLowFreq = Math.tanh((secondHalfErr - firstHalfErr) * 20);

      // Mid Freq: ERROR CYCLE DETECTION (zero crossings around mean error)
      // This is the KEY feature for cyclic loss!
      const errMean = errorSnippet.reduce((a, b) => a + b, 0) / errorWindowSize;
      let errZeroCrossings = 0;
      for (let i = 1; i < errorWindowSize; i++) {
        if ((errorSnippet[i] - errMean) * (errorSnippet[i - 1] - errMean) < 0) {
          errZeroCrossings++;
        }
      }
      // More crossings = more cyclic error pattern
      // Normalize: 0 crossings = -1, many crossings = +1
      errorMidFreq = Math.tanh(
        (errZeroCrossings / (errorWindowSize / 8) - 1) * 2,
      );

      // High Freq: Error volatility (variance of error changes)
      const errDiffs: number[] = [];
      for (let i = 1; i < errorWindowSize; i++) {
        errDiffs.push(errorSnippet[i] - errorSnippet[i - 1]);
      }
      const errDiffMean = errDiffs.reduce((a, b) => a + b, 0) / errDiffs.length;
      const errDiffVar =
        errDiffs.reduce((a, b) => a + Math.pow(b - errDiffMean, 2), 0) /
        errDiffs.length;
      errorHighFreq = Math.tanh(errDiffVar * 100);
    }

    // Error Spike: Instantaneous vs recent average (pattern break detection)
    // [LATENCY FIX] More sensitive: 2x error ratio -> saturate to 1.0
    let errorSpike = 0;
    if (
      this.recentMSE > 0.000001 &&
      !isNaN(absError) &&
      !isNaN(this.recentMSE)
    ) {
      errorSpike = Math.tanh((absError / this.recentMSE - 1) * 4);
    }

    // Error Phase: Are we at a peak or trough of the error cycle?
    // This helps the LSTM anticipate the next phase
    let errorPhase = 0;
    if (this.recentMSEWindow.length >= 16) {
      const recent8 = this.recentMSEWindow.slice(-8);
      const prev8 = this.recentMSEWindow.slice(-16, -8);
      const recentAvg = recent8.reduce((a, b) => a + b, 0) / 8;
      const prevAvg = prev8.reduce((a, b) => a + b, 0) / 8;
      // Positive = rising (heading to peak), Negative = falling (heading to trough)
      errorPhase = Math.tanh((recentAvg - prevAvg) * 50);
    }

    // Error Trend (smoothed improvement/degradation)
    // FIXED: Actually compute the trend instead of leaving it as 0!
    // Compare short-term average vs medium-term average to detect improvement/degradation.
    let errorTrend = 0;
    if (this.recentMSEWindow.length >= 16) {
      const recent8 = this.recentMSEWindow.slice(-8);
      const older8 = this.recentMSEWindow.slice(-16, -8);
      const recentAvg = recent8.reduce((a, b) => a + b, 0) / 8;
      const olderAvg = older8.reduce((a, b) => a + b, 0) / 8;
      // Negative = improving (recent < older), Positive = degrading (recent > older)
      // Scale: 0.01 difference = noticeable signal
      errorTrend = Math.tanh((recentAvg - olderAvg) * 50);
    }

    // Network Entropy (utilization level)
    let entropyLevel = this.calculateNetworkEntropy(net);

    // NaN Protection & Clamping
    // [LATENCY FIX] Use INSTANT error, not smoothed average!
    // The LSTM needs to see the raw frame error to react in real-time.
    const instantLoss = Math.tanh(absError * 10); // 0.1 error -> ~1.0, direct reflex

    // PROPRIOCEPTION: "What am I doing right now?"
    // Normalize these to [-1, 1] roughly
    // Centered on typical midpoint values so 0 = "normal" state
    const currentLRFeature = Math.tanh(
      (net.liveHyperparams.learningRate - 0.02) * 50,
    ); // Centered on 0.02
    const currentDensityFeature = Math.tanh(
      (net.currentTargetDensity - 0.2) * 5,
    ); // Centered on 0.2

    const features = [
      errorLowFreq, // 0: Error trend (getting better/worse)
      errorMidFreq, // 1: Error cycle detection (KEY for cyclic loss!)
      errorHighFreq, // 2: Error volatility
      errorSpike, // 3: Pattern break detection (now 2x more sensitive)
      errorPhase, // 4: Error cycle phase (peak/trough)
      errorTrend, // 5: Smoothed improvement/degradation
      entropyLevel, // 6: Network utilization
      instantLoss, // 7: INSTANT loss (reflex nerve, not rearview mirror)
      currentLRFeature, // 8: PROPRIOCEPTION - "I know my own speed"
      currentDensityFeature, // 9: PROPRIOCEPTION - "I know my own weight"
    ].map((f) => {
      // Paranoid check: If ANY math failed above, default to 0 to save the LSTM
      if (typeof f !== "number" || isNaN(f) || !isFinite(f)) {
        // this.logEvent(`[WARNING] Bad Feature detected. Zeroing.`);
        return 0;
      }
      return Math.max(-1, Math.min(1, f));
    });

    // STRATEGY SELECTION
    // LSTM selects one of 5 discrete strategies instead of tuning 8 continuous params
    // Pass recentMSE so the Brain knows if it needs to PANIC
    const strategyResult = this.metaController.selectStrategy(
      features,
      this.recentMSE,
    );
    const adaptiveParams = strategyResult.params;
    const currentStrategy = strategyResult.strategyName;

    // C. Apply Strategy Parameters
    const currentParams = net.liveHyperparams;

    // --- STAGNATION OVERRIDE (Homeostatic Plasticity) ---
    // If the system is stagnant, force a "Hyper-Evolution" mode.
    // The LSTM might be too cautious; we physically force higher variance.
    let growthMultiplier = 1.0;
    if (
      net.lossWindow.length >= 200 &&
      net.regressionSlope > -0.0001 &&
      net.regressionSlope < 0.0001
    ) {
      // We are flatlining. Boost mutation (Growth/Decay).
      growthMultiplier = 3.0;
      this.logEvent(
        `[OVERRIDE] Stagnation detected. Boosting evolutionary pressure.`,
      );
    }
    // -----------------------------------------------------

    // EXPLORATION PHASE: Allow faster adaptation during early learning
    // First 500 steps = exploration, then settle into exploitation
    const isExploring = this.stepCount < 500;

    // LOSS-BASED EMERGENCY MODE: If loss is bad (>0.025), be more aggressive
    // const isEmergency = this.recentMSE > 0.025; // UNUSED after Quantization Update

    // Learning Rate: Use delta clamp for stability
    // Learning Rate: GEOMETRIC SMOOTHING (The "Continuity" Law)
    // Prevent teleportation. Max change = 10% per step.
    // This allows exponential adaptation but strictly enforces a continuous path.
    const targetLR = adaptiveParams.learningRate;
    const liveCurrentLR = currentParams.learningRate;
    const maxChange = 0.02; // 2% max jump (Grandma Mode)

    // Calculate clamped ratio
    const ratio = targetLR / Math.max(1e-9, liveCurrentLR); // Safety div
    const clampedRatio = Math.max(
      1.0 - maxChange,
      Math.min(1.0 + maxChange, ratio),
    );

    // Apply smooth update
    currentParams.learningRate = liveCurrentLR * clampedRatio;

    // MAJOR HYPERPARAMS: DIRECT APPLICATION (No Smoothing)
    // These are now quantized (0.05 steps) in the Driver, so we must SNAP to them.
    currentParams.leak = adaptiveParams.leak;
    currentParams.spectral = adaptiveParams.spectral;
    currentParams.inputScale = adaptiveParams.inputScale;
    currentParams.smoothingFactor = adaptiveParams.smoothingFactor;
    currentParams.outputGain = adaptiveParams.outputGain;

    // L-V Dynamics: Keep smoothing (these affect structure, need to be stable)
    const lvAlpha = isExploring ? 0.3 : 0.1;
    currentParams.lvGrowth =
      currentParams.lvGrowth * (1 - lvAlpha) +
      adaptiveParams.lvGrowth * growthMultiplier * lvAlpha;
    currentParams.lvDecay =
      currentParams.lvDecay * (1 - lvAlpha) +
      adaptiveParams.lvDecay * growthMultiplier * lvAlpha;

    // ENFORCE PHYSICS (The "Clutch")
    // Run Spectral Normalization periodically (or every step for small nets)
    // Power Iteration is cheap-ish, but for 4096 neurons it adds up.
    // Step Frequency: Every 1 step for N < 256, Every 10 for N > 256
    const spectralFreq = N < 256 ? 1 : 10;
    if (this.stepCount % spectralFreq === 0) {
      this.applySpectralNormalization(net, 1);
    }

    // D. Learning (Reward) - STRATEGY SELECTION REINFORCE
    // Reward based on loss improvement - LSTM learns which strategy to select
    const lossImprovement = this.prevRecentMSE - this.recentMSE;
    const reward = Math.max(-1, Math.min(1, lossImprovement * 50)); // 50x: 0.02 diff = 1.0 reward

    this.metaController.rewardStrategy(reward);

    if (this.stepCount % 50 === 0) {
      // Log every 50 steps
      this.logEvent(
        `[STRATEGY] ${currentStrategy} | LR:${currentParams.learningRate.toFixed(4)} | Reward:${reward.toFixed(3)} | Loss:${this.recentMSE.toFixed(4)}`,
      );
    }

    // 6. Structural Adaptation (L-V + Growth)
    // Only run structure updates if NOT solved/locked
    let status = "STABLE";

    // Using recentMSE for convergence threshold (not smoothed avgLossBuffer)
    // Require sustained low-loss (grace period) before locking to prevent premature SOLVED
    if (this.recentMSE < this.config.solvedThreshold) {
      this.solvedGracePeriod++;
      // Require 100 consecutive low-loss steps before declaring SOLVED
      if (this.solvedGracePeriod >= 100) {
        net.isLocked = true;
        status = "SOLVED";
        net.patienceCounter = 0;
      } else {
        status = "CONVERGING"; // Almost there, but not locked yet
      }
    } else {
      this.solvedGracePeriod = 0; // Reset grace period if loss goes back up

      // UNLOCK MECHANISM: If locked but loss degraded above threshold, resume learning
      if (net.isLocked && this.recentMSE > this.config.unlockThreshold) {
        net.isLocked = false;
        this.logEvent(
          `[UNLOCK] Loss exceeded ${this.config.unlockThreshold.toFixed(4)}, resuming learning.`,
        );
        status = "UNLOCKED";
      } else if (net.isLocked) {
        // Still locked, loss is between solved and unlock thresholds
        status = "SOLVED";
      } else {
        net.isLocked = false;

        // Check trend
        net.lossWindow.push(absError);
        if (net.lossWindow.length > 200) net.lossWindow.shift();

        // Calc Slope
        let slope = 0;
        if (net.lossWindow.length >= 200) {
          const first = net.lossWindow.slice(0, 100).reduce((a, b) => a + b, 0);
          const last = net.lossWindow.slice(100).reduce((a, b) => a + b, 0);
          slope = last - first;
        }
        net.regressionSlope = slope;

        if (slope < this.config.learningSlopeThreshold) {
          status = "LEARNING";
          net.patienceCounter = Math.max(0, net.patienceCounter - 1);
        } else {
          status = "STAGNANT";
          net.patienceCounter++;
        }

        // --- L-V Dynamics ---
        // dn/dt = Growth - Decay
        // Safe, clamped updates
        const D = net.currentTargetDensity;
        const change = dna.lvGrowth * absError - dna.lvDecay * D; // Simple: Error drives growth, Density drives decay
        net.currentTargetDensity = Math.max(0.1, Math.min(0.5, D + change));

        // Decrement Cooldown
        if (this.structuralCooldown > 0) this.structuralCooldown--;

        // ========== COMBINATORIAL EXHAUSTION DETECTION (LOCAL Model Bootstrap) ==========
        // With N neurons and D density, config space ≈ 100^(N²D)
        // If stuck for proportionally long, the network has "searched" enough permutations.
        // Forcing growth adds new dimensions to the search space.

        const configSpaceProxy = N * N * net.currentTargetDensity; // ~number of synapses
        // FIXED: Minimum was 100, way too high for tiny networks
        const exhaustionThreshold = Math.max(20, configSpaceProxy * 30); // 30 steps per synapse, min 20
        const isTiny = N < 16;
        const isExhausted = net.patienceCounter > exhaustionThreshold;
        const isFailingBadly = this.recentMSE > 0.05; // ~22% error - need more capacity

        // ADDITIONAL OVERRIDE: If running for 500+ steps with tiny network and still failing, just grow
        const hasRunLongEnough = this.stepCount > 500;

        if (
          isTiny &&
          (isExhausted || hasRunLongEnough) &&
          isFailingBadly &&
          this.structuralCooldown === 0
        ) {
          // OVERRIDE: Network has probabilistically exhausted its configuration space
          if (this.performMitosis(net)) {
            status = "GROWING (EXHAUSTION)";
            net.patienceCounter = 0;
            this.structuralCooldown = 20; // Even faster bootstrap
            this.logEvent(
              `[EXHAUSTION] N=${N}, patience=${net.patienceCounter}, steps=${this.stepCount} → GROW`,
            );
          }
        }
        // ========== END EXHAUSTION DETECTION ==========

        // ========== DDL-INSPIRED STRUCTURAL CONTROL ==========
        // Synthesize structural intent from the selected strategy
        // logic:
        //   EXPLORE/RESET -> High Growth Bias
        //   STABILIZE -> High Prune Bias
        //   EXPLOIT -> Neutral/Maintenence

        const beta = (adaptiveParams.lvGrowth + adaptiveParams.lvDecay) * 10; // Magnitude based on plasticity
        let actionBias = 0;

        if (currentStrategy === "EXPLORE" || currentStrategy === "RESET") {
          actionBias = 0.8; // Strong growth bias
        } else if (currentStrategy === "STABILIZE") {
          actionBias = -0.8; // Strong prune bias
        } else {
          actionBias = 0.0; // Neutral
        }

        // Placeholder for attention (random for now until we add attention output back)
        const attention = new Array(this.config.maxNeurons)
          .fill(0)
          .map(() => Math.random());

        // const { beta, actionBias, attention } = structuralAction;

        // Only act if beta exceeds skip threshold AND not in cooldown
        if (beta > 0.3 && this.structuralCooldown === 0) {
          if (actionBias > 0.3) {
            // GROW: LSTM decided network needs more capacity
            if (this.performMitosis(net)) {
              status = "GROWING (DDL)";
              // Cooldown scales with beta (higher beta = more confident = shorter cooldown)
              this.structuralCooldown = Math.floor(200 / beta);
              this.logEvent(
                `[DDL-GROW] β=${beta.toFixed(2)}, action=${actionBias.toFixed(2)}`,
              );
            }
          } else if (actionBias < -0.3) {
            // PRUNE: LSTM decided network has excess capacity
            // Use attention to target specific neurons for pruning
            const topAttentionIdx = attention.indexOf(Math.max(...attention));
            this.deltaBasedPrune(net, attention, beta);
            status = "PRUNING (DDL)";
            this.structuralCooldown = Math.floor(100 / beta);
            this.logEvent(
              `[DDL-PRUNE] β=${beta.toFixed(2)}, target=${topAttentionIdx}`,
            );
          } else {
            // REWIRE: Neutral action bias = redistribute connections
            this.performSmartRewiring(
              net,
              Math.floor(net.currentSize * beta * 0.01),
            );
            status = "REWIRING (DDL)";
            this.structuralCooldown = Math.floor(50 / beta);
          }
        }
        // ========== END DDL STRUCTURAL CONTROL ==========

        // Synapse Regrowth (Filling the L-V Deficit)
        const targetConns = Math.floor(N * N * net.currentTargetDensity);
        // Count active
        let active = 0;
        for (let i = 0; i < N * maxN; i++) if (net.weights[i] !== 0) active++;

        // Only allow synapse regrowth if not in deep cooldown (allow small trickle?)
        // Allow "healing" but not "reshaping" during cooldown
        if (active < targetConns && this.structuralCooldown < 100) {
          this.regrowSynapses(net, targetConns - active);
          if (!status.includes("GROW") && !status.includes("OPTIMIZE"))
            status = "GROWING_SYNAPSES";
        }

        // Synapse Pruning (Heartbeat) - REDUCED frequency (every 500 steps)
        // BLOCKED if in cooldown
        if (this.stepCount % 500 === 0 && this.structuralCooldown === 0) {
          this.pruneSynapses(net);
          // No major cooldown for heartbeat, just momentary
        }
      }
    }

    this.stepCount++;

    return this.getMetrics(prediction, targetVal, absError, status);
  }

  // --- Helpers ---

  private performMitosis(net: NetworkState): boolean {
    if (net.currentSize >= this.config.maxNeurons) return false;

    const newIdx = net.currentSize;
    net.currentSize++;

    // Init new neuron
    net.activations[newIdx] = 0;
    net.prevActivations[newIdx] = 0;
    net.neuronEnergy[newIdx] = 1.0;
    net.readout[newIdx] = 0;
    net.neuronTypes[newIdx] = Math.random() < 0.8 ? 1 : -1;

    // Note: inputWeights initialization moved down to use Hebbian logic

    // Connect (Sparse)
    const density = net.currentTargetDensity;
    const spectral = net.liveHyperparams.spectral;
    const maxN = this.config.maxNeurons;

    // Initialize weights with Hebbian alignment to current input signal
    // If current signal is high, start with positive weight to "catch" it.
    const currentInput = this.lastInput;
    net.inputWeights[newIdx] =
      (Math.random() < 0.5 ? 1 : -1) * net.liveHyperparams.inputScale;
    if (Math.abs(currentInput) > 0.1) {
      // Hebbian nudge: Align with signal sign
      net.inputWeights[newIdx] =
        Math.sign(currentInput) * net.liveHyperparams.inputScale * 0.5;
    }

    // Connect (Activity-Based Preferential Attachment)
    // Grow towards "Food" (High Activation Neurons)
    const modifiedRows = new Set<number>();
    modifiedRows.add(newIdx); // Always normalize the new neuron

    for (let i = 0; i < net.currentSize; i++) {
      // Get candidate activity (Energy)
      const activity = Math.abs(net.activations[i]);
      // Sigmoid probability: High activity = High chance of connection
      // Base chance = density. Boosted by activity.
      const connectionProb = density * (1.0 + Math.tanh(activity * 2));

      // Inbound (They talk to me)
      if (Math.random() < connectionProb) {
        net.weights[newIdx * maxN + i] = (Math.random() * 2 - 1) * spectral;
      }
      // Outbound (I talk to them)
      if (Math.random() < connectionProb) {
        net.weights[i * maxN + newIdx] = (Math.random() * 2 - 1) * spectral;
        modifiedRows.add(i); // MARK FOR NORMALIZATION
      }
    }

    // --- NEW: IMMEDIATE ROW NORMALIZATION (The "Group Norm" principle) ---
    // Ensure ALL touched neurons comply with the spectral radius immediately.
    // We do NOT wait for the global Sinkhorn clutch.
    for (const rowIdx of modifiedRows) {
      this.normalizeSingleRow(net, rowIdx, net.liveHyperparams.spectral);
    }
    // -------------------------------------------------------------------

    this.logEvent(`[MITOSIS] Neuron ${newIdx} added.`);

    this.logEvent(`[MITOSIS] Neuron ${newIdx} added.`);

    // We can lower the global Sinkhorn iterations since we did the local fix
    this.applySpectralNormalization(net, 1);
    return true;
  }

  private regrowSynapses(net: NetworkState, count: number): void {
    const maxN = this.config.maxNeurons;
    const N = net.currentSize;

    // [ACCELERATION] Dynamic Limit
    // Instead of fixed 50, scale based on deficit size.
    // Allow filling 10% of the requested deficit per step, capped at 500.
    // For 1024 neurons with 20% target density (~200k synapses), this fills much faster.
    const dynamicLimit = Math.max(50, Math.min(500, Math.floor(count * 0.1)));
    const limit = Math.min(count, dynamicLimit);

    let added = 0;
    const modifiedRows = new Set<number>();

    for (let k = 0; k < limit; k++) {
      const i = Math.floor(Math.random() * N);
      const j = Math.floor(Math.random() * N);
      const idx = i * maxN + j;
      if (net.weights[idx] === 0) {
        // Weight = small random value * spectral
        // Scaled down to prevent explosion
        net.weights[idx] =
          (Math.random() * 2 - 1) * (net.liveHyperparams.spectral * 0.1);
        added++;
        modifiedRows.add(i); // i is the Row (Receiver)
      }
    }
    net.totalRegrown += added;

    // CRITICAL: Renormalize modified rows immediately
    if (added > 0) {
      for (const rowIdx of modifiedRows) {
        this.normalizeSingleRow(net, rowIdx, net.liveHyperparams.spectral);
      }
    }
  }

  /**
   * Smart Regrowth: Connects 'Active' neurons to 'Inactive' ones to spread information,
   * rather than purely random connections. "Wake up the sleeping neurons."
   */
  private performSmartRewiring(net: NetworkState, count: number): void {
    const maxN = this.config.maxNeurons;
    const N = net.currentSize;
    const limit = Math.min(count, 50);

    // Identify Candidates
    const activeIndices: number[] = [];
    const inactiveIndices: number[] = [];

    for (let i = 0; i < N; i++) {
      const act = Math.abs(net.activations[i]);
      if (act > 0.1) activeIndices.push(i);
      else inactiveIndices.push(i);
    }

    let added = 0;
    const modifiedRows = new Set<number>();

    // Strategy 1: Wake up the dead (Connect Active -> Inactive)
    // This prevents adding neurons when we have "free real estate" already.
    if (activeIndices.length > 0 && inactiveIndices.length > 0) {
      for (let k = 0; k < limit && added < limit; k++) {
        const source =
          activeIndices[Math.floor(Math.random() * activeIndices.length)];
        const target =
          inactiveIndices[Math.floor(Math.random() * inactiveIndices.length)];
        // FIX: Wiring Direction was backwards!
        // We want Active (Source) -> Inactive (Target)
        // In our 1D array, index is `row * maxN + col`.
        // If `row` is the receiver (post-synaptic) and `col` is the sender (pre-synaptic),
        // then `target` should be the row and `source` should be the col.
        const idx = target * maxN + source;

        if (net.weights[idx] === 0) {
          // Gentle nudge - extremely small weight (1%) to test the connection without shock
          net.weights[idx] =
            (Math.random() * 2 - 1) * (net.liveHyperparams.spectral * 0.01);
          added++;
          modifiedRows.add(target); // Target is the receiver (Row)
        }
      }
    }

    // Fallback: If everyone is active (High Entropy) or everyone is dead,
    // fallback to standard random exploration to find new loops.
    if (added < limit) {
      this.regrowSynapses(net, limit - added);
    }

    if (added > 0) {
      this.logEvent(
        `[ENGINEERING] Rewired ${added} synapses to optimize capacity.`,
      );
      // CRITICAL: Strictly enforce spectral radius for modified neurons immediately
      for (const rowIdx of modifiedRows) {
        this.normalizeSingleRow(net, rowIdx, net.liveHyperparams.spectral);
      }
    }
  }

  /**
   * DDL-INSPIRED ATTENTION-WEIGHTED PRUNING
   * Uses learned attention to target specific neurons for synapse removal.
   * @param net Network state
   * @param attention Softmax attention over neurons (higher = more likely to prune)
   * @param beta Magnitude of pruning action
   */
  private deltaBasedPrune(
    net: NetworkState,
    attention: number[],
    beta: number,
  ): void {
    const N = net.currentSize;
    const maxN = this.config.maxNeurons;

    // How many synapses to prune? Proportional to beta
    const targetPruneCount = Math.floor(N * beta * 0.05); // ~5% of network per beta unit
    let pruned = 0;

    // For each neuron, prune synapses proportional to attention
    for (let i = 0; i < N && pruned < targetPruneCount; i++) {
      const neuronAttention = attention[i] || 0;

      // Skip neurons with very low attention
      if (neuronAttention < 1.0 / N) continue;

      for (let j = 0; j < N && pruned < targetPruneCount; j++) {
        const idx = i * maxN + j;
        const w = net.weights[idx];

        if (w !== 0 && Math.abs(w) < this.config.pruneThreshold) {
          // Check random with attention probability
          if (Math.random() < neuronAttention * beta * 0.5) {
            net.weights[idx] = 0;
            pruned++;
          }
        }
      }
    }

    if (pruned > 0) {
      this.logEvent(
        `[DDL-PRUNE] Removed ${pruned} synapses via attention targeting.`,
      );
    }
  }

  private pruneSynapses(net: NetworkState): void {
    // [SAFETY] "DO NO HARM" PROTOCOL
    // If we are above the unlock threshold, we are STRUGGLING.
    // Pruning now only causes instability - conserve every synapse we have.
    if (this.recentMSE > this.config.unlockThreshold) {
      // this.logEvent(`[PRUNE SKIPPED] Loss too high (${this.recentMSE.toFixed(4)}). Conserving mass.`);
      return;
    }

    // Lotka-Volterra Selection: Prune based on Fitness + Intrinsic Merit
    // Rule: Keep if |w| + r_i > threshold
    const threshold = this.config.pruneThreshold;
    const N = net.currentSize;
    const maxN = this.config.maxNeurons;

    let pruned = 0;

    // Iterate per neuron (Row-wise processing)
    for (let i = 0; i < N; i++) {
      let removedSum = 0;
      const keptIndices: number[] = [];

      // Calculate Intrinsic Merit ($r_i$) for the Source Neuron (i)
      // 1. Quality: Is this a "Kernel Knot"? (High Readout Weight)
      const readoutContribution = Math.abs(net.readout[i]);
      const qualityBonus = readoutContribution * 0.5; // Strong protection for knots

      for (let j = 0; j < N; j++) {
        const idx = i * maxN + j;
        const w = net.weights[idx];

        if (w !== 0) {
          // 2. Novelty/Diversity: Is this a "Bridge" connection? (E->I or I->E)
          // Encourages signal transfer between neuron types
          const isDiverse = net.neuronTypes[i] !== net.neuronTypes[j];
          const diversityBonus = isDiverse ? 0.005 : 0;

          // 3. Engram Protection (Delta Gating): Protect stable features
          // If trace is high, this synapse is an entrenched "Engram".
          const engramBonus = net.engramTrace[idx] * 0.05;

          // Total Intrinsic Growth Rate ($r_i$)
          const intrinsicMerit = qualityBonus + diversityBonus + engramBonus;
          const magnitude = Math.abs(w);

          // LV Selection Check
          if (magnitude + intrinsicMerit < threshold) {
            removedSum += magnitude; // Accumulate ONLY the weight magnitude (energy)
            net.weights[idx] = 0;
            net.engramTrace[idx] = 0; // Clear trace for dead synapse
            pruned++;
          } else {
            keptIndices.push(idx);
          }
        }
      }

      // Pass 2: Redistribute "Ghost" Energy (Homeostatic Plasticity)
      // If we removed weight, add that magnitude back to the remaining strong weights
      // This preserves the total drive/energy of the neuron (Conservation of Energy)
      if (removedSum > 0 && keptIndices.length > 0) {
        // ENERGY REDISTRIBUTION REMOVED: Caused instability (Loss 0.07 -> 8.9)
        // We just prune the weak ones. The system will regrow if needed.
        // We DO normalize the row to ensure we don't drop TOO far below spectral density if we want to maintain activity?
        // Actually, standard Echo State Networks don't re-normalize after pruning usually, but we are maintaining a spectral radius.
        // Let's just Enforce Upper Bound.
        // If we trimmed mass, we are safer (less gain). No need to normalize UP.
        // Normalizing UP (Redistribution) caused the explosion.
      }
    }

    if (pruned > 0) {
      this.logEvent(`[PRUNE] Removed ${pruned} weak synapses.`);
      this.metaController.resetShortTermMemory(); // New topology requires fresh momentum
    }
  }

  private applySupervisedHebbian(
    net: NetworkState,
    error: number,
    learningRate: number,
  ): void {
    const N = net.currentSize;
    const maxN = this.config.maxNeurons;
    // Clamp error to avoid massive shocks
    const clampedError = Math.max(-1, Math.min(1, error));
    // const targetSpectral = net.liveHyperparams.spectral; // Not directly used here, but good context

    // For Feedback Alignment, we need to calculate the local error signal
    // This simulates the "Nudge" from EqProp by backpropagating the error through feedback weights.
    // The feedbackWeights are typically the transpose of the forward weights, or learned separately.
    // For simplicity here, we assume `net.feedbackWeights` is an array of N values,
    // where each `feedbackWeights[i]` represents the influence of the global error on neuron `i`.

    for (let i = 0; i < N; i++) {
      // Local Error Signal for neuron 'i'
      const localError = clampedError * net.feedbackWeights[i];

      // Skip updates if local error is tiny
      if (Math.abs(localError) < 0.001) continue;

      for (let j = 0; j < N; j++) {
        const idx = i * maxN + j;
        const w = net.weights[idx];

        // Only update existing connections
        if (w !== 0) {
          const x = net.prevActivations[j]; // Pre-synaptic activation (input from J)

          // Contrastive Hebbian Approximation (Feedback Alignment inspired)
          // dW = η * (LocalError * Input) - Decay
          let delta = learningRate * localError * x;

          // Add a simple weight decay term to prevent weights from growing indefinitely
          // This acts as a soft bound, similar to the -y^2*w term in Oja's rule.
          delta -= learningRate * 0.001 * w;

          // Clamp Delta to prevent single-step shocks
          delta = Math.max(-0.01, Math.min(0.01, delta));

          net.weights[idx] += delta;

          // Update Engram Trace (Strength of Hebbian link)
          // If a synapse is being actively updated, its trace should increase,
          // indicating it's an important, active connection.
          net.engramTrace[idx] = Math.min(
            1.0,
            net.engramTrace[idx] * 0.99 + Math.abs(delta) * 10,
          );
        }
      }
      // Note: We do NOT re-normalize every step here, relying on the global Spectral Norm
      // to catch drifts every few steps. This allows "learning transients" to exist briefly.
      // If per-row normalization is desired, it would be called here:
      // this.normalizeSingleRow(net, i, targetSpectral);
    }
  }

  private applySpectralNormalization(
    net: NetworkState,
    iterations: number = 1,
  ): void {
    const N = net.currentSize;
    const maxN = this.config.maxNeurons;
    const targetSpectral = net.liveHyperparams.spectral;

    // Power Iteration to estimate largest singular value (sigma)
    // u = Wv / ||Wv||
    // v = W^T u / ||W^T u||
    // sigma = u^T W v

    // We use persistent u and v vectors (in net state) to accelerate convergence across steps

    for (let iter = 0; iter < iterations; iter++) {
      // 1. v = W^T * u
      // W is stored row-major: W[row][col]
      // W^T * u => component j of v is sum_i( W[i][j] * u[i] )
      let vNorm = 0;
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let i = 0; i < N; i++) {
          sum += net.weights[i * maxN + j] * net.spectralU[i];
        }
        net.spectralV[j] = sum;
        vNorm += sum * sum;
      }
      vNorm = Math.sqrt(vNorm);
      // Normalize v
      if (vNorm > 1e-9) {
        for (let j = 0; j < N; j++) net.spectralV[j] /= vNorm;
      } else {
        // Degenerate case, reset
        for (let j = 0; j < N; j++) net.spectralV[j] = Math.random() - 0.5;
      }

      // 2. u = W * v
      // component i of u is sum_j( W[i][j] * v[j] )
      let uNorm = 0;
      for (let i = 0; i < N; i++) {
        let sum = 0;
        for (let j = 0; j < N; j++) {
          sum += net.weights[i * maxN + j] * net.spectralV[j];
        }
        net.spectralU[i] = sum;
        uNorm += sum * sum;
      }
      uNorm = Math.sqrt(uNorm);
      // Normalize u
      if (uNorm > 1e-9) {
        for (let i = 0; i < N; i++) net.spectralU[i] /= uNorm;
      } else {
        for (let i = 0; i < N; i++) net.spectralU[i] = Math.random() - 0.5;
      }
    }

    // 3. Estimate Sigma (Rayleigh Quotient): sigma = u^T W v
    // Since u = W v / ||W v||, actually ||W v|| is a good estimate if v is converged.
    // Let's compute explicit sigma = u . (W v). We already computed (W v) as un-normalized U in step 2.
    // But we normalized it. So sigma is essentially the uNorm from step 2 *if* v was singular vector.
    // Let's do the rigorous dot product u . (Wv) just to be sure, or rely on uNorm.
    // uNorm is ||W v||. If v is right singular vector, ||W v|| = ||sigma * u|| = sigma.
    // So uNorm is the spectral radius estimate.

    let sigma = 0;
    // Re-verify sigma = u^T * (W * v)
    for (let i = 0; i < N; i++) {
      // W * v term for row i
      let Wv_i = 0;
      for (let j = 0; j < N; j++) {
        Wv_i += net.weights[i * maxN + j] * net.spectralV[j];
      }
      sigma += net.spectralU[i] * Wv_i;
    }

    // Save for UI
    net.currentSpectralRadius = sigma;

    // 4. Normalize
    // If sigma > target, scale down W
    if (sigma > targetSpectral && sigma > 0) {
      const scale = targetSpectral / sigma;
      // Apply scale to ENTIRE matrix
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          net.weights[i * maxN + j] *= scale;
        }
      }
      // Update the estimate too so we don't normalize twice
      net.currentSpectralRadius = targetSpectral;

      // Also scale u and v (?) No, they are unit vectors.
    }
  }

  private getMetrics(
    prediction: number,
    target: number,
    loss: number,
    status: string,
  ): SimulationMetrics {
    // Simple wrapper for UI
    const net = this.network;
    let active = 0;
    for (let i = 0; i < net.weights.length; i++)
      if (net.weights[i] !== 0) active++;

    return {
      loss,
      avgLoss: this.recentMSE, // HONEST 20-sample average for UI (not over-smoothed)
      activeConnections: active,
      targetConnections: Math.floor(
        net.currentSize * net.currentSize * net.currentTargetDensity,
      ),
      regrown: net.totalRegrown,
      prediction,
      target,
      neuronCount: net.currentSize,
      patience: net.patienceCounter,
      spectralRadius: net.currentSpectralRadius,
      adaptationStatus: status,
      step: this.stepCount,
      dna: { ...net.liveHyperparams },
      metaController: this.metaController.getState(),
      logs: this.logQueue.splice(0, this.logQueue.length),
    };
  }

  // "Group Normalization" for a single neuron (Row Normalization)
  private normalizeSingleRow(
    net: NetworkState,
    rowIdx: number,
    targetSpectral: number,
  ): void {
    const N = net.currentSize;
    const maxN = this.config.maxNeurons;
    let sum = 0;

    // Calc Row Sum (L1 Norm)
    for (let j = 0; j < N; j++) {
      sum += Math.abs(net.weights[rowIdx * maxN + j]);
    }

    if (sum > 0) {
      const scale = targetSpectral / sum;
      for (let j = 0; j < N; j++) {
        net.weights[rowIdx * maxN + j] *= scale;
      }
    }
  }

  /**
   * Calculates the Shannon Entropy of the reservoir's activation states.
   * High Entropy = High Information Capacity (Neurons are doing different things).
   * Low Entropy = Redundancy or Death (Neurons are synced or off).
   * Returns 0-1 normalized entropy.
   */
  private calculateNetworkEntropy(net: NetworkState): number {
    const N = net.currentSize;
    if (N === 0) return 0;

    // 1. Discretize activations into bins to estimate probability distribution
    // We use 10 bins between -1 and 1
    const bins = new Float32Array(10).fill(0);

    for (let i = 0; i < N; i++) {
      const val = Math.max(-1, Math.min(1, net.activations[i]));
      const binIdx = Math.min(9, Math.floor((val + 1) * 4.99)); // Map [-1,1] to [0,9]
      bins[binIdx]++;
    }

    // 2. Calculate Entropy: H = -Sum(p * log2(p))
    let entropy = 0;
    for (let i = 0; i < bins.length; i++) {
      const p = bins[i] / N;
      if (p > 0) {
        entropy -= p * Math.log2(p);
      }
    }

    // Normalize entropy to [0, 1] range relative to max possible entropy
    const maxEntropy = Math.log2(bins.length);
    return entropy / maxEntropy;
  }

  private logEvent(msg: string) {
    // Append current loss context to every message for debugging excellence
    // Use recentMSE if available, otherwise 'Init'
    const lossVal = this.recentMSE ? this.recentMSE.toFixed(4) : "Init";
    this.logQueue.push(`${msg} | Loss: ${lossVal}`);
  }

  // Public API
  public getNetworkState() {
    return this.network;
  }
  public getSteps() {
    return this.stepCount;
  }
  public isLocked() {
    return this.network.isLocked;
  }
  public getLSTMDebugState() {
    return this.metaController.getLSTMDebugState();
  }
}
