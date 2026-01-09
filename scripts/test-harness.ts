import { EvolutionaryChaosNetwork, DEFAULT_CONFIG } from '../src/engine/simulationEngine';
import { TASKS } from '../src/tasks/tasks';

// Definitive Stability Test
const SIMULATION_STEPS = 3000;
const STABILITY_CHECK_STEPS = 1000;

function runStabilityAnalysis() {
    console.log("🚀 Starting Definitive Stability Analysis...");
    console.log("------------------------------------------------");

    // 1. Setup - Aggressive Config (User's preference)
    const config = {
        ...DEFAULT_CONFIG,
        lvGrowth: 0.1,
        learningRate: 0.045
    };

    const task = TASKS.MACKEY_GLASS;
    const engine = new EvolutionaryChaosNetwork(task, config);

    let lockedStep = -1;
    let maxActivation = 0;
    let initialLockedLoss = 0;

    console.log(`Phase 1: Convergence to Grandmaster (Target Loss < ${config.solvedThreshold})`);

    for (let step = 0; step < SIMULATION_STEPS; step++) {
        const metrics = engine.step();

        // Track stats
        const activations = engine.getNetworkState().activations;
        const currentMaxAct = Math.max(...activations.map(Math.abs));
        if (currentMaxAct > maxActivation) maxActivation = currentMaxAct;

        if (step % 100 === 0) {
            process.stdout.write(`Step ${step}: Loss ${metrics.avgLoss.toFixed(5)} | Status: ${metrics.adaptationStatus} \r`);
        }

        // Check for Lock
        if (engine.isLocked() && lockedStep === -1) {
            lockedStep = step;
            initialLockedLoss = metrics.avgLoss;
            console.log(`\n\n🏆 GRANDMASTER ACHIEVED at Step ${step}`);
            console.log(`Initial Locked Loss: ${initialLockedLoss.toFixed(6)}`);
            console.log("Entering Phase 2: Stability Drift Check (1000 Steps while Frozen)...");

            // Phase 2: Run while Locked
            monitorGrandmasterStability(engine);
            return;
        }
    }

    if (lockedStep === -1) {
        console.log("\n❌ Failed to achieve Grandmaster within limit. Increase steps or check parameters.");
    }
}

function monitorGrandmasterStability(engine: EvolutionaryChaosNetwork) {
    let peakLoss = 0;
    const history: number[] = [];

    // Snapshot of readout weights at start of lock
    // Only capture reference to array if we want to debug mutability,
    // but we need a COPY to compare drift.
    const initialReadout = Float32Array.from(engine.getNetworkState().readout);

    for (let i = 0; i < STABILITY_CHECK_STEPS; i++) {
        const metrics = engine.step();
        history.push(metrics.avgLoss);

        if (metrics.avgLoss > peakLoss) peakLoss = metrics.avgLoss;

        // Compare readout weights to ensure True Freeze worked
        let weightChange = 0;
        for (let j = 0; j < initialReadout.length; j++) {
            weightChange += Math.abs(engine.getNetworkState().readout[j] - initialReadout[j]);
        }

        if (weightChange > 0.000001 && i === 0) {
            console.log(`\n⚠️ WARNING: Weights changed immediately after Lock! True Freeze failed? Delta: ${weightChange}`);
        }

        if (i % 100 === 0) {
            console.log(`GM Step +${i}: Loss ${metrics.avgLoss.toFixed(6)} | Weight Drift: ${weightChange.toFixed(8)} | Peak Act: ${Math.max(...engine.getNetworkState().activations.map(Math.abs)).toFixed(2)}`);
        }

        if (!engine.isLocked()) {
            console.log(`\n❌ CRASH: Unlocked at GM Step +${i} (Loss ${metrics.avgLoss.toFixed(5)})`);
            console.log("Analysis: The network could not hold the solution.");
            return;
        }
    }

    console.log("\n✅ Stability Analysis Complete");
    console.log(`Final Loss: ${history[history.length - 1].toFixed(6)}`);
    console.log(`Peak Drift Loss: ${peakLoss.toFixed(6)}`);

    // Variance analysis
    const avg = history.reduce((a, b) => a + b, 0) / history.length;
    console.log(`Average Locked Loss: ${avg.toFixed(6)}`);
}

runStabilityAnalysis();
