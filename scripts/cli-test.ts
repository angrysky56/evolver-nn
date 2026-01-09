#!/usr/bin/env tsx
/**
 * CLI Testing Script for Evolver-NN
 *
 * Comprehensive testing tool for the Evolutionary Chaos Network simulation.
 * Supports multiple test modes, custom configurations, and formatted output.
 *
 * Usage:
 *   npm run test:quick          # Quick convergence test
 *   npm run test:stability      # Full stability analysis
 *   npm run test:benchmark      # Test multiple configurations
 *   tsx scripts/cli-test.ts --mode custom --steps 1000 --lr 0.05
 */

import { EvolutionaryChaosNetwork, DEFAULT_CONFIG, SimulationConfig } from '../src/engine/simulationEngine';
import { TASKS, Task } from '../src/tasks/tasks';

// ==================== TYPES ====================

type TestMode = 'stability' | 'quick' | 'benchmark' | 'custom';

interface CLIOptions {
    mode: TestMode;
    steps?: number;
    stabilitySteps?: number;
    neurons?: number;
    lr?: number;
    growth?: number;
    task?: string;
    verbose?: boolean;
    json?: boolean;
}

interface TestResult {
    success: boolean;
    lockedAtStep: number;
    finalLoss: number;
    peakDrift: number;
    avgLockedLoss: number;
    maxActivation: number;
    config: Partial<SimulationConfig>;
    crashReason?: string;
}

// ==================== ANSI COLORS ====================

const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    dim: '\x1b[2m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
};

function colorize(text: string, color: keyof typeof colors): string {
    return `${colors[color]}${text}${colors.reset}`;
}

// ==================== ARGUMENT PARSER ====================

function parseArgs(args: string[]): CLIOptions {
    const options: CLIOptions = {
        mode: 'quick',
        verbose: false,
        json: false,
    };

    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        const nextArg = args[i + 1];

        switch (arg) {
            case '--mode':
            case '-m':
                if (['stability', 'quick', 'benchmark', 'custom'].includes(nextArg)) {
                    options.mode = nextArg as TestMode;
                    i++;
                }
                break;
            case '--steps':
            case '-s':
                options.steps = parseInt(nextArg, 10);
                i++;
                break;
            case '--stability-steps':
                options.stabilitySteps = parseInt(nextArg, 10);
                i++;
                break;
            case '--neurons':
            case '-n':
                options.neurons = parseInt(nextArg, 10);
                i++;
                break;
            case '--lr':
            case '--learning-rate':
                options.lr = parseFloat(nextArg);
                i++;
                break;
            case '--growth':
            case '-g':
                options.growth = parseFloat(nextArg);
                i++;
                break;
            case '--task':
            case '-t':
                options.task = nextArg;
                i++;
                break;
            case '--verbose':
            case '-v':
                options.verbose = true;
                break;
            case '--json':
            case '-j':
                options.json = true;
                break;
            case '--help':
            case '-h':
                printHelp();
                process.exit(0);
        }
    }

    return options;
}

function printHelp(): void {
    console.log(`
${colorize('Evolver-NN CLI Testing Tool', 'bright')}

${colorize('USAGE:', 'cyan')}
  npm run test:quick          # Quick convergence test (500 steps)
  npm run test:stability      # Full stability analysis (3000 steps + 1000 stability)
  npm run test:benchmark      # Test multiple configurations
  npm run test                # Interactive mode selection

  tsx scripts/cli-test.ts [options]

${colorize('OPTIONS:', 'cyan')}
  -m, --mode <mode>           Test mode: stability | quick | benchmark | custom
                              Default: quick

  -s, --steps <number>        Number of simulation steps
                              Default: 500 (quick), 3000 (stability)

  --stability-steps <number>  Steps to monitor after Grandmaster lock
                              Default: 1000

  -n, --neurons <number>      Initial neuron count
                              Default: 8

  --lr, --learning-rate <num> Learning rate
                              Default: 0.045

  -g, --growth <number>       LV growth rate
                              Default: 0.1

  -t, --task <name>           Task name (MACKEY_GLASS, etc.)
                              Default: MACKEY_GLASS

  -v, --verbose               Show detailed output

  -j, --json                  Output results as JSON

  -h, --help                  Show this help message

${colorize('EXAMPLES:', 'cyan')}
  # Quick test with custom learning rate
  tsx scripts/cli-test.ts --mode quick --lr 0.05

  # Stability test with more neurons
  tsx scripts/cli-test.ts --mode stability --neurons 16

  # Custom configuration with JSON output
  tsx scripts/cli-test.ts --mode custom --steps 1000 --lr 0.06 --growth 0.15 --json
`);
}

// ==================== TEST RUNNER ====================

function runStabilityTest(
    config: Partial<SimulationConfig>,
    task: Task,
    steps: number,
    stabilitySteps: number,
    verbose: boolean
): TestResult {
    const engine = new EvolutionaryChaosNetwork(task, config);

    let lockedStep = -1;
    let maxActivation = 0;
    let initialLockedLoss = 0;

    if (!verbose) {
        console.log(colorize(`\n🔬 Phase 1: Convergence to Grandmaster (Target Loss < ${config.solvedThreshold})`, 'cyan'));
    }

    let lastAvgLoss = 1.0;

    // Phase 1: Convergence
    for (let step = 0; step < steps; step++) {
        const metrics = engine.step();
        lastAvgLoss = metrics.avgLoss;

        // Track max activation
        const activations = engine.getNetworkState().activations;
        const currentMaxAct = Math.max(...Array.from(activations).map(Math.abs));
        if (currentMaxAct > maxActivation) maxActivation = currentMaxAct;

        // Progress output
        if (verbose && step % 50 === 0) {
            const meta = metrics.metaController;
            const metaStr = `Gate: ${meta.gate.toFixed(2)} | ST: ${meta.shortTermActivity.toFixed(2)} | LT: ${meta.longTermActivity.toFixed(2)}`;
            console.log(`Step ${step}: Loss ${metrics.avgLoss.toFixed(5)} | Status: ${metrics.adaptationStatus} | ${metaStr}`);
        } else if (!verbose && step % 100 === 0) {
            process.stdout.write(`  Step ${step.toString().padStart(4)}: Loss ${metrics.avgLoss.toFixed(5)} | Status: ${metrics.adaptationStatus.padEnd(15)} | Max Act: ${currentMaxAct.toFixed(2).padStart(6)}\r`);
        }

        // Check for Lock
        if (engine.isLocked() && lockedStep === -1) {
            lockedStep = step;
            initialLockedLoss = metrics.avgLoss;
            process.stdout.write('\n');
            console.log(colorize(`\n🏆 GRANDMASTER ACHIEVED at Step ${step}`, 'green'));
            console.log(`   Initial Locked Loss: ${colorize(initialLockedLoss.toFixed(6), 'yellow')}`);
            console.log(colorize(`\n🔍 Phase 2: Stability Drift Check (${stabilitySteps} Steps while Frozen)`, 'cyan'));

            // Phase 2: Stability monitoring
            const stabilityResult = monitorStability(engine, stabilitySteps, verbose);

            return {
                success: true,
                lockedAtStep: lockedStep,
                finalLoss: stabilityResult.finalLoss,
                peakDrift: stabilityResult.peakDrift,
                avgLockedLoss: stabilityResult.avgLoss,
                maxActivation,
                config,
            };
        }

        // Check for explosion
        if (currentMaxAct > 100 || metrics.avgLoss > 10) {
            process.stdout.write('\n');
            console.log(colorize(`\n💥 EXPLOSION DETECTED at Step ${step}`, 'red'));
            return {
                success: false,
                lockedAtStep: -1,
                finalLoss: metrics.avgLoss,
                peakDrift: 0,
                avgLockedLoss: 0,
                maxActivation: currentMaxAct,
                config,
                crashReason: `Gain explosion (Max Activation: ${currentMaxAct.toFixed(2)}, Loss: ${metrics.avgLoss.toFixed(2)})`,
            };
        }
    }

    process.stdout.write('\n');
    if (lockedStep === -1) {
        console.log(colorize(`\n❌ Failed to achieve Grandmaster within ${steps} steps`, 'red'));
        return {
            success: false,
            lockedAtStep: -1,
            finalLoss: lastAvgLoss,
            peakDrift: 0,
            avgLockedLoss: 0,
            maxActivation,
            config,
            crashReason: 'Failed to converge',
        };
    }

    return {
        success: false,
        lockedAtStep: -1,
        finalLoss: 0,
        peakDrift: 0,
        avgLockedLoss: 0,
        maxActivation,
        config,
    };
}

function monitorStability(
    engine: EvolutionaryChaosNetwork,
    steps: number,
    verbose: boolean
): { finalLoss: number; peakDrift: number; avgLoss: number; crashed: boolean } {
    let peakLoss = 0;
    const history: number[] = [];

    // Snapshot readout weights at lock
    const initialReadout = Float32Array.from(engine.getNetworkState().readout);

    for (let i = 0; i < steps; i++) {
        const metrics = engine.step();
        history.push(metrics.avgLoss);

        if (metrics.avgLoss > peakLoss) peakLoss = metrics.avgLoss;

        // Check weight drift
        let weightChange = 0;
        for (let j = 0; j < initialReadout.length; j++) {
            weightChange += Math.abs(engine.getNetworkState().readout[j] - initialReadout[j]);
        }

        if (weightChange > 0.000001 && i === 0) {
            console.log(colorize(`   ⚠️  WARNING: Weights changed immediately after Lock! Delta: ${weightChange}`, 'yellow'));
        }

        if (verbose && i % 50 === 0) {
            const maxAct = Math.max(...Array.from(engine.getNetworkState().activations).map(Math.abs));
            console.log(`GM Step +${i}: Loss ${metrics.avgLoss.toFixed(6)} | Weight Drift: ${weightChange.toFixed(8)} | Peak Act: ${maxAct.toFixed(2)}`);
        } else if (!verbose && i % 100 === 0) {
            const maxAct = Math.max(...Array.from(engine.getNetworkState().activations).map(Math.abs));
            process.stdout.write(`  GM Step +${i.toString().padStart(4)}: Loss ${metrics.avgLoss.toFixed(6)} | Drift: ${weightChange.toFixed(8)} | Peak Act: ${maxAct.toFixed(2).padStart(6)}\r`);
        }

        if (!engine.isLocked()) {
            process.stdout.write('\n');
            console.log(colorize(`\n❌ CRASH: Unlocked at GM Step +${i} (Loss ${metrics.avgLoss.toFixed(5)})`, 'red'));
            return {
                finalLoss: metrics.avgLoss,
                peakDrift: peakLoss,
                avgLoss: history.reduce((a, b) => a + b, 0) / history.length,
                crashed: true,
            };
        }
    }

    process.stdout.write('\n');
    const avgLoss = history.reduce((a, b) => a + b, 0) / history.length;

    console.log(colorize('\n✅ Stability Analysis Complete', 'green'));
    console.log(`   Final Loss: ${colorize(history[history.length - 1].toFixed(6), 'yellow')}`);
    console.log(`   Peak Drift Loss: ${colorize(peakLoss.toFixed(6), 'yellow')}`);
    console.log(`   Average Locked Loss: ${colorize(avgLoss.toFixed(6), 'yellow')}`);

    return {
        finalLoss: history[history.length - 1],
        peakDrift: peakLoss,
        avgLoss,
        crashed: false,
    };
}

function runBenchmark(): void {
    console.log(colorize('\n📊 Running Benchmark Suite\n', 'bright'));

    const configurations = [
        { name: 'Baseline (Restored)', lvGrowth: 0.02, learningRate: 0.02 },
        { name: 'Efficient', lvGrowth: 0.025, learningRate: 0.025 },
        { name: 'High Precision', lvGrowth: 0.015, learningRate: 0.015 },
    ];

    const results: TestResult[] = [];

    for (const cfg of configurations) {
        console.log(colorize(`\n━━━ Testing: ${cfg.name} ━━━`, 'cyan'));
        console.log(`    LV Growth: ${cfg.lvGrowth}, Learning Rate: ${cfg.learningRate}`);

        const config = {
            ...DEFAULT_CONFIG,
            lvGrowth: cfg.lvGrowth,
            learningRate: cfg.learningRate,
        };

        const result = runStabilityTest(config, TASKS.MACKEY_GLASS, 2000, 500, false);
        results.push({ ...result, config: { ...config, name: cfg.name } as any });
    }

    // Print summary table
    console.log(colorize('\n\n📋 Benchmark Results Summary\n', 'bright'));
    console.log('┌─────────────┬─────────────┬──────────────┬─────────────┬──────────────┐');
    console.log('│ Config      │ Converged   │ Locked @Step │ Final Loss  │ Avg Locked   │');
    console.log('├─────────────┼─────────────┼──────────────┼─────────────┼──────────────┤');

    for (const result of results) {
        const configName = (result.config as any).name || 'Unknown';
        const converged = result.success ? colorize('✓ Yes', 'green') : colorize('✗ No', 'red');
        const lockedAt = result.lockedAtStep >= 0 ? result.lockedAtStep.toString().padStart(8) : '    -   ';
        const finalLoss = result.finalLoss > 0 ? result.finalLoss.toFixed(6) : '    -   ';
        const avgLoss = result.avgLockedLoss > 0 ? result.avgLockedLoss.toFixed(6) : '    -   ';

        console.log(`│ ${configName.padEnd(11)} │ ${converged.padEnd(11)} │ ${lockedAt}     │ ${finalLoss}  │ ${avgLoss}   │`);
    }

    console.log('└─────────────┴─────────────┴──────────────┴─────────────┴──────────────┘');
}

// ==================== MAIN ====================

function main(): void {
    const args = process.argv.slice(2);
    const options = parseArgs(args);

    // Print header
    if (!options.json) {
        console.log(colorize('\n╔═══════════════════════════════════════╗', 'bright'));
        console.log(colorize('║  Evolver-NN CLI Testing Tool         ║', 'bright'));
        console.log(colorize('╚═══════════════════════════════════════╝', 'bright'));
    }

    // Benchmark mode is special
    if (options.mode === 'benchmark') {
        runBenchmark();
        return;
    }

    // Determine parameters based on mode
    let steps: number;
    let stabilitySteps: number;
    let config: Partial<SimulationConfig>;

    switch (options.mode) {
        case 'quick':
            steps = options.steps || 1000;
            stabilitySteps = options.stabilitySteps || 200;
            config = {
                ...DEFAULT_CONFIG,
                lvGrowth: options.growth || DEFAULT_CONFIG.lvGrowth,
                learningRate: options.lr || DEFAULT_CONFIG.learningRate,
                initialNeurons: options.neurons || DEFAULT_CONFIG.initialNeurons,
            };
            break;

        case 'stability':
            steps = options.steps || 3000;
            stabilitySteps = options.stabilitySteps || 1000;
            config = {
                ...DEFAULT_CONFIG,
                lvGrowth: options.growth || DEFAULT_CONFIG.lvGrowth,
                learningRate: options.lr || DEFAULT_CONFIG.learningRate,
                initialNeurons: options.neurons || DEFAULT_CONFIG.initialNeurons,
            };
            break;

        case 'custom':
            steps = options.steps || 2300;
            stabilitySteps = options.stabilitySteps || 500;
            config = {
                ...DEFAULT_CONFIG,
                lvGrowth: options.growth || DEFAULT_CONFIG.lvGrowth,
                learningRate: options.lr || DEFAULT_CONFIG.learningRate,
                initialNeurons: options.neurons || DEFAULT_CONFIG.initialNeurons,
            };
            break;

        default:
            steps = 2500;
            stabilitySteps = 200;
            config = DEFAULT_CONFIG;
    }

    const task = TASKS[options.task || 'MACKEY_GLASS'] || TASKS.MACKEY_GLASS;

    if (!options.json) {
        console.log(colorize(`\n⚙️  Configuration:`, 'cyan'));
        console.log(`   Mode: ${colorize(options.mode, 'yellow')}`);
        console.log(`   Steps: ${colorize(steps.toString(), 'yellow')}`);
        console.log(`   Stability Steps: ${colorize(stabilitySteps.toString(), 'yellow')}`);
        console.log(`   Initial Neurons: ${colorize(config.initialNeurons?.toString() || '8', 'yellow')}`);
        console.log(`   Learning Rate: ${colorize(config.learningRate?.toString() || '0.045', 'yellow')}`);
        console.log(`   LV Growth: ${colorize(config.lvGrowth?.toString() || '0.1', 'yellow')}`);
    }

    // Run test
    const result = runStabilityTest(config, task, steps, stabilitySteps, options.verbose || false);

    // Output results
    if (options.json) {
        console.log(JSON.stringify(result, null, 2));
    } else {
        console.log(colorize('\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'bright'));
        if (result.success) {
            console.log(colorize('🎉 TEST PASSED', 'green'));
        } else {
            console.log(colorize('❌ TEST FAILED', 'red'));
            if (result.crashReason) {
                console.log(colorize(`   Reason: ${result.crashReason}`, 'red'));
            }
        }
        console.log(colorize('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n', 'bright'));
    }

    process.exit(result.success ? 0 : 1);
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
