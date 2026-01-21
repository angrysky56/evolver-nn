export const MACKEY_GLASS_TAU = 17;

export interface Task {
    id: string;
    name: string;
    type: 'FORECAST' | 'CLASSIFY';
    description: string;
    generator: (step: number, history: number[]) => { input: number; target: number };
    // Optional: Generate initial seed data for this task
    seed?: () => number[];
    seedParams?: { leak: number; spectral: number; inputScale: number };
}

export const TASKS: Record<string, Task> = {
    MACKEY_GLASS: {
        id: 'MACKEY_GLASS',
        name: 'Mackey-Glass Chaos',
        type: 'FORECAST',
        description: 'Predicting chaotic fluid dynamics. Needs memory.',
        seed: () => {
            // Generate chaotic seed data
            const data: number[] = [];
            let val = 1.2;
            for (let i = 0; i < MACKEY_GLASS_TAU + 50; i++) {
                const xtTau = (i >= MACKEY_GLASS_TAU) ? data[i - MACKEY_GLASS_TAU] : 1.2;
                const delta = 0.2 * xtTau / (1 + Math.pow(xtTau, 10)) - 0.1 * val;
                val += delta;
                data.push(val);
            }
            return data;
        },
        generator: (_step, history) => {
            const beta = 0.2;
            const gamma = 0.1;
            const n = 10;
            const xt = history[history.length - 1] || 1.2;
            const xtTau = history[history.length - 1 - MACKEY_GLASS_TAU] || 1.2;
            const delta = beta * xtTau / (1 + Math.pow(xtTau, n)) - gamma * xt;
            const nextVal = xt + delta;
            return { input: xt, target: nextVal };
        }
    },
    SINE_WAVE: {
        id: 'SINE_WAVE',
        name: 'Simple Sine Wave',
        type: 'FORECAST',
        description: 'Basic periodic motion. Low difficulty.',
        generator: (step) => {
            return { input: Math.sin(step * 0.1), target: Math.sin((step + 1) * 0.1) };
        }
    },
    SQUARE_WAVE: {
        id: 'SQUARE_WAVE',
        name: 'Square Switch',
        type: 'FORECAST',
        description: 'Abrupt binary switching. Requires fast adaptation.',
        // Give network the phase info (normalized step position in cycle)
        // This way it can learn WHEN transitions happen
        generator: (step) => {
            const current = Math.sin(step * 0.05) > 0 ? 0.8 : -0.8;
            const next = Math.sin((step + 1) * 0.05) > 0 ? 0.8 : -0.8;
            return { input: current, target: next };
        }
    },
    TEMPORAL_MNIST: {
        id: 'TEMPORAL_MNIST',
        name: 'Temporal MNIST (0 vs 1)',
        type: 'CLASSIFY',
        description: 'Classify noisy patterns. Needs Neurogenesis to solve.',
        // Use deterministic pseudo-random noise based on step (seeded by step number)
        generator: (step) => {
            const patternIdx = Math.floor(step / 100) % 2;
            const localStep = step % 100;
            // Deterministic noise: use sine of step as "random" noise (reproducible)
            const noise = Math.sin(step * 12.9898) * 0.3;

            if (patternIdx === 0) {
                // Pattern 0: Smooth sine wave → classify as -0.8
                return { input: Math.sin(localStep * 0.2) + noise, target: -0.8 };
            } else {
                // Pattern 1: Sawtooth wave → classify as 0.8
                return { input: ((localStep % 10) / 5 - 1) + noise, target: 0.8 };
            }
        }
    }
};
