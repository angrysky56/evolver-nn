export const MACKEY_GLASS_TAU = 17;

export interface Task {
    id: string;
    name: string;
    type: 'FORECAST' | 'CLASSIFY';
    description: string;
    generator: (step: number, history: number[]) => number | { input: number; target: number };
    seedParams?: { leak: number; spectral: number; inputScale: number };
}

export const TASKS: Record<string, Task> = {
    MACKEY_GLASS: {
        id: 'MACKEY_GLASS',
        name: 'Mackey-Glass Chaos',
        type: 'FORECAST',
        description: 'Predicting chaotic fluid dynamics. Needs memory.',
        // seedParams commented out in user code, will use default/fallback
        generator: (_step, history) => {
            const beta = 0.2;
            const gamma = 0.1;
            const n = 10;
            const xt = history[history.length - 1] || 1.2;
            const xtTau = history[history.length - 1 - MACKEY_GLASS_TAU] || 1.2;
            const delta = beta * xtTau / (1 + Math.pow(xtTau, n)) - gamma * xt;
            return xt + delta;
        }
    },
    SINE_WAVE: {
        id: 'SINE_WAVE',
        name: 'Simple Sine Wave',
        type: 'FORECAST',
        description: 'Basic periodic motion. Low difficulty.',
        generator: (step) => Math.sin(step * 0.1)
    },
    SQUARE_WAVE: {
        id: 'SQUARE_WAVE',
        name: 'Square Switch',
        type: 'FORECAST',
        description: 'Abrupt binary switching. Requires fast adaptation.',
        generator: (step) => Math.sin(step * 0.05) > 0 ? 0.8 : -0.8
    },
    TEMPORAL_MNIST: {
        id: 'TEMPORAL_MNIST',
        name: 'Temporal MNIST (0 vs 1)',
        type: 'CLASSIFY',
        description: 'Classify noisy patterns. Needs Neurogenesis to solve.',
        generator: (step) => {
            const patternIdx = Math.floor(step / 100) % 2;
            const localStep = step % 100;
            const noise = (Math.random() * 0.6) - 0.3;

            if (patternIdx === 0) {
                return { input: Math.sin(localStep * 0.2) + noise, target: -0.8 };
            } else {
                return { input: ((localStep % 10) / 5 - 1) + noise, target: 0.8 };
            }
        }
    }
};
