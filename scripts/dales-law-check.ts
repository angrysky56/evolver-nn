import { EvolutionaryChaosNetwork } from '../src/engine/simulationEngine';

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

export function validateDalesLaw(engine: EvolutionaryChaosNetwork): boolean {
    const state = engine.getNetworkState();
    const N = state.currentSize;
    let violations = 0;

    for (let i = 0; i < N; i++) {
        const type = state.neuronTypes[i]; // 1 or -1
        if (type === 0) continue; // Should not happen if initialized correctly

        for (let j = 0; j < N; j++) {
            const weight = state.weights[i * 512 + j];
            if (weight === 0) continue;

            if ((type === 1 && weight < 0) || (type === -1 && weight > 0)) {
                violations++;
            }
        }
    }

    if (violations > 0) {
        console.log(colorize(`\n❌ DALE'S LAW VIOLATION: ${violations} synapses have incorrect sign!`, 'red'));
        return false;
    }
    return true;
}
