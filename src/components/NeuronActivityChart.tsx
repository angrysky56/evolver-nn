import React from "react";

interface NeuronActivityChartProps {
  activations: number[];
  neuronTypes: number[]; // 1 = excitatory, -1 = inhibitory
  maxDisplay?: number;
}

export const NeuronActivityChart: React.FC<NeuronActivityChartProps> = ({
  activations,
  neuronTypes,
  maxDisplay = 4096,
}) => {
  const N = Math.min(activations.length, maxDisplay);
  if (N === 0) return null;

  const maxAct = Math.max(...activations.slice(0, N).map(Math.abs), 0.1);

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-800 p-3">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
          Neuron Activity
        </span>
        <span className="text-[9px] text-slate-600">
          {N} / {activations.length} neurons
        </span>
      </div>
      <div className="flex gap-[2px] h-12 items-end">
        {Array.from({ length: N }).map((_, i) => {
          const act = activations[i] || 0;
          const absAct = Math.abs(act);
          const height = (absAct / maxAct) * 100;
          const isExcitatory = neuronTypes[i] === 1;
          const color = isExcitatory ? "#f87171" : "#60a5fa";

          return (
            <div
              key={i}
              className="flex-1 rounded-t transition-all duration-100"
              style={{
                height: `${height}%`,
                backgroundColor: color,
                opacity: 0.4 + (absAct / maxAct) * 0.6,
              }}
              title={`Neuron ${i}: ${act.toFixed(3)} (${isExcitatory ? "E" : "I"})`}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-[9px] text-slate-600 mt-1">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-400" /> Excitatory
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-blue-400" /> Inhibitory
        </span>
      </div>
    </div>
  );
};
