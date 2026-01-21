import React from "react";

interface TrainingStatsBarProps {
  learningRate: number;
  activeSynapses: number;
  targetSynapses: number;
  excitCount: number;
  inhibCount: number;
  spectralRadius: number;
  measuredSpectralRadius: number;
  leak: number;
}

export const TrainingStatsBar: React.FC<TrainingStatsBarProps> = ({
  learningRate,
  activeSynapses,
  targetSynapses,
  excitCount,
  inhibCount,
  spectralRadius,
  measuredSpectralRadius,
  leak,
}) => {
  const total = excitCount + inhibCount;
  const eiRatio = total > 0 ? ((excitCount / total) * 100).toFixed(0) : "0";

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-800 p-2 flex gap-4 items-center justify-between text-xs">
      <div className="flex items-center gap-1">
        <span className="text-slate-500">LR:</span>
        <span className="font-mono text-emerald-400">
          {learningRate.toFixed(3)}
        </span>
      </div>

      <div className="flex items-center gap-1">
        <span className="text-slate-500">Synapses:</span>
        <span className="font-mono text-cyan-400">{activeSynapses}</span>
        <span className="text-slate-600">/ {targetSynapses}</span>
      </div>

      <div className="flex items-center gap-1">
        <span className="text-slate-500">E/I:</span>
        <span className="font-mono text-purple-400">{eiRatio}%</span>
        <span className="text-slate-600">
          ({excitCount}:{inhibCount})
        </span>
      </div>

      <div className="flex items-center gap-1">
        <span className="text-slate-500">ρ:</span>
        <span
          className={`${measuredSpectralRadius > 1.05 ? "text-red-500" : "text-yellow-400"} font-mono`}
        >
          {measuredSpectralRadius.toFixed(2)}
        </span>
        <span className="text-slate-600">/ {spectralRadius.toFixed(2)}</span>
      </div>

      <div className="flex items-center gap-1">
        <span className="text-slate-500">λ:</span>
        <span className="font-mono text-orange-400">{leak.toFixed(2)}</span>
      </div>
    </div>
  );
};
