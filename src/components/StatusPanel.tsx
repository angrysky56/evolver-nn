import React from "react";
import { Trophy, Zap, TrendingUp, TrendingDown } from "lucide-react";

interface StatusPanelProps {
  avgLoss: number;
  neuronCount: number;
  maxNeurons: number;
  patience: number;
  patienceLimit: number;
  adaptationStatus: string;
  regrown: number;
}

const getRank = (loss: number) => {
  // Technical performance levels based on RMS error (sqrt of MSE)
  // Loss 0.25 = 50% RMS error, Loss 0.01 = 10% RMS error
  if (loss > 0.25)
    return { label: "Failing", color: "text-red-500", bg: "bg-red-950" }; // >50% error
  if (loss > 0.2)
    return { label: "Poor", color: "text-orange-500", bg: "bg-orange-950" }; // >45% error
  if (loss > 0.1)
    return { label: "Weak", color: "text-yellow-500", bg: "bg-yellow-950" }; // >30% error
  if (loss > 0.05)
    return { label: "Fair", color: "text-slate-400", bg: "bg-slate-700" }; // >22% error
  if (loss > 0.02)
    return { label: "Good", color: "text-cyan-400", bg: "bg-cyan-900" }; // >14% error
  if (loss > 0.01)
    return {
      label: "Excellent",
      color: "text-purple-400",
      bg: "bg-purple-900",
    }; // >10% error
  return {
    label: "Converged",
    color: "text-emerald-400",
    bg: "bg-emerald-900",
  }; // <10% error
};

export const StatusPanel: React.FC<StatusPanelProps> = ({
  avgLoss,
  neuronCount,
  maxNeurons,
  patience,
  patienceLimit,
  adaptationStatus,
  regrown,
}) => {
  const rank = getRank(avgLoss);
  const patiencePercent = (patience / patienceLimit) * 100;

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-800 p-3 space-y-3">
      {/* Rank Badge */}
      <div
        className={`${rank.bg} rounded-lg p-3 flex items-center justify-between`}
      >
        <div className="flex items-center gap-2">
          <Trophy className={rank.color} size={20} />
          <span className={`font-bold ${rank.color}`}>{rank.label}</span>
        </div>
        <span className="font-mono text-sm text-slate-300">
          {avgLoss.toFixed(4)}
        </span>
      </div>

      {/* Brain Size */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">
          Brain Size
        </span>
        <span className="font-mono text-sm text-white">
          {neuronCount} <span className="text-slate-500">/ {maxNeurons}</span>
        </span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-1.5">
        <div
          className="h-full bg-cyan-500 rounded-full transition-all"
          style={{ width: `${(neuronCount / maxNeurons) * 100}%` }}
        />
      </div>

      {/* Growth Pressure */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">
          Growth Pressure
        </span>
        <span
          className={`font-mono text-xs ${patiencePercent > 80 ? "text-orange-400" : "text-slate-400"}`}
        >
          {Math.floor(patience)} / {patienceLimit}
        </span>
      </div>
      <div className="w-full bg-slate-800 rounded-full h-1.5">
        <div
          className={`h-full rounded-full transition-all ${
            adaptationStatus === "LOCKED" ? "bg-purple-500" : "bg-yellow-500"
          }`}
          style={{ width: `${Math.min(patiencePercent, 100)}%` }}
        />
      </div>

      {/* Status & Plasticity */}
      <div className="flex items-center justify-between pt-1 border-t border-slate-800">
        <div className="flex items-center gap-1">
          {adaptationStatus.includes("GROW") ? (
            <TrendingUp size={12} className="text-emerald-400" />
          ) : (
            <TrendingDown size={12} className="text-slate-500" />
          )}
          <span className="text-[10px] text-slate-400">{adaptationStatus}</span>
        </div>
        <div className="flex items-center gap-1">
          <Zap size={12} className="text-emerald-400" />
          <span className="text-[10px] font-mono text-emerald-400">
            {regrown}
          </span>
        </div>
      </div>
    </div>
  );
};
