import React from "react";
import { AlertTriangle, Cpu, Activity, Layers } from "lucide-react";

interface LSTMDebugState {
  shortTermHidden: number[];
  shortTermCell: number[];
  longTermHidden: number[];
  longTermCell: number[];
  gate: number;
  eligibilityTraces: number[];
  outputWeights: number[];
  rawOutputs: number[] | null;
  hiddenSize: number;
  numOutputs: number;
  health: {
    nanCount: number;
    maxTrace: number;
    maxCell: number;
    maxWeight: number;
  };
}

interface Props {
  debugState: LSTMDebugState | null;
}

const NeuronBar: React.FC<{
  values: number[];
  label: string;
  color: string;
  maxVal?: number;
}> = ({ values, label, color, maxVal = 1 }) => {
  return (
    <div className="mb-2">
      <div className="text-[8px] text-slate-500 mb-1">{label}</div>
      <div className="flex gap-[1px] h-4">
        {values.slice(0, 16).map((v, i) => {
          const normalized = Math.min(1, Math.abs(v) / maxVal);
          const isNaN = Number.isNaN(v);
          return (
            <div
              key={i}
              className={`flex-1 rounded-sm ${isNaN ? "bg-red-600 animate-pulse" : ""}`}
              style={{
                backgroundColor: isNaN ? undefined : `${color}`,
                opacity: isNaN ? 1 : 0.2 + normalized * 0.8,
              }}
              title={`Neuron ${i}: ${v?.toFixed(4) ?? "NaN"}`}
            />
          );
        })}
      </div>
    </div>
  );
};

const OutputBar: React.FC<{ values: number[] | null; labels: string[] }> = ({
  values,
  labels,
}) => {
  if (!values) return null;
  return (
    <div className="grid grid-cols-4 gap-1">
      {values.slice(0, 8).map((v, i) => {
        const normalized = Math.abs(v);
        const isPositive = v >= 0;
        const isNaN = Number.isNaN(v);
        return (
          <div key={i} className="text-center">
            <div className="text-[7px] text-slate-600 truncate">
              {labels[i]}
            </div>
            <div
              className={`h-2 rounded-sm ${isNaN ? "bg-red-600 animate-pulse" : ""}`}
              style={{
                backgroundColor: isNaN
                  ? undefined
                  : isPositive
                    ? "#22c55e"
                    : "#ef4444",
                opacity: isNaN ? 1 : 0.3 + Math.min(normalized * 0.2, 0.7),
              }}
            />
          </div>
        );
      })}
    </div>
  );
};

export const LSTMVisualization: React.FC<Props> = ({ debugState }) => {
  if (!debugState) {
    return (
      <div className="bg-slate-950 rounded-lg border border-slate-800 p-3">
        <div className="text-[10px] text-slate-500 italic">
          LSTM not initialized
        </div>
      </div>
    );
  }

  const {
    health,
    shortTermHidden,
    shortTermCell,
    longTermHidden,
    longTermCell,
    rawOutputs,
    gate,
    eligibilityTraces,
    hiddenSize,
    numOutputs,
  } = debugState;
  const outputLabels = [
    "Leak",
    "Spec",
    "InSc",
    "LR",
    "LVG",
    "LVD",
    "Gain",
    "Smth",
  ];

  // Compute trace activity summary
  const avgTraceAbs =
    eligibilityTraces.length > 0
      ? eligibilityTraces.reduce((a, b) => a + Math.abs(b || 0), 0) /
        eligibilityTraces.length
      : 0;

  const isHealthy = health.nanCount === 0;

  return (
    <div className="bg-slate-950 rounded-lg border border-slate-800 p-3 space-y-2">
      {/* Header with health indicator */}
      <div className="flex items-center justify-between">
        <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider flex items-center gap-2">
          <Cpu size={12} className="text-purple-400" />
          LSTM Internals
        </div>
        {!isHealthy && (
          <div className="flex items-center gap-1 text-red-400 text-[9px] animate-pulse">
            <AlertTriangle size={10} />
            {health.nanCount} NaN
          </div>
        )}
      </div>

      {/* Health Stats */}
      <div className="grid grid-cols-4 gap-1 text-[8px]">
        <div className="text-center">
          <div
            className={`font-mono ${health.nanCount > 0 ? "text-red-400" : "text-emerald-400"}`}
          >
            {health.nanCount}
          </div>
          <div className="text-slate-600">NaN</div>
        </div>
        <div className="text-center">
          <div
            className={`font-mono ${health.maxCell > 5 ? "text-orange-400" : "text-slate-300"}`}
          >
            {health.maxCell.toFixed(2)}
          </div>
          <div className="text-slate-600">MaxCell</div>
        </div>
        <div className="text-center">
          <div
            className={`font-mono ${health.maxTrace > 3 ? "text-orange-400" : "text-slate-300"}`}
          >
            {health.maxTrace.toFixed(2)}
          </div>
          <div className="text-slate-600">MaxTrc</div>
        </div>
        <div className="text-center">
          <div
            className={`font-mono ${health.maxWeight > 4 ? "text-orange-400" : "text-slate-300"}`}
          >
            {health.maxWeight.toFixed(2)}
          </div>
          <div className="text-slate-600">MaxWgt</div>
        </div>
      </div>

      {/* Short-Term LSTM */}
      <div className="border-t border-slate-800 pt-2">
        <div className="text-[9px] text-yellow-400 flex items-center gap-1 mb-1">
          <Activity size={8} /> Short-Term LSTM
        </div>
        <NeuronBar
          values={shortTermHidden}
          label="Hidden State"
          color="#facc15"
        />
        <NeuronBar
          values={shortTermCell}
          label="Cell State"
          color="#fef08a"
          maxVal={10}
        />
      </div>

      {/* Long-Term LSTM */}
      <div className="border-t border-slate-800 pt-2">
        <div className="text-[9px] text-cyan-400 flex items-center gap-1 mb-1">
          <Activity size={8} /> Long-Term LSTM
        </div>
        <NeuronBar
          values={longTermHidden}
          label="Hidden State"
          color="#22d3ee"
        />
        <NeuronBar
          values={longTermCell}
          label="Cell State"
          color="#a5f3fc"
          maxVal={10}
        />
      </div>

      {/* Gate Indicator */}
      <div className="border-t border-slate-800 pt-2">
        <div className="flex justify-between items-center text-[9px] mb-1">
          <span className="text-slate-500">Blend Gate</span>
          <span className="font-mono text-slate-300">{gate.toFixed(3)}</span>
        </div>
        <div className="relative w-full bg-slate-800 rounded-full h-2">
          <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/40 to-cyan-500/40 rounded-full" />
          <div
            className="absolute top-0 h-full w-1.5 bg-white rounded-full shadow-lg transition-all"
            style={{ left: `calc(${gate * 100}% - 3px)` }}
          />
        </div>
        <div className="flex justify-between text-[7px] text-slate-600 mt-0.5">
          <span>ST (React)</span>
          <span>LT (Plan)</span>
        </div>
      </div>

      {/* Eligibility Traces Summary */}
      <div className="border-t border-slate-800 pt-2">
        <div className="text-[9px] text-purple-400 flex items-center gap-1 mb-1">
          <Layers size={8} /> Eligibility Traces (Hindsight)
        </div>
        <div className="flex items-center gap-2">
          <div className="flex-1 bg-slate-800 rounded-full h-2">
            <div
              className="h-full bg-purple-500 rounded-full transition-all"
              style={{ width: `${Math.min(avgTraceAbs * 50, 100)}%` }}
            />
          </div>
          <span className="text-[8px] font-mono text-slate-400">
            {avgTraceAbs.toFixed(3)}
          </span>
        </div>
        <div className="text-[7px] text-slate-600 mt-0.5">
          {hiddenSize}×{numOutputs} = {eligibilityTraces.length} traces
        </div>
      </div>

      {/* Raw Outputs Preview */}
      <div className="border-t border-slate-800 pt-2">
        <div className="text-[9px] text-slate-500 mb-1">
          Raw Outputs (pre-sigmoid)
        </div>
        <OutputBar values={rawOutputs} labels={outputLabels} />
      </div>
    </div>
  );
};
