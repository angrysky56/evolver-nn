import React from 'react';
import { Dna, BrainCircuit, Hourglass, Trophy, Network, Zap, Timer } from 'lucide-react';
import { SimulationMetrics } from '../engine/simulationEngine';

interface MetricsPanelProps {
  metrics: SimulationMetrics;
  maxNeurons: number;
  patienceLimit: number;
}

const getRank = (loss: number) => {
  if (loss > 0.5) return { label: 'Unranked', color: 'text-slate-500' };
  if (loss > 0.2) return { label: 'Novice', color: 'text-slate-400' };
  if (loss > 0.1) return { label: 'Apprentice', color: 'text-cyan-600' };
  if (loss > 0.05) return { label: 'Adept', color: 'text-cyan-400' };
  if (loss > 0.02) return { label: 'Master', color: 'text-purple-400' };
  if (loss > 0.01) return { label: 'Grandmaster', color: 'text-yellow-400' };
  return { label: 'Solved', color: 'text-emerald-400' };
};

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ metrics, maxNeurons, patienceLimit }) => {
  const rank = getRank(metrics.avgLoss);
  const meta = metrics.metaController;

  return (
    <div className="w-96 bg-slate-900 border-l border-slate-800 p-6 flex flex-col gap-6 overflow-y-auto">
      {/* Hyperparameter DNA Panel */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
        <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-2 flex items-center gap-2">
          <Dna size={12} className="text-blue-400" /> Evolving Hyperparameters
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col items-center">
            <div className="text-[9px] text-slate-500 mb-1">Leak Rate</div>
            <div className="text-xs font-mono text-emerald-400">{metrics.dna.leak.toFixed(4)}</div>
          </div>
          <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col items-center">
            <div className="text-[9px] text-slate-500 mb-1">Spectral</div>
            <div className="text-xs font-mono text-purple-400">{metrics.dna.spectral.toFixed(4)}</div>
          </div>
          <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col items-center">
            <div className="text-[9px] text-slate-500 mb-1">Input Scale</div>
            <div className="text-xs font-mono text-blue-400">{metrics.dna.inputScale.toFixed(4)}</div>
          </div>
          <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col items-center">
            <div className="text-[9px] text-slate-500 mb-1">Learning Rate</div>
            <div className="text-xs font-mono text-yellow-400">{metrics.dna.learningRate.toFixed(4)}</div>
          </div>
          <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col items-center">
            <div className="text-[9px] text-slate-500 mb-1">Smooth Factor</div>
            <div className="text-xs font-mono text-orange-400">{metrics.dna.smoothingFactor.toFixed(4)}</div>
          </div>
          <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col items-center">
            <div className="text-[9px] text-slate-500 mb-1">L-V Growth</div>
            <div className="text-xs font-mono text-green-400">{metrics.dna.lvGrowth.toFixed(4)}</div>
          </div>
          <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col items-center col-span-2">
            <div className="text-[9px] text-slate-500 mb-1">L-V Decay</div>
            <div className="text-xs font-mono text-red-400">{metrics.dna.lvDecay.toFixed(5)}</div>
          </div>
        </div>
      </div>

      {/* Bicameral Meta-Controller Panel */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
        <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-3 flex items-center gap-2">
          <BrainCircuit size={12} className="text-purple-400" /> Bicameral Controller
        </div>
        <div className="grid grid-cols-2 gap-3 mb-3">
          {/* Short-Term */}
          <div className="bg-slate-950 p-2 rounded border border-slate-800">
            <div className="flex items-center gap-1 mb-1">
              <Zap size={10} className="text-yellow-400" />
              <span className="text-[9px] text-slate-500">Short-Term</span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
              <div
                className="h-full bg-yellow-400 transition-all duration-200"
                style={{ width: `${Math.min(meta.shortTermActivity * 100, 100)}%` }}
              />
            </div>
            <div className="text-[9px] font-mono text-yellow-400/70 text-right mt-0.5">
              {meta.shortTermActivity.toFixed(2)}
            </div>
          </div>

          {/* Long-Term */}
          <div className="bg-slate-950 p-2 rounded border border-slate-800">
            <div className="flex items-center gap-1 mb-1">
              <Timer size={10} className="text-cyan-400" />
              <span className="text-[9px] text-slate-500">Long-Term</span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
              <div
                className="h-full bg-cyan-400 transition-all duration-200"
                style={{ width: `${Math.min(meta.longTermActivity * 100, 100)}%` }}
              />
            </div>
            <div className="text-[9px] font-mono text-cyan-400/70 text-right mt-0.5">
              {meta.longTermActivity.toFixed(2)}
            </div>
          </div>
        </div>

        {/* Gate */}
        <div className="bg-slate-950 p-2 rounded border border-slate-800">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[9px] text-slate-500">Gate (ST ← → LT)</span>
            <span className="text-[9px] font-mono text-slate-400">{meta.gate.toFixed(2)}</span>
          </div>
          <div className="relative w-full bg-slate-800 rounded-full h-2 overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/30 to-cyan-500/30" />
            <div
              className="absolute top-0 h-full w-1 bg-white rounded-full shadow-lg transition-all duration-200"
              style={{ left: `calc(${meta.gate * 100}% - 2px)` }}
            />
          </div>
        </div>
      </div>

      {/* Neurogenesis Panel */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-3">
        <div className="flex items-center justify-between mb-2">
          <div>
            <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1 flex items-center gap-2">
              <BrainCircuit size={12} /> Brain Size
            </div>
            <div className="text-xl font-mono text-white leading-none">
              {metrics.neuronCount} <span className="text-sm text-slate-500">/ {maxNeurons}</span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1 flex items-center justify-end gap-1">
              <Hourglass size={10} /> Growth Pressure
            </div>
            <div
              className={`text-xs font-mono ${
                metrics.patience > patienceLimit * 0.8 ? 'text-orange-400' : 'text-slate-400'
              }`}
            >
              {Math.floor(metrics.patience)} / {patienceLimit}
            </div>
          </div>
        </div>

        {/* Patience Bar */}
        <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ease-linear ${
              metrics.adaptationStatus === 'LOCKED' ? 'bg-purple-500' : 'bg-yellow-500'
            }`}
            style={{ width: `${Math.min(100, (metrics.patience / patienceLimit) * 100)}%` }}
          />
        </div>
      </div>

      {/* Rank Badge */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-full bg-slate-900 border border-slate-800 ${rank.color}`}>
            <Trophy size={20} />
          </div>
          <div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">Competition Status</div>
            <div className={`font-bold ${rank.color}`}>{rank.label}</div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-[10px] text-slate-500">Avg Loss</div>
          <div className="font-mono text-sm text-slate-300">{metrics.avgLoss.toFixed(4)}</div>
        </div>
      </div>

      {/* Network Health */}
      <div>
        <h3 className="text-slate-400 text-xs font-bold uppercase tracking-widest mb-3 flex items-center gap-2">
          <Network size={12} /> L-V Dynamics
        </h3>

        <div className="grid grid-cols-2 gap-3">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <div className="text-slate-500 text-[10px] mb-1">Active Synapses</div>
            <div className="text-xl font-mono text-slate-100">{metrics.activeConnections}</div>
            <div className="text-[10px] text-slate-600 mt-1 flex justify-between">
              <span>Dyn. Target:</span>
              <span className="text-cyan-500">{metrics.targetConnections}</span>
            </div>
          </div>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 relative overflow-hidden group">
            <div className="text-slate-500 text-[10px] mb-1">Plasticity Events</div>
            <div className="text-xl font-mono text-emerald-400">{metrics.regrown}</div>
            <div className="text-[10px] text-emerald-900 mt-1 group-hover:text-emerald-700 transition-colors">
              Regrowth
            </div>
          </div>
        </div>
      </div>

      <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
        <div className="flex justify-between items-center text-xs text-slate-400 mb-1">
          <span>Current Error (MSE)</span>
          <span className={`font-mono ${metrics.loss < 0.01 ? 'text-emerald-400' : 'text-orange-400'}`}>
            {metrics.loss.toFixed(5)}
          </span>
        </div>
        <div className="w-full bg-slate-900 rounded-full h-1.5 overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              metrics.loss < 0.01 ? 'bg-emerald-500' : 'bg-orange-500'
            }`}
            style={{ width: `${Math.min(metrics.loss * 1000, 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
};

