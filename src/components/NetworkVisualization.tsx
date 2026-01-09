import React, { useRef, useEffect } from 'react';
import { Lock, Activity, Hourglass, Plus } from 'lucide-react';
import { NetworkState } from '../engine/simulationEngine';

interface NetworkVisualizationProps {
  networkState: NetworkState;
  adaptationStatus: string;
}

const getStatusColor = (s: string) => {
  switch (s) {
    case 'LOCKED':
      return 'text-purple-400';
    case 'LEARNING':
      return 'text-emerald-400';
    case 'STAGNANT':
      return 'text-orange-400';
    case 'GROWING':
      return 'text-yellow-400';
    case 'GROWING_SYNAPSES':
      return 'text-blue-400';
    default:
      return 'text-slate-400';
  }
};

export const NetworkVisualization: React.FC<NetworkVisualizationProps> = ({ networkState, adaptationStatus }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const net = networkState;
    const N = net.currentSize;
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) * 0.4;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
    ctx.fillRect(0, 0, w, h);

    // Calc positions
    const positions = [];
    for (let i = 0; i < N; i++) {
        const angle = (i / N) * Math.PI * 2;
        positions.push({
            x: cx + Math.cos(angle) * r,
            y: cy + Math.sin(angle) * r,
        });
    }

    ctx.lineWidth = 1;

    // Edges - visualize signed weights
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const weight = net.weights[i * 512 + j];
        if (weight === 0) continue;

        const signal = net.prevActivations[j] * Math.abs(weight) * net.liveHyperparams.spectral;
        const absSignal = Math.abs(signal);

        if (absSignal > 0.05) {
          ctx.beginPath();
          const opacity = Math.min(absSignal * 2, 0.8);
          // Show positive weights as cyan, negative as red
          ctx.strokeStyle = weight > 0
            ? `rgba(56, 189, 248, ${opacity})`   // Excitatory (positive)
            : `rgba(248, 113, 113, ${opacity})`; // Inhibitory (negative)

          ctx.moveTo(positions[j].x, positions[j].y);
          ctx.lineTo(positions[i].x, positions[i].y);
          ctx.stroke();
        }
      }
    }

    // Nodes
    for (let i = 0; i < N; i++) {
      const val = net.activations[i];
      const intensity = Math.min(Math.abs(val), 1);
      const pos = positions[i];

      // Input effect highlighting
      const inputW = net.inputWeights[i] * net.liveHyperparams.inputScale;
      if (Math.abs(inputW) > 0.1) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
        ctx.strokeStyle = inputW > 0 ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.05)';
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 2 + intensity * 5, 0, Math.PI * 2);
      ctx.fillStyle = val > 0 ? `rgba(56, 189, 248, ${0.4 + intensity})` : `rgba(248, 113, 113, ${0.4 + intensity})`;
      ctx.fill();
    }
  };

  useEffect(() => {
    draw();
  }, [networkState]); // Draw whenever state updates (on frame)

  return (
    <div className="flex-1 relative flex items-center justify-center p-4 bg-slate-950">
      <canvas
        ref={canvasRef}
        width={600}
        height={600}
        className="max-w-full max-h-full rounded-full border border-slate-800/50 shadow-2xl shadow-cyan-900/10"
      />

      {/* Legend */}
      <div className="absolute top-6 left-6 flex flex-col gap-2 pointer-events-none">
        <div className="bg-slate-900/80 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-700 text-xs flex items-center gap-2 shadow-lg">
          <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.5)]"></span>
          Active Path
        </div>

        {/* Dynamic Status Indicator */}
        <div
          className={`bg-slate-900/80 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-700/50 text-xs flex items-center gap-2 shadow-lg ${getStatusColor(
            adaptationStatus
          )}`}
        >
          {adaptationStatus === 'LOCKED' && <Lock size={12} />}
          {adaptationStatus === 'LEARNING' && <Activity size={12} />}
          {adaptationStatus === 'STAGNANT' && <Hourglass size={12} />}
          {adaptationStatus === 'GROWING' && <Plus size={12} />}
          <span className="font-mono uppercase tracking-wide">{adaptationStatus.replace('_', ' ')}</span>
        </div>
      </div>
    </div>
  );
};
