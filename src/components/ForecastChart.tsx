import React, { useRef, useEffect } from 'react';
import { TrendingUp } from 'lucide-react';

interface ForecastChartProps {
  history: Array<{ target: number; prediction: number }>;
  step: number;
}

export const ForecastChart: React.FC<ForecastChartProps> = ({ history, step }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // ... drawing code ...
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Zero line
    ctx.strokeStyle = '#1e293b';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    const mapY = (val: number) => h / 2 - val * h * 0.25;

    // Draw target
    ctx.beginPath();
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    for (let i = 0; i < history.length; i++) {
        const x = (i / (history.length - 1)) * w;
        const y = mapY(history[i].target);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw prediction
    ctx.beginPath();
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2;
    for (let i = 0; i < history.length; i++) {
        const x = (i / (history.length - 1)) * w;
        const y = mapY(history[i].prediction);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  useEffect(() => {
    drawChart();
  }, [history, step]);

  return (
    <div className="flex flex-col flex-1 min-h-[180px]">
      <h3 className="text-slate-400 text-xs font-bold uppercase tracking-widest mb-3 flex items-center gap-2">
        <TrendingUp size={12} /> Live Forecast
      </h3>
      <div className="flex-1 bg-slate-950 border border-slate-800 rounded-xl relative p-0 overflow-hidden shadow-inner">
        <canvas ref={canvasRef} width={340} height={200} className="w-full h-full" />
        <div className="absolute top-2 left-2 flex flex-col gap-1 pointer-events-none">
          <div className="text-[10px] text-cyan-400 font-mono flex items-center gap-1">
            <div className="w-2 h-0.5 bg-cyan-400"></div> Target
          </div>
          <div className="text-[10px] text-purple-400 font-mono flex items-center gap-1">
            <div className="w-2 h-0.5 bg-purple-400"></div> Prediction
          </div>
        </div>
      </div>
    </div>
  );
};
