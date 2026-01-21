import React from "react";

interface LossHistoryChartProps {
  history: number[];
  height?: number;
  currentStep?: number;
  events?: Array<{ step: number; message: string }>;
}

export const LossHistoryChart: React.FC<LossHistoryChartProps> = ({
  history,
  height = 80,
  currentStep = 0,
  events = [],
}) => {
  if (history.length < 2) return null;

  const maxLoss = Math.max(...history, 0.1);
  const minLoss = Math.min(...history, 0);
  const range = maxLoss - minLoss || 0.1;

  // Create SVG path for loss curve
  const points = history.map((loss, i) => {
    const x = (i / (history.length - 1)) * 100;
    const y = ((maxLoss - loss) / range) * 100;
    return `${x},${y}`;
  });

  const pathD = `M ${points.join(" L ")}`;

  // Calculate current loss color
  const currentLoss = history[history.length - 1] || 0;
  const lossColor =
    currentLoss < 0.01
      ? "#10b981"
      : currentLoss < 0.02
        ? "#eab308"
        : currentLoss < 0.05
          ? "#f97316"
          : "#ef4444";

  // Calculate visible step range
  const historyLength = history.length;
  const startStep = Math.max(0, currentStep - historyLength + 1);

  // Filter and map events to chart coordinates
  const visibleEvents = events.filter(
    (e) => e.step >= startStep && e.step <= currentStep,
  );

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-800 p-3 relative">
      <div className="flex justify-between items-center mb-2">
        <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
          Loss History
        </span>
        <span className="text-xs font-mono" style={{ color: lossColor }}>
          {currentLoss.toFixed(4)}
        </span>
      </div>
      <svg
        width="100%"
        height={height}
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        className="overflow-visible"
      >
        {/* Grid lines */}
        <line
          x1="0"
          y1="25"
          x2="100"
          y2="25"
          stroke="#334155"
          strokeWidth="0.3"
        />
        <line
          x1="0"
          y1="50"
          x2="100"
          y2="50"
          stroke="#334155"
          strokeWidth="0.3"
        />
        <line
          x1="0"
          y1="75"
          x2="100"
          y2="75"
          stroke="#334155"
          strokeWidth="0.3"
        />

        {/* Event Markers */}
        {visibleEvents.map((e, i) => {
          const x = ((e.step - startStep) / (historyLength - 1)) * 100;
          let color = "#94a3b8"; // default slate-400
          let opacity = 0.5;
          let width = 0.5;

          if (e.message.includes("Emergency")) {
            color = "#ef4444";
            opacity = 0.8;
            width = 1;
          } else if (e.message.includes("Stagnation")) {
            color = "#eab308";
            opacity = 0.8;
            width = 1;
          } else if (e.message.includes("[OPTIMIZE]")) {
            color = "#3b82f6";
            opacity = 0.4;
          } else if (e.message.includes("[L-V SELECTION]")) {
            color = "#8b5cf6";
            opacity = 0.4;
          }

          return (
            <line
              key={i}
              x1={x}
              y1="0"
              x2={x}
              y2="100"
              stroke={color}
              strokeWidth={width}
              opacity={opacity}
              strokeDasharray={width === 1 ? "2,2" : undefined}
            />
          );
        })}

        {/* Loss curve */}
        <path
          d={pathD}
          fill="none"
          stroke={lossColor}
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
        />

        {/* Area fill under curve */}
        <path d={`${pathD} L 100,100 L 0,100 Z`} fill={`${lossColor}20`} />
      </svg>
      <div className="flex justify-between text-[9px] text-slate-600 mt-1">
        <span>{startStep % 1000}</span>
        <span className="text-slate-700">
          Loss Range: {minLoss.toFixed(3)} - {maxLoss.toFixed(3)}
        </span>
        <span>{currentStep % 1000}</span>
      </div>
    </div>
  );
};
