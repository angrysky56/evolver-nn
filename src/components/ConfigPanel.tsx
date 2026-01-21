import React from "react";
import { Settings, Sliders } from "lucide-react";
import { SimulationConfig } from "../engine/simulationEngine";

interface ConfigPanelProps {
  config: SimulationConfig;
  onConfigChange: (key: keyof SimulationConfig, value: number) => void;
  isOpen: boolean;
  onToggle: () => void;
}

interface ConfigSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (v: number) => string;
}

const ConfigSlider: React.FC<ConfigSliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}) => (
  <div className="flex flex-col gap-1">
    <div className="flex justify-between text-[10px]">
      <span className="text-slate-400">{label}</span>
      <span className="font-mono text-cyan-400">
        {format ? format(value) : value}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
    />
  </div>
);

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
  config,
  onConfigChange,
  isOpen,
  onToggle,
}) => {
  if (!isOpen) {
    return (
      <button
        onClick={onToggle}
        className="fixed bottom-4 right-4 p-3 bg-slate-900 border border-slate-700 rounded-full shadow-lg hover:bg-slate-800 transition-colors z-50"
        title="Open Settings"
      >
        <Settings size={20} className="text-cyan-400" />
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 w-80 max-h-[80vh] bg-slate-900 border border-slate-700 rounded-xl shadow-2xl z-50 overflow-hidden flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Sliders size={14} className="text-cyan-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Configuration
          </span>
        </div>
        <button
          onClick={onToggle}
          className="text-slate-500 hover:text-white text-lg leading-none"
        >
          &times;
        </button>
      </div>

      {/* Scrollable Content */}
      <div className="p-4 flex flex-col gap-4 overflow-y-auto">
        {/* Network Structure */}
        <div>
          <h4 className="text-[9px] text-slate-500 uppercase font-bold tracking-widest mb-2">
            Network Structure
          </h4>
          <div className="flex flex-col gap-3">
            <ConfigSlider
              label="Max Neurons"
              value={config.maxNeurons}
              min={16}
              max={1024}
              step={16}
              onChange={(v) => onConfigChange("maxNeurons", v)}
            />
            <ConfigSlider
              label="Initial Neurons"
              value={config.initialNeurons}
              min={1}
              max={64}
              step={1}
              onChange={(v) => onConfigChange("initialNeurons", v)}
            />
          </div>
        </div>

        {/* AUTOMATION WARNING */}
        <div className="p-3 bg-slate-950 rounded border border-slate-800">
          <div className="text-[10px] text-slate-500 mb-1">
            AUTOMATION ENGAGED
          </div>
          <p className="text-[9px] text-slate-400 leading-relaxed">
            Physics parameters (Learning Rate, Leak, etc.) are now fully
            controlled by the
            <span className="text-cyan-400"> Regime Driver</span>. Manual
            overrides have been disabled.
          </p>
        </div>

        {/* Adaptation Targets (Meta-Params only) */}
        <div>
          <h4 className="text-[9px] text-slate-500 uppercase font-bold tracking-widest mb-2">
            Goals
          </h4>
          <div className="flex flex-col gap-3">
            <ConfigSlider
              label="Solved Threshold"
              value={config.solvedThreshold}
              min={0.0001}
              max={0.01}
              step={0.0001}
              onChange={(v) => onConfigChange("solvedThreshold", v)}
              format={(v) => v.toFixed(4)}
            />
          </div>
        </div>
      </div>

      {/* Footer hint */}
      <div className="p-2 text-center text-[9px] text-slate-600 border-t border-slate-800">
        Changes apply on next reset
      </div>
    </div>
  );
};
