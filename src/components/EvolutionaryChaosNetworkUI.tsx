import React, { useState, useEffect, useRef } from "react";
import {
  EvolutionaryChaosNetwork,
  SimulationMetrics,
  NetworkState,
  SimulationConfig,
  DEFAULT_CONFIG,
} from "../engine/simulationEngine";
import { TASKS, Task } from "../tasks/tasks";
import { ControlHeader } from "./ControlHeader";
import { NetworkVisualization } from "./NetworkVisualization";
import { ForecastChart } from "./ForecastChart";
import { ConfigPanel } from "./ConfigPanel";
import { LossHistoryChart } from "./LossHistoryChart";
import { NeuronActivityChart } from "./NeuronActivityChart";
import { TrainingStatsBar } from "./TrainingStatsBar";
import { LSTMVisualization } from "./LSTMVisualization";
import {
  Dna,
  BrainCircuit,
  Trophy,
  Network,
  Zap,
  Timer,
  TrendingUp,
  Terminal,
} from "lucide-react";

const getRank = (loss: number) => {
  // Technical performance levels based on RMS error
  if (loss > 0.25)
    return { label: "Failing", color: "text-red-500", bg: "bg-red-950" };
  if (loss > 0.2)
    return { label: "Poor", color: "text-orange-500", bg: "bg-orange-950" };
  if (loss > 0.1)
    return { label: "Weak", color: "text-yellow-500", bg: "bg-yellow-950" };
  if (loss > 0.05)
    return { label: "Fair", color: "text-slate-400", bg: "bg-slate-700" };
  if (loss > 0.02)
    return { label: "Good", color: "text-cyan-400", bg: "bg-cyan-900" };
  if (loss > 0.01)
    return {
      label: "Excellent",
      color: "text-purple-400",
      bg: "bg-purple-900",
    };
  return {
    label: "Converged",
    color: "text-emerald-400",
    bg: "bg-emerald-900",
  };
};

const EvolutionaryChaosNetworkUI: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [currentTask, setCurrentTask] = useState<Task>(TASKS.MACKEY_GLASS);
  const [metrics, setMetrics] = useState<SimulationMetrics | null>(null);
  const [config, setConfig] = useState<SimulationConfig>({ ...DEFAULT_CONFIG });
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  // Replaced Toasts with Persistent Log History
  const [systemLogs, setSystemLogs] = useState<
    { step: number; message: string }[]
  >([]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [lstmDebug, setLstmDebug] = useState<any>(null);

  const waveformHistory = useRef<Array<{ target: number; prediction: number }>>(
    [],
  );
  const lossHistory = useRef<number[]>([]);
  const processorRef = useRef<EvolutionaryChaosNetwork | null>(null);
  const animationFrameRef = useRef<number>();

  const initialize = (task: Task, cfg: SimulationConfig = config) => {
    processorRef.current = new EvolutionaryChaosNetwork(task, cfg);
    const initialMetrics = processorRef.current.step();
    setMetrics(initialMetrics);
    setNetworkState(processorRef.current.getNetworkState());
    setLstmDebug(processorRef.current.getLSTMDebugState());
    setStep(0);
    waveformHistory.current = [];
    lossHistory.current = [];
  };

  useEffect(() => {
    initialize(currentTask, config);
  }, []);

  const changeTask = (task: Task) => {
    setCurrentTask(task);
    processorRef.current = new EvolutionaryChaosNetwork(task, config);
    if (processorRef.current) {
      setMetrics(processorRef.current.step());
      setNetworkState(processorRef.current.getNetworkState());
      setLstmDebug(processorRef.current.getLSTMDebugState());
      setStep(0);
      waveformHistory.current = [];
      lossHistory.current = [];
    }
  };

  const handleReset = () => {
    setIsRunning(false);
    processorRef.current = new EvolutionaryChaosNetwork(currentTask, config);
    if (processorRef.current) {
      setMetrics(processorRef.current.step());
      setNetworkState(processorRef.current.getNetworkState());
      setLstmDebug(processorRef.current.getLSTMDebugState());
      waveformHistory.current = [];
      lossHistory.current = [];
    }
    setStep(0);
  };

  const handleConfigChange = (key: keyof SimulationConfig, value: number) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const loop = () => {
    if (!processorRef.current || !isRunning) return;
    let newMetrics;
    const processor = processorRef.current;

    for (let k = 0; k < 2; k++) {
      newMetrics = processor.step();
      setMetrics(newMetrics);
      // Use EXACT step from engine for perfect sync
      const currentSimStep = newMetrics.step;
      setStep(currentSimStep);

      // Handle Logs -> Persistent Console
      if (newMetrics.logs && newMetrics.logs.length > 0) {
        const newEntries = newMetrics.logs.map((msg) => ({
          step: currentSimStep,
          message: msg.replace("[SYSTEM_EVENT] ", ""), // Clean up prefix
        }));

        setSystemLogs((prev) => {
          // Keep last 100 entries for history
          const updated = [...prev, ...newEntries].slice(-100);
          return updated;
        });
      }

      waveformHistory.current.push({
        target: newMetrics.target,
        prediction: newMetrics.prediction,
      });
      if (waveformHistory.current.length > 200) waveformHistory.current.shift();

      // Push (Step, Loss) tuple if chart supports it, or just Loss
      // Chart uses index as step currently? No, chart uses history array.
      // But we render X-axis based on currentStep.
      lossHistory.current.push(newMetrics.avgLoss);
      if (lossHistory.current.length > 200) lossHistory.current.shift();
    }

    if (newMetrics) setMetrics(newMetrics);
    setNetworkState(processor.getNetworkState());
    setLstmDebug(processor.getLSTMDebugState());
    setStep((s) => s + 2); // This line was redundant if setStep(processor.getSteps()) is used
    animationFrameRef.current = requestAnimationFrame(loop);
  };

  useEffect(() => {
    if (isRunning) {
      animationFrameRef.current = requestAnimationFrame(loop);
    } else {
      if (animationFrameRef.current !== undefined)
        cancelAnimationFrame(animationFrameRef.current);
    }
    return () => {
      if (animationFrameRef.current !== undefined)
        cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isRunning]);

  // Toast Cleanup Effect REMOVED (No ephemeral toasts)

  if (!metrics || !networkState)
    return <div className="p-10 text-slate-400">Initializing Cortex...</div>;

  const rank = getRank(metrics.avgLoss);
  const meta = metrics.metaController;

  let excitCount = 0,
    inhibCount = 0;
  for (let i = 0; i < networkState.currentSize; i++) {
    if (networkState.neuronTypes[i] === 1) excitCount++;
    else inhibCount++;
  }

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans">
      <ControlHeader
        tasks={TASKS}
        currentTask={currentTask}
        step={step}
        isRunning={isRunning}
        onToggleRun={() => setIsRunning(!isRunning)}
        onReset={handleReset}
        onChangeTask={changeTask}
      />

      {/* Toast Container REMOVED */}

      {/* Main: 3-Column Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* LEFT COLUMN: Network Viz + Forecast (smaller) */}
        <div className="w-80 flex flex-col border-r border-slate-800 bg-slate-900/50">
          {/* Mini Network Viz */}
          <div className="h-64 relative">
            <NetworkVisualization
              networkState={networkState}
              adaptationStatus={metrics.adaptationStatus}
              step={step}
            />
          </div>

          {/* LSTM Internals Visualization - Moved to Left Panel */}
          <div className="flex-1 mt-4 p-3 border-t border-slate-800 overflow-y-auto min-h-0">
            <LSTMVisualization debugState={lstmDebug} />
          </div>
        </div>

        {/* MIDDLE COLUMN: Charts & Training Info */}
        <div className="flex-1 p-4 flex flex-col gap-4 overflow-y-auto">
          {/* Top Row: Activity + Stats */}
          <div className="grid grid-cols-2 gap-4">
            {/* Loss History moved to Right Panel */}
            <NeuronActivityChart
              activations={Array.from(
                networkState.activations.slice(0, networkState.currentSize),
              )}
              neuronTypes={Array.from(
                networkState.neuronTypes.slice(0, networkState.currentSize),
              )}
              maxDisplay={32}
            />
            {/* E-I Balance */}
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-3">
              <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider mb-2">
                E-I Balance
              </div>
              <div className="flex items-center gap-2 text-sm">
                <div className="flex-1 bg-slate-800 rounded-full h-4 overflow-hidden flex">
                  <div
                    className="bg-red-500 h-full"
                    style={{
                      width: `${(excitCount / (excitCount + inhibCount)) * 100}%`,
                    }}
                  />
                  <div
                    className="bg-blue-500 h-full"
                    style={{
                      width: `${(inhibCount / (excitCount + inhibCount)) * 100}%`,
                    }}
                  />
                </div>
              </div>
              <div className="flex justify-between mt-2 text-xs">
                <span className="text-red-400">E: {excitCount}</span>
                <span className="text-blue-400">I: {inhibCount}</span>
                <span className="text-slate-400">
                  {((excitCount / (excitCount + inhibCount)) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          {/* Training Stats Bar */}
          <TrainingStatsBar
            learningRate={networkState.liveHyperparams.learningRate}
            activeSynapses={metrics.activeConnections}
            targetSynapses={metrics.targetConnections}
            excitCount={excitCount}
            inhibCount={inhibCount}
            spectralRadius={networkState.liveHyperparams.spectral}
            measuredSpectralRadius={metrics.spectralRadius}
            leak={networkState.liveHyperparams.leak}
          />

          {/* Hyperparameter DNA Grid */}
          <div className="bg-slate-900 rounded-lg border border-slate-800 p-3">
            <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider mb-2 flex items-center gap-2">
              <Dna size={12} className="text-blue-400" /> Evolving
              Hyperparameters
            </div>
            <div className="grid grid-cols-4 gap-2">
              {[
                {
                  label: "Leak",
                  value: metrics.dna.leak,
                  color: "text-emerald-400",
                },
                {
                  label: "Spectral",
                  value: metrics.dna.spectral,
                  color: "text-purple-400",
                },
                {
                  label: "Input Scale",
                  value: metrics.dna.inputScale,
                  color: "text-blue-400",
                },
                {
                  label: "Learning Rate",
                  value: metrics.dna.learningRate,
                  color: "text-yellow-400",
                },
                {
                  label: "Smooth Factor",
                  value: metrics.dna.smoothingFactor,
                  color: "text-orange-400",
                },
                {
                  label: "L-V Growth",
                  value: metrics.dna.lvGrowth,
                  color: "text-green-400",
                },
                {
                  label: "L-V Decay",
                  value: metrics.dna.lvDecay,
                  color: "text-red-400",
                  precision: 5,
                },
                {
                  label: "Out Gain",
                  value: metrics.dna.outputGain,
                  color: "text-pink-400",
                },
              ].map(({ label, value, color, precision = 4 }) => (
                <div
                  key={label}
                  className="bg-slate-950 p-2 rounded border border-slate-800 text-center"
                >
                  <div className="text-[9px] text-slate-500 mb-1">{label}</div>
                  <div className={`text-xs font-mono ${color}`}>
                    {value.toFixed(precision)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* L-V Dynamics Info */}
          <div className="bg-slate-900 rounded-lg border border-slate-800 p-3">
            <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider mb-2 flex items-center gap-2">
              <Network size={12} /> L-V Dynamics
            </div>
            <div className="grid grid-cols-4 gap-3 text-sm">
              <div className="text-center">
                <div className="text-2xl font-mono text-white">
                  {metrics.activeConnections}
                </div>
                <div className="text-[10px] text-slate-500">
                  Active Synapses
                </div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-mono text-cyan-400">
                  {metrics.targetConnections}
                </div>
                <div className="text-[10px] text-slate-500">Target</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-mono text-emerald-400">
                  {metrics.regrown}
                </div>
                <div className="text-[10px] text-slate-500">Regrown</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-mono text-slate-300">
                  {metrics.neuronCount}
                </div>
                <div className="text-[10px] text-slate-500">Neurons</div>
              </div>
            </div>
          </div>

          {/* System Console (Persistent Log) */}
          <div className="bg-slate-950 rounded-lg border border-slate-800 p-3 flex-1 flex flex-col min-h-[200px]">
            <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider mb-2 flex items-center gap-2">
              <Terminal size={12} className="text-slate-400" /> System Console
            </div>
            <div className="flex-1 overflow-y-auto font-mono text-[10px] space-y-1 p-2 bg-slate-900 rounded border border-slate-800/50">
              {systemLogs.length === 0 && (
                <div className="text-slate-600 italic">
                  System ready. Waiting for events...
                </div>
              )}

              {systemLogs.map((log, idx) => (
                <div
                  key={idx}
                  className="flex gap-2 border-l-2 pl-2 border-slate-700 hover:bg-slate-800/50 transition-colors"
                >
                  <span className="text-slate-500 w-12 shrink-0">
                    #{log.step}
                  </span>
                  <span
                    className={`
                        ${log.message.includes("MITOSIS") ? "text-purple-400 font-bold" : ""}
                        ${log.message.includes("REGROWTH") ? "text-emerald-400" : ""}
                        ${log.message.includes("PRUNE") ? "text-red-400" : ""}
                        ${log.message.includes("SINKHORN") ? "text-blue-400" : ""}
                     `}
                  >
                    {log.message}
                  </span>
                </div>
              ))}
              {/* Auto-scroll anchor */}
              <div ref={(el) => el?.scrollIntoView({ behavior: "smooth" })} />
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Status & Meta-Controller */}
        <div className="w-72 flex flex-col gap-3 p-3 border-l border-slate-800 bg-slate-900">
          {/* Rank Badge */}
          <div
            className={`${rank.bg} rounded-lg p-3 flex items-center justify-between`}
          >
            <div className="flex items-center gap-2">
              <Trophy className={rank.color} size={18} />
              <span className={`font-bold text-sm ${rank.color}`}>
                {rank.label}
              </span>
            </div>
            <span className="font-mono text-sm text-slate-300">
              {metrics.avgLoss.toFixed(4)}
            </span>
          </div>

          {/* Brain Size */}
          <div className="bg-slate-950 rounded-lg border border-slate-800 p-3">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                Brain Size
              </span>
              <span className="font-mono text-sm">
                {metrics.neuronCount} / {config.maxNeurons}
              </span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-2">
              <div
                className="h-full bg-cyan-500 rounded-full"
                style={{
                  width: `${(metrics.neuronCount / config.maxNeurons) * 100}%`,
                }}
              />
            </div>
          </div>

          {/* Growth Pressure */}
          <div className="bg-slate-950 rounded-lg border border-slate-800 p-3">
            <div className="flex justify-between items-center mb-2">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                Growth Pressure
              </span>
              <span
                className={`font-mono text-xs ${metrics.patience > config.patienceLimit * 0.8 ? "text-orange-400" : "text-slate-400"}`}
              >
                {Math.floor(metrics.patience)} / {config.patienceLimit}
              </span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-2">
              <div
                className={`h-full rounded-full ${metrics.adaptationStatus === "LOCKED" ? "bg-purple-500" : "bg-yellow-500"}`}
                style={{
                  width: `${Math.min((metrics.patience / config.patienceLimit) * 100, 100)}%`,
                }}
              />
            </div>
            <div className="text-[10px] text-slate-500 mt-1 flex items-center gap-1">
              <TrendingUp size={10} />
              {metrics.adaptationStatus}
            </div>
          </div>

          {/* Bicameral Meta-Controller */}
          <div className="bg-slate-950 rounded-lg border border-slate-800 p-3">
            <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-2 flex items-center gap-2">
              <BrainCircuit size={12} className="text-purple-400" /> Bicameral
              Controller
            </div>

            {/* Short-Term */}
            <div className="mb-2">
              <div className="flex items-center gap-1 mb-1">
                <Zap size={10} className="text-yellow-400" />
                <span className="text-[9px] text-slate-500">Short-Term</span>
                <span className="text-[9px] font-mono text-yellow-400 ml-auto">
                  {meta.shortTermActivity.toFixed(2)}
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5">
                <div
                  className="h-full bg-yellow-400 rounded-full"
                  style={{
                    width: `${Math.min(meta.shortTermActivity * 100, 100)}%`,
                  }}
                />
              </div>
            </div>

            {/* Long-Term */}
            <div className="mb-2">
              <div className="flex items-center gap-1 mb-1">
                <Timer size={10} className="text-cyan-400" />
                <span className="text-[9px] text-slate-500">Long-Term</span>
                <span className="text-[9px] font-mono text-cyan-400 ml-auto">
                  {meta.longTermActivity.toFixed(2)}
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5">
                <div
                  className="h-full bg-cyan-400 rounded-full"
                  style={{
                    width: `${Math.min(meta.longTermActivity * 100, 100)}%`,
                  }}
                />
              </div>
            </div>

            {/* Gate */}
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-[9px] text-slate-500">
                  Gate (ST ↔ LT)
                </span>
                <span className="text-[9px] font-mono text-slate-400">
                  {meta.gate.toFixed(2)}
                </span>
              </div>
              <div className="relative w-full bg-slate-800 rounded-full h-2">
                <div className="absolute inset-0 bg-gradient-to-r from-yellow-500/30 to-cyan-500/30 rounded-full" />
                <div
                  className="absolute top-0 h-full w-1 bg-white rounded-full shadow-lg"
                  style={{ left: `calc(${meta.gate * 100}% - 2px)` }}
                />
              </div>
            </div>
          </div>

          {/* Current Error */}
          <div className="bg-slate-950 rounded-lg border border-slate-800 p-3">
            <div className="flex justify-between items-center text-xs mb-1">
              <span className="text-slate-500">Current Error (MSE)</span>
              <span
                className={`font-mono ${metrics.loss < 0.01 ? "text-emerald-400" : "text-orange-400"}`}
              >
                {metrics.loss.toFixed(5)}
              </span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-1.5">
              <div
                className={`h-full rounded-full ${metrics.loss < 0.01 ? "bg-emerald-500" : "bg-orange-500"}`}
                style={{ width: `${Math.min(metrics.loss * 1000, 100)}%` }}
              />
            </div>
          </div>

          {/* Forecast Chart - Moved to Right Panel */}
          <div className="bg-slate-950 rounded-lg border border-slate-800 p-3">
            <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider mb-2">
              Live Forecast
            </div>
            <ForecastChart history={waveformHistory.current} step={step} />
          </div>

          {/* Loss History - Moved to Right Panel */}
          <div className="bg-slate-950 rounded-lg border border-slate-800 p-3">
            <LossHistoryChart
              history={lossHistory.current}
              height={80}
              currentStep={step}
              events={systemLogs}
            />
          </div>
        </div>
      </div>

      <ConfigPanel
        config={config}
        onConfigChange={handleConfigChange}
        isOpen={isConfigOpen}
        onToggle={() => setIsConfigOpen(!isConfigOpen)}
      />
    </div>
  );
};

export default EvolutionaryChaosNetworkUI;
