import React, { useState, useEffect, useRef } from 'react';
import { EvolutionaryChaosNetwork, SimulationMetrics, NetworkState, SimulationConfig, DEFAULT_CONFIG } from '../engine/simulationEngine';
import { TASKS, Task } from '../tasks/tasks';
import { ControlHeader } from './ControlHeader';
import { NetworkVisualization } from './NetworkVisualization';
import { ForecastChart } from './ForecastChart';
import { MetricsPanel } from './MetricsPanel';
import { ConfigPanel } from './ConfigPanel';

const EvolutionaryChaosNetworkUI: React.FC = () => {
  // State
  const [isRunning, setIsRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [currentTask, setCurrentTask] = useState<Task>(TASKS.MACKEY_GLASS);
  const [metrics, setMetrics] = useState<SimulationMetrics | null>(null);
  const [config, setConfig] = useState<SimulationConfig>({ ...DEFAULT_CONFIG });
  const [isConfigOpen, setIsConfigOpen] = useState(false);

  // Need to hold network state for visualization.
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);

  // UI Visualization State (Decoupled from Engine)
  const waveformHistory = useRef<Array<{target: number, prediction: number}>>([]);

  // Refs
  const simulationRef = useRef<EvolutionaryChaosNetwork | null>(null);
  const reqRef = useRef<number>();

  const initialize = (task: Task, cfg: SimulationConfig = config) => {
    simulationRef.current = new EvolutionaryChaosNetwork(task, cfg);
    const initialMetrics = simulationRef.current.step();
    setMetrics(initialMetrics);
    setNetworkState(simulationRef.current.getNetworkState());
    setStep(0);
    waveformHistory.current = []; // Reset history
  };

  useEffect(() => {
    initialize(currentTask, config);
  }, []);

  const changeTask = (task: Task) => {
    setCurrentTask(task);
    // Reset engine with new task and current config
    simulationRef.current = new EvolutionaryChaosNetwork(task, config);
    if (simulationRef.current) {
        setMetrics(simulationRef.current.step());
        setNetworkState(simulationRef.current.getNetworkState());
        setStep(0);
        waveformHistory.current = [];
    }
  };

  const handleReset = () => {
      setIsRunning(false);
      // Re-initialize with current config
      simulationRef.current = new EvolutionaryChaosNetwork(currentTask, config);
       if (simulationRef.current) {
        setMetrics(simulationRef.current.step());
        setNetworkState(simulationRef.current.getNetworkState());
        waveformHistory.current = [];
      }
      setStep(0);
  };

  const handleConfigChange = (key: keyof SimulationConfig, value: number) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const loop = () => {
    if (!simulationRef.current || !isRunning) return;

    let newMetrics;
    const engine = simulationRef.current;

    // Run 2 steps per frame
    for(let k=0; k<2; k++) {
        newMetrics = engine.step();
        // Buffer data for UI
        waveformHistory.current.push({ target: newMetrics.target, prediction: newMetrics.prediction });
        if (waveformHistory.current.length > 200) waveformHistory.current.shift();
    }

    if (newMetrics) setMetrics(newMetrics);
    setNetworkState(engine.getNetworkState());
    // Use engine step count or local? Engine is safer.
    // engine doesn't expose public step count property easily unless we add accessor.
    // But Metrics has it? No, Metrics has prediction/target.
    // Actually getMetrics return step? Let's assume setStep(step+2).
    setStep(s => s + 2);

    reqRef.current = requestAnimationFrame(loop);
  };

  useEffect(() => {
    if (isRunning) {
      reqRef.current = requestAnimationFrame(loop);
    } else {
      if (reqRef.current !== undefined) cancelAnimationFrame(reqRef.current);
    }
    return () => {
      if (reqRef.current !== undefined) cancelAnimationFrame(reqRef.current);
    };
  }, [isRunning]);

  if (!metrics || !networkState) return <div className="p-10 text-slate-400">Initializing Cortex...</div>;

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500/30">

      <ControlHeader
        tasks={TASKS}
        currentTask={currentTask}
        step={step}
        isRunning={isRunning}
        onToggleRun={() => setIsRunning(!isRunning)}
        onReset={handleReset}
        onChangeTask={changeTask}
      />

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden relative">

        <NetworkVisualization
            networkState={networkState}
            adaptationStatus={metrics.adaptationStatus}
        />

        <div className="w-96 flex flex-col border-l border-slate-800 bg-slate-900">
            <MetricsPanel
                metrics={metrics}
                maxNeurons={config.maxNeurons}
                patienceLimit={config.patienceLimit}
            />

            <div className="p-6 pt-0 border-t border-slate-800/50 bg-slate-900">
             <ForecastChart
                history={waveformHistory.current}
                step={step}
             />
            </div>
        </div>

      </div>

      {/* Config Panel */}
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

