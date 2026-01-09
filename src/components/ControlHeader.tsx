import React, { useState } from 'react';
import { Task } from '../tasks/tasks';
import { Activity, Play, Pause, ChevronDown, RefreshCw } from 'lucide-react';

interface ControlHeaderProps {
  tasks: Record<string, Task>;
  currentTask: Task;
  step: number;
  isRunning: boolean;
  onToggleRun: () => void;
  onReset: () => void;
  onChangeTask: (task: Task) => void;
}

export const ControlHeader: React.FC<ControlHeaderProps> = ({
  tasks,
  currentTask,
  step,
  isRunning,
  onToggleRun,
  onReset,
  onChangeTask,
}) => {
  const [isTaskMenuOpen, setIsTaskMenuOpen] = useState(false);

  const handleTaskChange = (task: Task) => {
    onChangeTask(task);
    setIsTaskMenuOpen(false);
  };

  return (
    <div className="flex items-center justify-between p-4 bg-slate-900 border-b border-slate-800 shadow-md z-10">
      <div className="flex items-center gap-3">
        <Activity className="text-cyan-400" />
        <div>
          <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500">
            Evolutionary Chaos Network
          </h1>
          <div className="text-[10px] text-slate-500 font-mono tracking-wide uppercase">
            Liquid State Machine • Automated Adaptation
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4 text-sm">
        {/* Task Dropdown */}
        <div className="relative">
          <button
            onClick={() => setIsTaskMenuOpen(!isTaskMenuOpen)}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-md border border-slate-700 transition-colors text-xs font-medium min-w-[160px] justify-between"
          >
            <span>{currentTask.name}</span>
            <ChevronDown size={14} className={`transition-transform ${isTaskMenuOpen ? 'rotate-180' : ''}`} />
          </button>

          {isTaskMenuOpen && (
            <div className="absolute top-full mt-2 right-0 w-80 bg-slate-900 border border-slate-700 rounded-lg shadow-xl overflow-hidden z-20">
              {Object.values(tasks).map((task) => (
                <button
                  key={task.id}
                  onClick={() => handleTaskChange(task)}
                  className="w-full text-left px-4 py-3 hover:bg-slate-800 transition-colors border-b border-slate-800 last:border-0 flex items-start gap-3 group"
                >
                  <div
                    className={`mt-0.5 w-4 h-4 rounded-full border flex items-center justify-center ${
                      currentTask.id === task.id ? 'border-cyan-500 bg-cyan-500/20' : 'border-slate-600'
                    }`}
                  >
                    {currentTask.id === task.id && <div className="w-2 h-2 rounded-full bg-cyan-500" />}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-slate-200 group-hover:text-cyan-400 transition-colors">
                        {task.name}
                        </span>
                        {task.type === 'CLASSIFY' && (
                        <span className="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 rounded">NEW</span>
                        )}
                    </div>
                    <div className="text-xs text-slate-500 leading-tight mt-1">{task.description}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="h-6 w-px bg-slate-700 mx-1"></div>

        <div className="flex flex-col items-end">
          <span className="text-slate-500 text-[10px] uppercase tracking-wider">Time Step</span>
          <span className="font-mono text-sm">{step}</span>
        </div>

        <button
          onClick={onToggleRun}
          className={`flex items-center gap-2 px-4 py-2 rounded-md font-semibold transition-all ${
            isRunning
              ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20'
              : 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 border border-emerald-500/20'
          }`}
        >
          {isRunning ? (
            <>
              <Pause size={16} /> Pause
            </>
          ) : (
            <>
              <Play size={16} /> Run
            </>
          )}
        </button>

        <button
          title="Reset Network Weights"
          onClick={onReset}
          className="p-2 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors border border-slate-700"
        >
          <RefreshCw size={16} />
        </button>
      </div>
    </div>
  );
};
