import React, { useRef, useEffect } from "react";
import { Lock, Activity, Hourglass, Plus, Zap } from "lucide-react";
import { NetworkState } from "../engine/simulationEngine";

interface NetworkVisualizationProps {
  networkState: NetworkState;
  adaptationStatus: string;
  step: number;
}

const getStatusColor = (s: string) => {
  switch (s) {
    case "LOCKED":
      return "text-purple-400";
    case "LEARNING":
      return "text-emerald-400";
    case "STAGNANT":
      return "text-orange-400";
    case "GROWING":
      return "text-yellow-400";
    case "GROWING_SYNAPSES":
      return "text-blue-400";
    case "OPTIMIZING":
      return "text-pink-400";
    default:
      return "text-slate-400";
  }
};

export const NetworkVisualization: React.FC<NetworkVisualizationProps> = ({
  networkState,
  adaptationStatus,
  step,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const net = networkState;
    const N = net.currentSize;
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const learningRate = net.liveHyperparams.learningRate;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(15, 23, 42, 0.4)";
    ctx.fillRect(0, 0, w, h);

    // 3D CONFIG - Plasticity Pulse
    // Modulate rotation/pulse by Learning Rate to show "Alive" state
    const plasticity = Math.min(1.0, learningRate * 20.0); // 0.05 -> 1.0
    const pulseSpeed = 0.0005 + 0.002 * plasticity;

    // Global "Breathing" zoom
    const breath = 1.0 + Math.sin(Date.now() * 0.002) * (0.02 * plasticity);

    const time = Date.now() * pulseSpeed;
    const rx = time * 0.5; // Rotate Y
    const ry = time * 0.3; // Rotate X (tumble)

    // Perspective
    const focalLength = 800;
    const sphereRadius = Math.min(w, h) * 0.35 * breath;

    // Project 3D point to 2D
    const project = (x: number, y: number, z: number) => {
      // Rotate Y
      let x1 = x * Math.cos(rx) - z * Math.sin(rx);
      let z1 = z * Math.cos(rx) + x * Math.sin(rx);

      // Rotate X
      let y2 = y * Math.cos(ry) - z1 * Math.sin(ry);
      let z2 = z1 * Math.cos(ry) + y * Math.sin(ry);

      // Perspective Scale
      const scale = focalLength / (focalLength + z2);

      return {
        x: cx + x1 * scale,
        y: cy + y2 * scale,
        scale: scale,
        z: z2,
      };
    };

    // Calc positions (Fibonacci Sphere)
    const positions: {
      x: number;
      y: number;
      z: number;
      scale: number;
      originalIdx: number;
    }[] = [];
    const phi = Math.PI * (3 - Math.sqrt(5)); // Golden Angle

    for (let i = 0; i < N; i++) {
      const y = 1 - (i / (N - 1)) * 2;
      const radius = Math.sqrt(1 - y * y);
      const theta = phi * i;

      const x = Math.cos(theta) * radius;
      const z = Math.sin(theta) * radius;

      const p3d = {
        x: x * sphereRadius,
        y: y * sphereRadius,
        z: z * sphereRadius,
      };

      const proj = project(p3d.x, p3d.y, p3d.z);
      positions.push({ ...proj, originalIdx: i });
    }

    ctx.lineWidth = 1;

    // EDGES: Show Signal Flow
    // Optimization: Only show stronger signals to reduce clutter
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const weight = net.weights[i * 512 + j];
        if (weight === 0) continue;

        const signal =
          net.activations[j] * // Use current activation for instant feedback
          Math.abs(weight);

        const absSignal = Math.abs(signal);

        // Higher threshold for clarity (Clutter Reduction)
        if (absSignal > 0.1) {
          ctx.beginPath();
          const opacity = Math.min(absSignal, 0.6);
          ctx.strokeStyle =
            weight > 0
              ? `rgba(56, 189, 248, ${opacity})` // Cyan
              : `rgba(248, 113, 113, ${opacity})`; // Red

          // Depth cues
          const depthScale = (positions[i].scale + positions[j].scale) / 2;
          ctx.lineWidth = (0.5 + absSignal) * depthScale;

          ctx.beginPath();
          ctx.moveTo(positions[j].x, positions[j].y);
          ctx.lineTo(positions[i].x, positions[i].y);
          ctx.stroke();
        }
      }
    }

    // Sort nodes to draw background first
    positions.sort((a, b) => b.z - a.z);

    // NODES
    let kernelKnots = 0;

    for (const pos of positions) {
      const i = pos.originalIdx;
      const val = net.activations[i];
      const intensity = Math.min(Math.abs(val), 1);

      // Kernel Knot Detection (High Readout Weight)
      const readoutW = net.readout[i];
      const isKnot = Math.abs(readoutW) > 0.01;
      if (isKnot) kernelKnots++;

      // 1. Draw Connection Halo
      if (isKnot) {
        ctx.beginPath();
        // Pulse gold halo based on plasticity
        const haloSize = (10 + Math.abs(readoutW) * 20) * pos.scale;
        ctx.arc(pos.x, pos.y, haloSize, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(234, 179, 8, ${0.1 + plasticity * 0.2})`; // Yellow-500
        ctx.fill();

        // Wireframe ring
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, haloSize, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(234, 179, 8, ${0.3})`;
        ctx.stroke();
      }

      // 2. Draw Neuron Body
      ctx.beginPath();
      // Base size
      let radius = (2 + intensity * 4) * pos.scale;
      if (isKnot) radius *= 1.5; // Knots are larger

      ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

      // Color Logic
      // Knots = Gold/Yellow
      // Excitatory = Cyan
      // Inhibitory = Red
      let fillStyle = "";
      let alpha = (0.4 + intensity) * Math.max(0.2, pos.scale * 0.8);

      if (isKnot) {
        // Kernel Knot color
        fillStyle = `rgba(250, 204, 21, ${0.9})`; // Yellow-400 (Solid)
      } else {
        // Standard Neuron
        fillStyle =
          val > 0
            ? `rgba(56, 189, 248, ${alpha})`
            : `rgba(248, 113, 113, ${alpha})`;
      }

      ctx.fillStyle = fillStyle;
      ctx.fill();
    }

    // HUD OVERLAY
    ctx.font = "10px monospace";
    ctx.fillStyle = "rgba(148, 163, 184, 0.8)"; // Slate-400

    // Left Bottom: Active Kernels
    ctx.fillText(`ACTIVE KNOTS: ${kernelKnots}`, 10, h - 10);

    // Right Bottom: Plasticity
    const pStr = (learningRate * 100).toFixed(2);
    ctx.textAlign = "right";
    ctx.fillStyle =
      learningRate > 0.01 ? "rgba(234, 179, 8, 1)" : "rgba(148, 163, 184, 0.8)";
    ctx.fillText(`PLASTICITY: ${pStr}%`, w - 10, h - 10);
    ctx.textAlign = "left";
  };

  useEffect(() => {
    // Redraw on every step update
    // Using requestAnimationFrame to ensure smooth sync with React updates
    const frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [networkState, step]); // Depend on STEP to force redraws!

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
        {/* Knot Indicator */}
        <div className="bg-slate-900/80 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-700 text-xs flex items-center gap-2 shadow-lg">
          <span className="w-2 h-2 rounded-full bg-yellow-400 shadow-[0_0_8px_rgba(250,204,21,0.5)]"></span>
          Kernel Knot
        </div>

        {/* Dynamic Status Indicator */}
        <div
          className={`bg-slate-900/80 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-700/50 text-xs flex items-center gap-2 shadow-lg ${getStatusColor(
            adaptationStatus,
          )}`}
        >
          {adaptationStatus === "LOCKED" && <Lock size={12} />}
          {adaptationStatus === "LEARNING" && <Activity size={12} />}
          {adaptationStatus === "STAGNANT" && <Hourglass size={12} />}
          {adaptationStatus === "GROWING" && <Plus size={12} />}
          {adaptationStatus === "OPTIMIZING" && <Zap size={12} />}
          <span className="font-mono uppercase tracking-wide">
            {adaptationStatus.replace("_", " ")}
          </span>
        </div>
      </div>
    </div>
  );
};
