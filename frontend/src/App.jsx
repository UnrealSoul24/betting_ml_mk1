import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Terminal,
  Database,
  Play,
  AlertCircle,
  ChevronsUp,
  Trophy,
  Activity,
  Wifi,
} from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

import ParlayCard from "./components/ParlayCard";
import BetCard from "./components/BetCard";
import PlayerDetailModal from "./components/PlayerDetailModal";

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [logs, setLogs] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [selectedDate, setSelectedDate] = useState(
    new Date().toISOString().split("T")[0],
  );
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [ws, setWs] = useState(null);
  const scrollRef = useRef(null);

  // WebSocket Connection for Logs
  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8000/logs");

    socket.onopen = () => {
      setIsConnected(true);
      addLog("System Connected. Ready for Analysis.", "🔌", "success");
    };

    socket.onmessage = (event) => {
      const msg = event.data;
      const milestoneMap = {
        "Injuries checked": { icon: "✅", status: "success" },
        "Retraining queued": { icon: "⏳", status: "loading" },
        "Predicting minutes": { icon: "⏳", status: "loading" },
        "Calculating final predictions": { icon: "⏳", status: "loading" },
        "Analysis complete": { icon: "✅", status: "success" },
        "Training": { icon: "⏳", status: "loading" },
      };

      for (const [key, val] of Object.entries(milestoneMap)) {
        if (msg.toLowerCase().includes(key.toLowerCase())) {
          addLog(msg, val.icon, val.status);
          return;
        }
      }
    };

    socket.onclose = () => {
      setIsConnected(false);
      addLog("System Disconnected.", "🔌", "error");
    };

    setWs(socket);

    return () => {
      socket.close();
    };
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const addLog = (msg, icon = "➜", status = "info") => {
    const time = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, { time, msg, icon, status }]);
  };

  const handleRunAnalysis = async () => {
    if (isLoading) return;
    setIsLoading(true);
    setPredictions([]);
    addLog("Initiating Daily Analysis Sequence...", "⏳", "loading");

    try {
      const response = await fetch("http://localhost:8000/analyze-today", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          date: selectedDate,
          force_train: false,
        }),
      });

      const data = await response.json();
      if (data.results || data.predictions) {
        setPredictions(data); // Always use full object for new UI
        const count = data.predictions ? data.predictions.length : (data.results ? data.results.length : 0);
        addLog(`Analysis Complete. ${count} predictions received.`, "✅", "success");
      }
    } catch (e) {
      addLog(`CRITICAL ERROR: ${e.message}`, "⚠️", "error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 text-gray-200">
      {/* Header */}
      <header className="max-w-7xl mx-auto flex justify-between items-center mb-12">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-neon-green/80 animate-pulse" />
          <h1 className="text-3xl font-bold tracking-tighter text-white">
            BETTING<span className="text-neon-green">.AI</span>{" "}
            <span className="text-xs opacity-50 font-mono border border-white/20 px-2 py-0.5 rounded">
              MK1
            </span>
          </h1>
        </div>
        <div className="flex items-center gap-4 text-sm font-mono text-gray-400">
          <div className="flex items-center gap-2">
            <Wifi
              size={14}
              className={isConnected ? "text-neon-green" : "text-red-500"}
            />
            {isConnected ? "ONLINE" : "OFFLINE"}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Controls & Console */}
        <div className="space-y-6 lg:col-span-1">
          {/* Control Panel */}
          <div className="glass p-6 rounded-2xl relative overflow-hidden group">
            <div className="absolute inset-0 bg-neon-green/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />

            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity className="text-neon-blue" /> Control Center
            </h2>

            <div className="mb-4">
              <label className="block text-xs font-mono text-gray-500 mb-2">
                TARGET DATE
              </label>
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2 text-white font-mono focus:border-neon-green/50 outline-none"
              />
            </div>

            <button
              onClick={handleRunAnalysis}
              disabled={isLoading || !isConnected}
              className="w-full h-14 bg-gradient-to-r from-neon-blue to-neon-purple rounded-xl font-bold text-white shadow-lg shadow-neon-blue/20 hover:shadow-neon-blue/40 transform hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ANALYZING...
                </>
              ) : (
                <>
                  <Play fill="currentColor" /> FIND TODAY'S BETS
                </>
              )}
            </button>

            <p className="mt-4 text-xs text-gray-500 text-center">
              Scans live schedule, trains models, and calculates variance.
            </p>
          </div>

          {/* Console Log */}
          <div className="glass p-4 rounded-2xl h-[400px] flex flex-col font-mono text-xs relative overflow-hidden">
            <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/5">
              <span className="flex items-center gap-2 text-gray-400">
                <Terminal size={14} /> SYSTEM LOG
              </span>
              <span className="w-2 h-2 rounded-full bg-neon-green animate-blink" />
            </div>

            <div
              ref={scrollRef}
              className="flex-1 overflow-y-auto space-y-1 scrollbar-hide text-green-400/80"
            >
              {logs.length === 0 && (
                <span className="opacity-30">Waiting for command...</span>
              )}
              {logs.map((log, i) => {
                let textColor = "text-green-400/80";
                if (log.status === "loading") textColor = "text-blue-400";
                if (log.status === "success") textColor = "text-neon-green";
                if (log.status === "error") textColor = "text-red-400";

                return (
                  <div key={i} className={clsx("break-words flex items-start text-xs font-mono mb-1", textColor)}>
                    <span className="opacity-30 mr-2 min-w-[60px]">[{log.time}]</span>
                    <div className="flex-1 flex items-start">
                      <span className={clsx("mr-2", log.status === "loading" && "animate-pulse")}>{log.icon}</span>
                      <span>{log.msg}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="lg:col-span-2 space-y-6">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Trophy className="text-amber-400" /> DAILY DASHBOARD
          </h2>

          {!isLoading && predictions.trixie && (
            <ParlayCard
              trixie={predictions.trixie}
              onLegClick={(leg) => setSelectedPlayer(leg)}
            />
          )}

          {(!predictions.predictions || predictions.predictions.length === 0) ? (
            <div className="h-[400px] border-2 border-dashed border-white/10 rounded-3xl flex items-center justify-center text-gray-600">
              NO DATA AVAILABLE
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {predictions.predictions.map((pred, i) => (
                <BetCard
                  key={i}
                  pred={pred}
                  index={i}
                  onClick={() => setSelectedPlayer(pred)}
                />
              ))}
            </div>
          )}
        </div>
      </main>

      {selectedPlayer && (
        <PlayerDetailModal
          isOpen={!!selectedPlayer}
          onClose={() => setSelectedPlayer(null)}
          player={selectedPlayer}
        />
      )}
    </div>
  );
}
