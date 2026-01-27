import { motion } from "framer-motion";
import { Zap, ShieldCheck, Trophy, CheckCircle, Crosshair } from "lucide-react";
import { clsx } from "clsx";

export default function ParlayCard({ trixie, onLegClick }) {
    if (!trixie || !trixie.active) return null;

    // Convert Prob to Percentage
    const probPct = (trixie.combined_prob * 100).toFixed(1) + "%";

    // Implied Odds (1/p) - purely theoretical
    const impliedOdds = trixie.combined_prob > 0 ? (1 / trixie.combined_prob).toFixed(2) : "0.00";

    return (
        <div className="w-full mb-8">
            <div className="relative p-1 rounded-2xl bg-gradient-to-r from-neon-purple via-pink-500 to-neon-blue p-[1px]">
                <div className="absolute inset-0 blur opacity-30 bg-gradient-to-r from-neon-purple to-neon-blue" />

                <div className="relative bg-[#0F0F16] rounded-2xl p-6 overflow-hidden">
                    {/* Header */}
                    <div className="flex justify-between items-center mb-6">
                        <div className="flex items-center gap-3">
                            <div className="bg-neon-purple/20 p-2 rounded-lg">
                                <Zap className="text-neon-purple" size={24} />
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-white tracking-tight">THE DAILY TRIXIE</h2>
                                <p className="text-xs text-gray-400">3-Player SGP Strategy (Main Line + Safe Anchors)</p>
                            </div>
                        </div>
/*
                        <div className="text-right">
                            <div className="text-xs text-gray-500 uppercase font-mono">Combined Prob</div>
                            <div className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-neon-purple to-white">
                                {probPct}
                            </div>
                        </div>
                        */
                    </div>

                    {/* Legs (Groups) */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {trixie.legs.map((leg, i) => (
                            <div
                                key={i}
                                onClick={() => onLegClick && onLegClick({ ...leg, player_name: leg.player })}
                                className="bg-white/5 border border-white/5 rounded-xl p-4 flex flex-col h-full bg-gradient-to-b from-white/5 to-transparent cursor-pointer hover:bg-white/10 transition-colors group relative overflow-hidden"
                            >
                                <div className="absolute top-0 right-0 p-2 opacity-50 text-6xl font-black text-white/5 select-none pointer-events-none">
                                    {String.fromCharCode(65 + i)}
                                </div>

                                {/* Header */}
                                <div className="mb-4 pb-3 border-b border-white/10">
                                    <div className="flex justify-between items-center mb-1">
                                        <span className="text-[10px] font-bold text-neon-purple px-2 py-0.5 rounded bg-neon-purple/10">Group {String.fromCharCode(65 + i)}</span>
                                        <span className={clsx(
                                            "text-[10px]",
                                            leg.badge.includes("DIAMOND") ? "text-cyan-400" :
                                                leg.badge.includes("GOLD") ? "text-yellow-400" : "text-gray-500"
                                        )}>{leg.badge}</span>
                                    </div>
                                    <h3 className="font-bold text-white text-lg truncate group-hover:text-neon-purple transition-colors">{leg.player}</h3>
                                </div>

                                {/* Bets List */}
                                <div className="space-y-2 flex-grow">
                                    {leg.bets.map((bet, j) => (
                                        <div key={j} className={clsx(
                                            "flex justify-between items-center p-2 rounded text-sm border",
                                            bet.type === 'MAIN' ? "bg-neon-green/10 border-neon-green/30" : "bg-black/20 border-white/5 opacity-80"
                                        )}>
                                            <div className="flex items-center gap-2">
                                                {bet.type === 'MAIN' ? (
                                                    <Crosshair size={14} className="text-neon-green" />
                                                ) : (
                                                    <CheckCircle size={14} className="text-gray-500" />
                                                )}
                                                <span className={clsx(
                                                    "font-bold font-mono",
                                                    bet.type === 'MAIN' ? "text-neon-green" : "text-gray-300"
                                                )}>{bet.desc}</span>
                                            </div>

                                            {bet.type === 'MAIN' ? (
                                                <span className="text-[10px] text-neon-green uppercase font-bold tracking-wider">Main</span>
                                            ) : (
                                                <span className="text-[10px] text-gray-500 uppercase">Safe</span>
                                            )}
                                        </div>
                                    ))}

                                    {leg.bets.length === 0 && (
                                        <div className="text-xs text-gray-500 italic p-2">No valid props found.</div>
                                    )}
                                </div>

                                <div className="w-full mt-4 pt-3 border-t border-white/5 flex justify-between items-center text-xs">
                                    <span className="text-gray-500">Unit Size</span>
                                    <span className="text-white font-mono font-bold">{leg.units}u</span>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="mt-4 text-center text-xs text-gray-500">
                        Top 3 Daily Plays optimized for SGP connectivity.
                    </div>
                </div>
            </div>
        </div>
    );
}
