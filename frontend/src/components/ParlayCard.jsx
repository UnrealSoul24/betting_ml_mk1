import { motion } from "framer-motion";
import { Zap, ShieldCheck } from "lucide-react";
import { clsx } from "clsx";

export default function ParlayCard({ trixie, onLegClick }) {
    if (!trixie) return null;

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
                                <p className="text-xs text-gray-400">High Confidence 3-Leg Parlay Strategy</p>
                            </div>
                        </div>
                        <div className="text-right">
                            <div className="text-xs text-gray-500 uppercase font-mono">Total Odds</div>
                            <div className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-neon-purple to-white">
                                {trixie.total_odds.toFixed(2)}x
                            </div>
                        </div>
                    </div>

                    {/* Legs */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {trixie.sgp_legs.map((leg, i) => (
                            <div
                                key={i}
                                onClick={() => onLegClick && onLegClick(leg)}
                                className="bg-white/5 border border-white/5 rounded-xl p-4 flex flex-col h-full bg-gradient-to-b from-white/5 to-transparent cursor-pointer hover:bg-white/10 transition-colors group"
                            >
                                {/* Player Header */}
                                <div className="border-b border-white/10 pb-3 mb-3">
                                    <div className="flex justify-between items-center mb-1">
                                        <span className="text-[10px] font-bold text-neon-purple px-2 py-0.5 rounded bg-neon-purple/10">SGP Leg #{i + 1}</span>
                                        <span className="text-[10px] font-mono text-gray-500">{leg.matchup}</span>
                                    </div>
                                    <h3 className="font-bold text-white text-lg truncate group-hover:text-neon-purple transition-colors">{leg.player_name}</h3>
                                </div>

                                {/* 3 Bets */}
                                <div className="space-y-2 flex-grow">
                                    {leg.bets.map((bet, j) => (
                                        <div key={j} className={clsx(
                                            "flex justify-between items-center p-2 rounded text-sm border",
                                            bet.type === 'MAIN' ? "bg-neon-green/10 border-neon-green/30" : "bg-black/20 border-white/5"
                                        )}>
                                            <span className={clsx(
                                                "font-bold",
                                                bet.type === 'MAIN' ? "text-neon-green" : "text-gray-300"
                                            )}>{bet.desc}</span>

                                            {bet.type === 'MAIN' ? (
                                                <ShieldCheck size={14} className="text-neon-green" />
                                            ) : (
                                                <span className="text-[10px] text-gray-500 uppercase">Safe Anchor</span>
                                            )}
                                        </div>
                                    ))}
                                </div>

                                <div className="mt-3 pt-2 border-t border-white/5 text-center">
                                    <span className="text-[10px] text-gray-400">Total SGP Odds</span>
                                    <div className="text-white font-mono font-bold">{leg.sgp_odds?.toFixed(2) || "-"}x</div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="mt-4 text-center text-xs text-gray-500">
                        Recommended: <span className="text-white font-bold">{trixie.rec_units} Units</span> on the full Parlay.
                    </div>
                </div>
            </div>
        </div>
    );
}
