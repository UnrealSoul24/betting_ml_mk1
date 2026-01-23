import { motion, AnimatePresence } from "framer-motion";
import { X, TrendingUp, TrendingDown, Target, Award, Activity, Zap, BarChart3, Percent, Brain } from "lucide-react";
import { clsx } from "clsx";

export default function PlayerDetailModal({ isOpen, onClose, player }) {
    if (!isOpen || !player) return null;

    // V2 Model Data
    const isV2 = player.MODEL_VERSION === 'V2';
    const uncertaintyIsLearned = player.UNCERTAINTY_SOURCE === 'learned';

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                {/* Backdrop */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={onClose}
                    className="absolute inset-0 bg-black/80 backdrop-blur-sm"
                />

                {/* Modal */}
                <motion.div
                    initial={{ scale: 0.95, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.95, opacity: 0 }}
                    className="relative w-full max-w-3xl bg-[#0a0a0f] border border-white/10 rounded-2xl shadow-2xl overflow-hidden max-h-[90vh] overflow-y-auto"
                >
                    {/* Close Button */}
                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 p-2 text-gray-400 hover:text-white bg-white/5 rounded-full z-10"
                    >
                        <X size={20} />
                    </button>

                    {/* Header Section */}
                    <div className="relative p-8 pb-0">
                        <div className="absolute top-0 right-0 w-64 h-64 bg-neon-purple/20 blur-[100px] pointer-events-none" />

                        <div className="flex items-start gap-6 relative">
                            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-gray-800 to-black border-2 border-white/10 flex items-center justify-center text-2xl font-bold text-gray-500">
                                {player.PLAYER_NAME.split(" ").map(n => n[0]).join("")}
                            </div>

                            <div className="flex-1">
                                <h2 className="text-3xl font-bold text-white mb-1">{player.PLAYER_NAME}</h2>
                                <div className="flex items-center gap-3 text-sm text-gray-400 font-mono">
                                    <span className="bg-white/5 px-2 py-0.5 rounded">{player.MATCHUP}</span>
                                    <span>vs {player.OPPONENT}</span>
                                </div>

                                {/* Model Version Badge */}
                                <div className="flex gap-2 mt-3">
                                    <span className={clsx(
                                        "px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider flex items-center gap-1",
                                        isV2 ? "bg-neon-purple/20 border border-neon-purple/50 text-neon-purple" : "bg-white/5 border border-white/10 text-gray-400"
                                    )}>
                                        <Brain size={10} /> {player.MODEL_VERSION || 'V1'}
                                    </span>
                                    {uncertaintyIsLearned && (
                                        <span className="px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider bg-neon-green/10 border border-neon-green/30 text-neon-green flex items-center gap-1">
                                            <Activity size={10} /> Learned Uncertainty
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>


                    {/* Vegas & Market Context - Always Show for Consistency */}
                    <div className="mx-8 mt-6 p-4 bg-gradient-to-r from-amber-500/10 to-orange-500/5 border border-amber-500/20 rounded-xl">
                        <div className="text-[10px] text-amber-400 uppercase tracking-widest mb-2 font-bold flex items-center gap-1">
                            <Zap size={10} /> Vegas Context
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div>
                                <div className="text-[10px] text-gray-500">Game Total O/U</div>
                                <div className={clsx("text-xl font-bold", player.VEGAS_TOTAL ? "text-white" : "text-gray-600")}>
                                    {player.VEGAS_TOTAL || "N/A"}
                                </div>
                            </div>

                            <div>
                                <div className="text-[10px] text-gray-500">Pace Adjustment</div>
                                {player.VEGAS_ADJUSTMENT ? (
                                    <div className={clsx(
                                        "text-xl font-bold",
                                        player.VEGAS_ADJUSTMENT > 1 ? "text-neon-green" : "text-red-400"
                                    )}>
                                        {player.VEGAS_ADJUSTMENT > 1 ? '+' : ''}{((player.VEGAS_ADJUSTMENT - 1) * 100).toFixed(1)}%
                                    </div>
                                ) : (
                                    <div className="text-xl font-bold text-gray-600">-</div>
                                )}
                            </div>

                            <div>
                                <div className="text-[10px] text-gray-500">Market Line ({player.BEST_PROP})</div>
                                <div className={clsx("text-xl font-bold", player.MARKET_LINE ? "text-white" : "text-gray-600")}>
                                    {player.MARKET_LINE || "OFF"}
                                </div>
                            </div>

                            <div>
                                <div className="text-[10px] text-gray-500">Expected Value</div>
                                {player.EV !== null && player.EV !== undefined ? (
                                    <div className={clsx(
                                        "text-xl font-bold",
                                        player.EV > 0 ? "text-neon-green" : player.EV < -5 ? "text-red-400" : "text-gray-400"
                                    )}>
                                        {player.EV > 0 ? '+' : ''}{player.EV}%
                                    </div>
                                ) : (
                                    <div className="text-xl font-bold text-gray-600">--</div>
                                )}
                            </div>
                        </div>
                    </div>


                    {/* Consistency + Confidence Header */}
                    <div className="px-8 mt-6 flex justify-between items-end border-b border-white/5 pb-4">
                        <div>
                            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Strategies</div>
                            <div className="flex gap-2 flex-wrap">
                                <div className="px-2 py-1 bg-white/5 rounded border border-white/10 text-[10px] text-gray-300">
                                    Confidence: <span className="text-white font-bold">{(player.UNITS * 100).toFixed(0)}%</span>
                                </div>
                                <div className="px-2 py-1 bg-white/5 rounded border border-white/10 text-[10px] text-gray-300">
                                    Prediction: <span className="text-white font-bold">{player.BEST_VAL?.toFixed(1)}</span>
                                </div>
                                {player.EV_RECOMMENDATION && (
                                    <div className={clsx(
                                        "px-2 py-1 rounded text-[10px] font-bold",
                                        player.EV_RECOMMENDATION === 'OVER' ? "bg-neon-green/20 text-neon-green border border-neon-green/30" :
                                            player.EV_RECOMMENDATION === 'UNDER' ? "bg-blue-500/20 text-blue-400 border border-blue-500/30" :
                                                "bg-white/5 text-gray-400 border border-white/10"
                                    )}>
                                        {player.EV_RECOMMENDATION}
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="text-right">
                            <div className="text-[10px] text-gray-400 uppercase tracking-widest">Consistency</div>
                            <div className={clsx(
                                "text-4xl font-black",
                                player.CONSISTENCY > 0.8 ? "text-neon-green" :
                                    player.CONSISTENCY > 0.6 ? "text-amber-400" : "text-red-500"
                            )}>
                                {player.CONSISTENCY > 0.8 ? "A+" :
                                    player.CONSISTENCY > 0.65 ? "A" :
                                        player.CONSISTENCY > 0.5 ? "B" :
                                            player.CONSISTENCY > 0.3 ? "C" : "D"}
                            </div>
                        </div>
                    </div>

                    <div className="p-8">
                        {/* PREDICTION vs MARKET COMPARISON */}
                        {player.MARKET_LINE && (
                            <div className="mb-6 p-4 bg-[#0f0f13] border border-white/10 rounded-xl">
                                <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-3 font-bold flex items-center gap-1">
                                    <BarChart3 size={10} /> Model vs Market ({player.BEST_PROP})
                                </div>
                                <div className="relative h-8 bg-white/5 rounded-full overflow-hidden">
                                    {/* Market Line Marker */}
                                    <div
                                        className="absolute top-0 bottom-0 w-1 bg-amber-500 z-10"
                                        style={{ left: `${Math.min(Math.max((player.MARKET_LINE / (player.BEST_VAL * 1.5)) * 100, 10), 90)}%` }}
                                    />
                                    {/* Prediction Fill */}
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${Math.min((player.BEST_VAL / (player.BEST_VAL * 1.5)) * 100, 100)}%` }}
                                        transition={{ duration: 0.8, ease: "easeOut" }}
                                        className={clsx(
                                            "absolute left-0 top-0 bottom-0 rounded-full",
                                            player.BEST_VAL > player.MARKET_LINE ? "bg-neon-green/50" : "bg-red-500/50"
                                        )}
                                    />
                                </div>
                                <div className="flex justify-between mt-2 text-xs">
                                    <span className="text-gray-400">0</span>
                                    <div className="flex gap-4">
                                        <span className="text-amber-400">Line: {player.MARKET_LINE}</span>
                                        <span className={player.BEST_VAL > player.MARKET_LINE ? "text-neon-green" : "text-red-400"}>
                                            Pred: {player.BEST_VAL?.toFixed(1)}
                                        </span>
                                    </div>
                                    <span className="text-gray-400">{(player.BEST_VAL * 1.5).toFixed(0)}</span>
                                </div>
                            </div>
                        )}

                        {/* DATA CENTER GRID with Uncertainty */}
                        <div className="bg-[#0f0f13] border border-white/10 rounded-xl overflow-hidden mb-8 shadow-2xl">
                            <div className="grid grid-cols-6 gap-0 text-center bg-white/5 text-[10px] text-gray-400 uppercase font-bold py-3 border-b border-white/5">
                                <div className="text-left pl-6">Stat</div>
                                <div>Season</div>
                                <div>L5</div>
                                <div>Pred ± Std</div>
                                <div>68% Range</div>
                                <div>Safe Line</div>
                            </div>

                            {['PTS', 'REB', 'AST'].map(stat => {
                                const sea = player[`SEASON_${stat}`] || 0;
                                const l5 = player.LAST_5 ? (stat === 'PTS' ? player.LAST_5.avg_pts : stat === 'REB' ? player.LAST_5.avg_reb : player.LAST_5.avg_ast) : 0;
                                const proj = player[`PRED_${stat}`] || 0;
                                const mae = player[`MAE_${stat}`] || 0;
                                const safe = player[`LINE_${stat}_LOW`] || 0;
                                const high = player[`LINE_${stat}_HIGH`] || 0;

                                const isHeating = l5 > sea * 1.05;
                                const isCooling = l5 < sea * 0.95;

                                return (
                                    <div key={stat} className="grid grid-cols-6 gap-0 text-center border-b border-white/5 py-4 hover:bg-white/5 transition-all group items-center">
                                        <div className="text-left pl-6 font-bold text-gray-300 text-sm">{stat}</div>

                                        <div className="text-gray-500 font-mono text-sm">{sea.toFixed(1)}</div>

                                        <div className={clsx(
                                            "font-mono font-bold text-sm",
                                            isHeating ? "text-neon-green" : isCooling ? "text-blue-400" : "text-gray-300"
                                        )}>
                                            {l5?.toFixed(1) || "-"}
                                            {isHeating && <TrendingUp size={12} className="inline ml-1 mb-0.5" />}
                                            {isCooling && <TrendingDown size={12} className="inline ml-1 mb-0.5" />}
                                        </div>

                                        <div className="text-white font-mono">
                                            <span className="font-black text-base">{proj.toFixed(1)}</span>
                                            {mae > 0 && (
                                                <span className="text-gray-500 text-xs ml-1">±{mae.toFixed(1)}</span>
                                            )}
                                        </div>

                                        <div className="text-gray-400 font-mono text-xs">
                                            {safe.toFixed(1)} - {high.toFixed(1)}
                                        </div>

                                        <div className="text-neon-green font-mono font-bold text-sm">
                                            &gt; {safe?.toFixed(1)}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        {/* Main Recommendation Row */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="bg-gradient-to-br from-neon-green/5 to-transparent border border-neon-green/20 rounded-xl p-5 flex items-center justify-between relative overflow-hidden">
                                <div className="absolute top-0 right-0 w-32 h-32 bg-neon-green/10 blur-[50px] pointer-events-none" />
                                <div>
                                    <div className="text-[10px] text-neon-green font-bold uppercase mb-1 tracking-widest flex items-center gap-1">
                                        <Target size={10} /> Model Pick
                                    </div>
                                    <div className="text-3xl font-black text-white">{player.BEST_PROP} {player.BEST_VAL?.toFixed(1)}</div>
                                    <div className="text-xs text-gray-400 mt-1">
                                        Safe: {player.LINE_LOW?.toFixed(1)} | MAE: ±{player[`MAE_${player.BEST_PROP}`]?.toFixed(1) || 'N/A'}
                                    </div>
                                </div>
                                <div className="text-right z-10">
                                    <div className="text-2xl font-bold text-white">{player.UNITS}u</div>
                                    <div className={clsx(
                                        "text-[10px] font-bold px-2 py-0.5 rounded border",
                                        player.BADGE?.includes("DIAMOND") ? "border-neon-purple text-neon-purple bg-neon-purple/10" : "border-gray-600 text-gray-400"
                                    )}>{player.BADGE}</div>
                                </div>
                            </div>

                            {/* Analysis + Model Info */}
                            <div className="bg-white/5 border border-white/5 rounded-xl p-5 flex flex-col justify-center">
                                <div className="text-[10px] text-gray-500 uppercase mb-2 font-bold">Analysis</div>
                                <div className="text-xs text-gray-300 leading-relaxed mb-3">
                                    {player.CONSISTENCY > 0.8
                                        ? "Elite Consistency (A). Low variance detected. Standard sizing approved."
                                        : player.CONSISTENCY > 0.5
                                            ? "Moderate Variance (B). Form is slightly volatile. Stick to recommended sizing."
                                            : "High Volatility (C/D). Position size automatically reduced."}
                                </div>
                                {isV2 && (
                                    <div className="text-[10px] text-neon-purple border-t border-white/5 pt-2 mt-auto">
                                        <span className="opacity-60">Model outputs probability distribution with learned uncertainty</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                </motion.div>
            </div>
        </AnimatePresence>
    );
}

