import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, TrendingUp, TrendingDown, Target, Award, Activity, Zap, BarChart3, Percent, Brain, ChevronDown, ChevronUp } from "lucide-react";
import { clsx } from "clsx";

export default function PlayerDetailModal({ isOpen, onClose, player }) {
    const [showAdvanced, setShowAdvanced] = useState(false);

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


                    {/* Minutes & Injury Context */}
                    <div className="mx-8 mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Minutes Projection */}
                        <div className="bg-white/5 border border-white/5 rounded-xl p-4 flex items-center justify-between">
                            <div>
                                <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1 font-bold flex items-center gap-1">
                                    <Activity size={10} /> Minutes Projection
                                </div>
                                <div className="flex items-baseline gap-2">
                                    <span className="text-2xl font-bold text-white">{player.PRED_MIN?.toFixed(1) || "-"}</span>
                                    <span className="text-xs text-gray-500">min</span>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="text-[10px] text-gray-500">Season Avg</div>
                                <div className="text-sm font-mono text-gray-300">{player.SEASON_AVG_MIN?.toFixed(1) || "-"}</div>
                                {player.MIN_DELTA_PCT !== undefined && (
                                    <div className={clsx(
                                        "text-xs font-bold",
                                        player.MIN_DELTA_PCT > 0 ? "text-neon-green" : player.MIN_DELTA_PCT < 0 ? "text-red-400" : "text-gray-500"
                                    )}>
                                        {player.MIN_DELTA_PCT > 0 ? "↑" : player.MIN_DELTA_PCT < 0 ? "↓" : ""}{Math.abs(player.MIN_DELTA_PCT).toFixed(0)}% vs avg
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Injury Context */}
                        {Array.isArray(player.MISSING_TEAMMATES) && player.MISSING_TEAMMATES.length > 0 ? (
                            <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4">
                                <div className="text-[10px] text-amber-500 uppercase tracking-widest mb-2 font-bold flex items-center gap-1">
                                    <Zap size={10} /> Injury Impact
                                </div>
                                <div className="flex flex-wrap gap-2">
                                    {player.MISSING_TEAMMATES.map((teammate, i) => (
                                        <div key={i} className="px-2 py-1 bg-amber-500/10 rounded text-[10px] text-amber-200 border border-amber-500/20">
                                            {teammate} <span className="opacity-50">(OUT)</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ) : (
                            <div className="bg-white/5 border border-white/5 rounded-xl p-4 flex items-center justify-center opacity-50">
                                <div className="text-xs text-gray-500 font-mono">No Key Injuries Detected</div>
                            </div>
                        )}
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
                                <div>Season / L5</div>
                                <div>Pred ± Std</div>
                                <div>Market Line</div>
                                <div>Win Prob</div>
                                <div>EV</div>
                            </div>

                            {['PTS', 'REB', 'AST', '3PM', 'BLK', 'STL'].map(stat => {
                                const sea = player[`SEASON_${stat}`] || 0;
                                let l5 = 0;
                                if (player.LAST_5) {
                                    if (stat === 'PTS') l5 = player.LAST_5.avg_pts;
                                    else if (stat === 'REB') l5 = player.LAST_5.avg_reb;
                                    else if (stat === 'AST') l5 = player.LAST_5.avg_ast;
                                    else if (stat === '3PM') l5 = player.LAST_5.avg_3pm;
                                    else if (stat === 'BLK') l5 = player.LAST_5.avg_blk;
                                    else if (stat === 'STL') l5 = player.LAST_5.avg_stl;
                                }

                                const proj = player[`PRED_${stat}`] || 0;
                                const mae = player[`MAE_${stat}`] || 0;

                                // Get Analysis Data
                                const analysis = player.PROPS_ANALYSIS?.[stat] || {};
                                const marketLine = analysis.market_line;
                                const marketOdds = analysis.odds_over;
                                const winProb = analysis.p_over;
                                const ev = analysis.ev_over;

                                const isHeating = l5 > sea * 1.05;
                                const isCooling = l5 < sea * 0.95;

                                return (
                                    <div key={stat} className="grid grid-cols-6 gap-0 text-center border-b border-white/5 py-4 hover:bg-white/5 transition-all group items-center">
                                        <div className="text-left pl-6 font-bold text-gray-300 text-sm">{stat}</div>

                                        <div className="flex flex-col justify-center items-center">
                                            <div className="text-gray-500 font-mono text-xs">{sea.toFixed(1)}</div>
                                            <div className={clsx(
                                                "font-mono font-bold text-xs",
                                                isHeating ? "text-neon-green" : isCooling ? "text-blue-400" : "text-gray-400"
                                            )}>
                                                {l5?.toFixed(1) || "-"}
                                            </div>
                                        </div>

                                        <div className="text-white font-mono">
                                            <span className="font-black text-base">{proj.toFixed(1)}</span>
                                            {mae > 0 && (
                                                <span className="text-gray-500 text-xs ml-1">±{mae.toFixed(1)}</span>
                                            )}
                                        </div>

                                        <div className="flex flex-col justify-center items-center">
                                            {marketLine ? (
                                                <>
                                                    <span className="text-white font-bold text-sm">{marketLine}</span>
                                                    <span className="text-[10px] text-gray-500">{marketOdds}</span>
                                                </>
                                            ) : (
                                                <span className="text-gray-600 text-xs">-</span>
                                            )}
                                        </div>

                                        <div className="text-gray-300 font-mono text-sm font-bold">
                                            {winProb ? (
                                                <span className={winProb > 60 ? "text-neon-green" : winProb > 55 ? "text-white" : "text-gray-500"}>
                                                    {winProb}%
                                                </span>
                                            ) : "-"}
                                        </div>

                                        <div className="font-mono font-bold text-sm">
                                            {ev !== undefined ? (
                                                <span className={
                                                    ev > 5 ? "text-neon-green" :
                                                        ev > 0 ? "text-white" :
                                                            "text-red-400"
                                                }>
                                                    {ev > 0 ? '+' : ''}{ev}%
                                                </span>
                                            ) : "-"}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        {/* Advanced Metrics (Collapsible) */}
                        <div className="mb-4 border border-white/5 rounded-xl bg-[#0f0f13] overflow-hidden">
                            <button
                                onClick={() => setShowAdvanced(!showAdvanced)} // Ensure setShowAdvanced is defined
                                className="w-full flex items-center justify-between p-3 bg-white/5 hover:bg-white/10 transition-colors"
                            >
                                <div className="text-[10px] text-gray-400 uppercase font-bold flex items-center gap-2">
                                    <Activity size={12} /> Advanced Metrics (Calculated Efficiency)
                                </div>
                                {showAdvanced ? <ChevronUp size={14} className="text-gray-400" /> : <ChevronDown size={14} className="text-gray-400" />}
                            </button>

                            <AnimatePresence>
                                {showAdvanced && (
                                    <motion.div
                                        initial={{ height: 0, opacity: 0 }}
                                        animate={{ height: "auto", opacity: 1 }}
                                        exit={{ height: 0, opacity: 0 }}
                                        className="overflow-hidden"
                                    >
                                        <div className="p-4 space-y-2 border-t border-white/5">
                                            <div className="grid grid-cols-4 text-[9px] text-gray-500 uppercase font-bold mb-2 pl-2">
                                                <div className="col-span-1">Stat</div>
                                                <div className="col-span-1 text-center">Rate / Min</div>
                                                <div className="col-span-1 text-center">Proj Min</div>
                                                <div className="col-span-1 text-right">Proj Total</div>
                                            </div>
                                            {['PTS', 'REB', 'AST', '3PM', 'PRA'].map(stat => {
                                                const rate = player[`PRED_${stat}_PER_MIN`] || 0;
                                                const mins = player.PRED_MIN || 0;
                                                const total = rate * mins;

                                                return (
                                                    <div key={stat} className="grid grid-cols-4 items-center bg-black/40 rounded p-2 border border-white/5">
                                                        <div className="col-span-1 font-bold text-gray-300 text-xs">{stat}</div>
                                                        <div className="col-span-1 font-mono text-neon-blue text-xs text-center">{rate.toFixed(3)}/m</div>
                                                        <div className="col-span-1 font-mono text-gray-400 text-xs text-center">× {mins.toFixed(1)}m</div>
                                                        <div className="col-span-1 font-mono text-white font-bold text-right text-xs">= {total.toFixed(1)}</div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
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
                                {/* Analysis Content */}
                                <div className="text-[10px] text-gray-500 uppercase mb-2 font-bold flex justify-between items-center">
                                    <span>Analysis</span>
                                    {player.MODEL_STATUS && (
                                        <span className={clsx(
                                            "leading-none px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider",
                                            player.MODEL_STATUS === "FRESH" ? "bg-neon-green/10 text-neon-green" :
                                                player.MODEL_STATUS === "FAILED" ? "bg-red-500/10 text-red-400" :
                                                    player.MODEL_STATUS === "UNKNOWN" ? "bg-white/5 text-gray-400" : "bg-blue-400/10 text-blue-400"
                                        )}>
                                            {player.MODEL_STATUS}
                                        </span>
                                    )}
                                </div>

                                <div className="text-xs text-gray-300 leading-relaxed mb-4">
                                    {player.CONSISTENCY > 0.8
                                        ? "Elite Consistency (A). Low variance detected. Standard sizing approved."
                                        : player.CONSISTENCY > 0.5
                                            ? "Moderate Variance (B). Form is slightly volatile. Stick to recommended sizing."
                                            : "High Volatility (C/D). Position size automatically reduced."}
                                </div>

                                {/* Model Metadata */}
                                <div className="mt-auto border-t border-white/5 pt-3 space-y-2">
                                    <div className="flex justify-between text-[10px]">
                                        <span className="text-gray-500">Last Trained</span>
                                        <span className="text-gray-300 font-mono">{player.MODEL_LAST_TRAINED || "Unknown"}</span>
                                    </div>
                                    {player.MODEL_TRIGGER_REASON && (
                                        <div className="flex justify-between text-[10px]">
                                            <span className="text-gray-500">Trigger</span>
                                            <span className="text-amber-400 font-mono">{player.MODEL_TRIGGER_REASON}</span>
                                        </div>
                                    )}

                                    {isV2 && (
                                        <div className="text-[9px] text-neon-purple mt-2 flex items-center gap-1 opacity-80">
                                            <Brain size={10} />
                                            <span>Learned Uncertainty Active</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>

                </motion.div>
            </div>
        </AnimatePresence>
    );
}

