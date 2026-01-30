import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { clsx } from "clsx";
import { Star, Edit2 } from "lucide-react";

const API_BASE = "http://localhost:8000"; // Ensure this matches your backend

export default function BetCard({ pred, onClick, index }) {
    // Local state for custom overrides (map of stat -> result)
    const [customAnalysis, setCustomAnalysis] = useState({});

    // Calculate Best Market Play dynamically (merging base pred with custom overrides)
    const activePred = useMemo(() => {
        // Deep copy pred to avoid mutating props
        const base = { ...pred, PROPS_ANALYSIS: { ...pred.PROPS_ANALYSIS } };

        // Merge custom analysis
        Object.entries(customAnalysis).forEach(([stat, result]) => {
            // Update the Top Level Logic if this becomes the best play
            // But primarily update the PROPS_ANALYSIS for that stat
            if (!base.PROPS_ANALYSIS[stat]) base.PROPS_ANALYSIS[stat] = {};

            // We only update what we need for the UI to reflect the change
            // The result from API returns { units, badge, line_low... }
            // We want to update the "Market Line" display effectively.

            // Actually, let's store the custom line in the analysis structure
            base.PROPS_ANALYSIS[stat] = {
                ...base.PROPS_ANALYSIS[stat],
                market_line: result.line, // The custom line
                recommendation: result.units >= 0.25 ? (result.val > result.line ? 'OVER' : 'UNDER') : 'PASS',
                custom_units: result.units,
                custom_badge: result.badge
            };
        });
        return base;
    }, [pred, customAnalysis]);

    const bestMarketProp = useMemo(() => {
        let bestProp = null;
        let highestScore = -1;

        const stats = ['PTS', 'REB', 'AST', '3PM', 'BLK', 'STL', 'RA', 'PR', 'PA', 'PRA'];

        stats.forEach(stat => {
            const analysis = activePred.PROPS_ANALYSIS?.[stat];
            if (!analysis) return;

            // Calculate Score (Effective Win Probability)
            let prob = 0;

            // 1. Custom Override?
            if (analysis.custom_units !== undefined) {
                // API returns win_prob as 0.xx, map to 0-100
                // If not available in custom response, approximate from units? 
                // We should ensure API returns it.
                // Fallback: Units * 20 + 50? No, let's look at what we have.
                // The API response is merged into analysis. 
                // Let's assume analysis.win_prob (decimal) * 100
                if (analysis.win_prob) prob = analysis.win_prob * 100;
                else prob = 50 + (analysis.custom_units * 20); // Rough proxy if missing
            } else {
                // 2. Default Backend Analysis
                // p_over is 0-100.
                const pOver = analysis.p_over || 0;
                // If UNDER is recommended, prob is (100 - pOver)? 
                // Analysis dict assumes p_over is "Probability of OVER".
                // effectiveWinProb logic:
                if (analysis.recommendation === 'UNDER') prob = 100 - pOver;
                else prob = pOver;
            }

            // Simple logic: if this stat breaks the previous high score
            if (prob > highestScore) {
                highestScore = prob;
                bestProp = stat;
            }
        });

        // Filter: If highest score is garbage (<50%), maybe don't show a star?
        // But usually we want to show the 'Relative Best'.
        return bestProp;
    }, [activePred]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="glass-card p-5 rounded-2xl relative overflow-hidden group hover:border-neon-green/50 transition-all cursor-pointer hover:bg-white/5"
        >
            {/* Badge Overlay */}
            <div className="absolute top-4 right-4 z-10">
                <div className={clsx(
                    "px-2 py-1 rounded text-[10px] font-bold border",
                    activePred.BADGE.includes("DIAMOND") ? "bg-neon-purple/20 border-neon-purple text-neon-purple" :
                        activePred.BADGE.includes("GOLD") ? "bg-amber-400/20 border-amber-400 text-amber-400" :
                            "bg-white/10 border-white/20 text-gray-400"
                )}>
                    {/* Show Custom Badge if Best Prop is modified? Complex. Keep simple for now */}
                    {activePred.BADGE}
                </div>
            </div>

            <div className="mb-4 pr-16" onClick={() => onClick(activePred)}>
                <h3 className="text-lg font-bold text-white truncate">{activePred.PLAYER_NAME}</h3>
                <p className="text-xs text-gray-500 font-mono mt-0.5">{activePred.MATCHUP}</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-4 gap-2 mb-4 bg-black/40 p-2 rounded-xl border border-white/5">
                {[
                    { label: 'PTS', val: activePred.PRED_PTS, mae: activePred.MAE_PTS },
                    { label: 'REB', val: activePred.PRED_REB, mae: activePred.MAE_REB },
                    { label: 'AST', val: activePred.PRED_AST, mae: activePred.MAE_AST },
                    { label: '3PM', val: activePred.PRED_3PM, mae: activePred.MAE_3PM },
                    { label: 'PRA', val: activePred.PRED_PRA, mae: activePred.MAE_PRA },
                    { label: 'RA', val: activePred.PRED_RA, mae: activePred.MAE_REB + activePred.MAE_AST }, // Approx
                    { label: 'PR', val: activePred.PRED_PR, mae: activePred.MAE_PTS + activePred.MAE_REB },
                    { label: 'PA', val: activePred.PRED_PA, mae: activePred.MAE_PTS + activePred.MAE_AST },
                ].map((stat) => (
                    <StatCell
                        key={stat.label}
                        stat={stat}
                        pred={activePred}
                        isBest={bestMarketProp === stat.label}
                        onUpdate={(result) => setCustomAnalysis(prev => ({ ...prev, [stat.label]: result }))}
                    />
                ))}
            </div>

            <div className="flex items-end justify-between border-t border-white/5 pt-3" onClick={() => onClick(activePred)}>
                <div>
                    <div className="text-[10px] text-white/40 uppercase tracking-widest mb-1">
                        <span className="text-neon-green font-bold mr-1">★</span>
                        BEST PLAY
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-black text-white">{bestMarketProp || activePred.BEST_PROP} {(activePred.PROPS_ANALYSIS?.[bestMarketProp]?.market_line || activePred.BEST_VAL)?.toFixed(1)}</span>
                    </div>
                </div>

                <div className="text-right">
                    <div className="text-[10px] text-gray-500 mb-1">UNITS</div>
                    <div className={clsx(
                        "text-xl font-bold font-mono",
                        activePred.UNITS >= 1.0 ? "text-neon-green" : "text-white"
                    )}>
                        {activePred.UNITS}u
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

function StatCell({ stat, pred, isBest, onUpdate }) {
    const [isEditing, setIsEditing] = useState(false);
    const [tempLine, setTempLine] = useState("");
    const [loading, setLoading] = useState(false);

    const analysis = pred.PROPS_ANALYSIS?.[stat.label];
    const marketLine = analysis?.market_line;

    // Determine status
    let statusColor = "text-white";
    let bgClass = "";

    // Calculate Win Probability 
    const prob = analysis?.p_over || 0;
    const effectiveWinProb = analysis?.recommendation === 'UNDER' ? (100 - prob) : prob;
    const isGoodPlay = effectiveWinProb > 55;

    if (analysis?.custom_units !== undefined) {
        // Custom State
        if (analysis.custom_units >= 0.75) statusColor = "text-neon-green";
        else if (analysis.custom_units >= 0.5) statusColor = "text-amber-400";
    } else {
        // Default State
        if (isBest) statusColor = "text-neon-green";
        else if (isGoodPlay) statusColor = "text-neon-green"; // Highlight other good plays
    }

    if (isBest) {
        bgClass = "bg-neon-green/10 ring-1 ring-neon-green/50";
    } else if (isGoodPlay) {
        // Subtler highlight for good plays that aren't the star
        bgClass = "bg-neon-green/5";
    }

    const handleEditStart = (e) => {
        e.stopPropagation();
        setTempLine(marketLine?.toString() || stat.val.toFixed(1));
        setIsEditing(true);
    };

    const handleSubmit = async () => {
        setIsEditing(false);
        const newLine = parseFloat(tempLine);
        if (isNaN(newLine)) return;

        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/analyze-custom-bet`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    player_id: pred.PLAYER_ID,
                    prop_type: stat.label,
                    custom_line: newLine,
                    prediction: stat.val,
                    mae: stat.mae || 1.0,
                    recent_stats: pred.LAST_5
                })
            });
            const data = await res.json();
            onUpdate({ ...data, line: newLine });
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={clsx(
            "text-center p-1 rounded min-h-[55px] flex flex-col justify-center relative transition-all group/cell",
            bgClass
        )}>
            {/* Hover Edit Icon */}
            {!isEditing && !loading && (
                <div
                    onClick={handleEditStart}
                    className="absolute top-0 right-0 p-1 opacity-0 group-hover/cell:opacity-100 cursor-pointer hover:bg-white/10 rounded"
                >
                    <Edit2 size={8} className="text-gray-400" />
                </div>
            )}

            {isBest && (
                <div className="absolute top-1 right-1 pointer-events-none">
                    <Star size={8} className="text-neon-green fill-neon-green" />
                </div>
            )}

            <div className="text-[9px] text-gray-500 font-bold">{stat.label}</div>

            <div className={clsx(
                "text-md font-bold leading-none my-0.5",
                statusColor
            )}>
                {stat.val?.toFixed(1) || "-"}
            </div>

            {/* Interactive Line Area */}
            {isEditing ? (
                <input
                    autoFocus
                    type="number"
                    value={tempLine}
                    onChange={(e) => setTempLine(e.target.value)}
                    onBlur={handleSubmit}
                    onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                    onClick={(e) => e.stopPropagation()}
                    className="w-full text-center bg-black/50 text-white text-[10px] border border-neon-green rounded px-0 py-0.5 focus:outline-none"
                    step="0.5"
                />
            ) : (
                <div
                    onClick={handleEditStart}
                    className="flex flex-col items-center cursor-pointer hover:bg-white/5 rounded px-1 transition-colors"
                >
                    {loading ? (
                        <div className="w-3 h-3 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                    ) : marketLine ? (
                        <>
                            <div className="text-[9px] text-gray-400 font-mono leading-tight">
                                {marketLine}
                            </div>
                            <div className={clsx(
                                "text-[8px] font-bold uppercase leading-tight",
                                analysis?.recommendation === 'OVER' ? "text-neon-green" :
                                    analysis?.recommendation === 'UNDER' ? "text-blue-400" :
                                        "text-gray-600"
                            )}>
                                {analysis?.custom_units !== undefined
                                    ? `${analysis.custom_units}u`
                                    : analysis?.recommendation || '-'}
                            </div>
                        </>
                    ) : (
                        <div className="h-4" /> // Spacer
                    )}
                </div>
            )}
        </div>
    );
}
