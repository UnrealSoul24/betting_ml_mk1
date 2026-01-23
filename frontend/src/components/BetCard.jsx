import { motion } from "framer-motion";
import { clsx } from "clsx";

export default function BetCard({ pred, onClick, index }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            onClick={() => onClick(pred)}
            className="glass-card p-5 rounded-2xl relative overflow-hidden group hover:border-neon-green/50 transition-all cursor-pointer hover:bg-white/5"
        >
            {/* Badge Overlay */}
            <div className="absolute top-4 right-4">
                <div className={clsx(
                    "px-2 py-1 rounded text-[10px] font-bold border",
                    pred.BADGE.includes("DIAMOND") ? "bg-neon-purple/20 border-neon-purple text-neon-purple" :
                        pred.BADGE.includes("GOLD") ? "bg-amber-400/20 border-amber-400 text-amber-400" :
                            "bg-white/10 border-white/20 text-gray-400"
                )}>
                    {pred.BADGE}
                </div>
            </div>

            <div className="mb-4 pr-16">
                <h3 className="text-lg font-bold text-white truncate">{pred.PLAYER_NAME}</h3>
                <p className="text-xs text-gray-500 font-mono mt-0.5">{pred.MATCHUP}</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-3 gap-2 mb-4 bg-black/40 p-2 rounded-xl border border-white/5">
                {[
                    { label: 'PTS', val: pred.PRED_PTS, low: pred.LINE_PTS_LOW, high: pred.LINE_PTS_HIGH },
                    { label: 'REB', val: pred.PRED_REB, low: pred.LINE_REB_LOW, high: pred.LINE_REB_HIGH },
                    { label: 'AST', val: pred.PRED_AST, low: pred.LINE_AST_LOW, high: pred.LINE_AST_HIGH },
                ].map((stat) => (
                    <div key={stat.label} className={clsx(
                        "text-center p-1 rounded",
                        pred.BEST_PROP === stat.label ? "bg-neon-green/10 ring-1 ring-neon-green/50" : ""
                    )}>
                        <div className="text-[10px] text-gray-500 font-bold">{stat.label}</div>
                        <div className={clsx(
                            "text-lg font-bold",
                            pred.BEST_PROP === stat.label ? "text-neon-green" : "text-white"
                        )}>
                            {stat.val?.toFixed(1) || "-"}
                        </div>
                    </div>
                ))}
            </div>

            <div className="flex items-end justify-between border-t border-white/5 pt-3">
                <div>
                    <div className="text-[10px] text-white/40 uppercase tracking-widest mb-1">
                        <span className="text-neon-green font-bold mr-1">★</span>
                        BEST PLAY
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-black text-white">{pred.BEST_PROP} {pred.BEST_VAL.toFixed(1)}</span>
                        <span className="text-xs text-gray-500">
                            (Safe Line: {pred.LINE_LOW?.toFixed(1)})
                        </span>
                    </div>
                </div>

                <div className="text-right">
                    <div className="text-[10px] text-gray-500 mb-1">UNITS</div>
                    <div className={clsx(
                        "text-xl font-bold font-mono",
                        pred.UNITS >= 1.0 ? "text-neon-green" : "text-white"
                    )}>
                        {pred.UNITS}u
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
