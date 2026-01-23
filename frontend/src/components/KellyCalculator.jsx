import { useState, useEffect } from "react";
import { Calculator } from "lucide-react";
import { clsx } from "clsx";

export default function KellyCalculator() {
    const [bankroll, setBankroll] = useState(1000);
    const [odds, setOdds] = useState(1.91);
    const [winProb, setWinProb] = useState(55);
    const [kellyFraction, setKellyFraction] = useState(0.25); // Quarter Kelly is standard for sports

    const [betSize, setBetSize] = useState(0);
    const [betPercent, setBetPercent] = useState(0);

    useEffect(() => {
        // Kelly Formula: f = (bp - q) / b
        // b = odds - 1
        // p = win probability
        // q = lose probability (1-p)

        const b = odds - 1;
        const p = winProb / 100;
        const q = 1 - p;

        if (b <= 0) return;

        const f = (b * p - q) / b;

        // Apply Fraction (Safety)
        const safeF = f * kellyFraction;

        if (safeF > 0) {
            setBetPercent(safeF * 100);
            setBetSize(bankroll * safeF);
        } else {
            setBetPercent(0);
            setBetSize(0);
        }

    }, [bankroll, odds, winProb, kellyFraction]);

    return (
        <div className="bg-[#0f0f13] border border-white/10 rounded-2xl p-6 relative overflow-hidden">
            <div className="flex items-center gap-2 mb-4">
                <div className="bg-neon-purple/20 p-2 rounded-lg">
                    <Calculator size={20} className="text-neon-purple" />
                </div>
                <h2 className="font-bold text-white">Bankroll Laboratory</h2>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                    <label className="block text-gray-500 text-xs mb-1">Bankroll ($)</label>
                    <input
                        type="number"
                        value={bankroll}
                        onChange={(e) => setBankroll(Number(e.target.value))}
                        className="w-full bg-white/5 border border-white/10 rounded p-2 text-white font-mono focus:border-neon-purple outline-none"
                    />
                </div>
                <div>
                    <label className="block text-gray-500 text-xs mb-1">Odds (Decimal)</label>
                    <input
                        type="number"
                        step="0.01"
                        value={odds}
                        onChange={(e) => setOdds(Number(e.target.value))}
                        className="w-full bg-white/5 border border-white/10 rounded p-2 text-white font-mono focus:border-neon-purple outline-none"
                    />
                </div>
                <div>
                    <label className="block text-gray-500 text-xs mb-1">Win Probability (%)</label>
                    <input
                        type="number"
                        value={winProb}
                        onChange={(e) => setWinProb(Number(e.target.value))}
                        className="w-full bg-white/5 border border-white/10 rounded p-2 text-white font-mono focus:border-neon-purple outline-none"
                    />
                </div>
                <div>
                    <label className="block text-gray-500 text-xs mb-1">Kelly Strategy</label>
                    <select
                        value={kellyFraction}
                        onChange={(e) => setKellyFraction(Number(e.target.value))}
                        className="w-full bg-white/5 border border-white/10 rounded p-2 text-white text-xs outline-none"
                    >
                        <option value={1.0}>Full Kelly (Aggressive)</option>
                        <option value={0.5}>Half Kelly (Standard)</option>
                        <option value={0.25}>Quarter Kelly (Safe)</option>
                        <option value={0.125}>Eighth Kelly (Conservative)</option>
                    </select>
                </div>
            </div>

            <div className="mt-6 pt-4 border-t border-white/5">
                <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-xs uppercase tracking-widest">Recommended Wager</span>
                    <div className="text-right">
                        <div className={clsx(
                            "text-2xl font-black font-mono",
                            betSize > 0 ? "text-neon-green" : "text-gray-500"
                        )}>
                            ${betSize.toFixed(2)}
                        </div>
                        <div className="text-xs text-gray-500">
                            {betPercent.toFixed(2)}% of Bankroll
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
