"""AI trader for pump.fun: scores new coins from live on-chain flow and trades them.

Paper trading is the default. Live trading reuses the repo's `PlatformAwareBuyer`
and `PlatformAwareSeller`, so the same transaction code the sniper uses sends
every order. See docs/ai-trader.md.
"""
