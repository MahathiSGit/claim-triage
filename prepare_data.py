"""Phase 1: turn the 2015 US flight-delay data into a simple 'claims' dataset.

Input : data/flights.csv   (Kaggle: usdot/flight-delays)
Output: data/claims.csv
"""
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 50_000
rng = np.random.default_rng(SEED)

cols = [
    "MONTH", "DAY_OF_WEEK", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE", "DISTANCE", "ARRIVAL_DELAY", "CANCELLED",
    "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY", "SECURITY_DELAY",
]
df = pd.read_csv(
    "data/flights.csv", usecols=cols, low_memory=False,
    dtype={"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str},
)

# 1. Keep flights that look claim-eligible: arrived 3+ hours late
df = df[(df["ARRIVAL_DELAY"] >= 180) & (df["CANCELLED"] == 0)].copy()

# 2. Label from the cause of delay (used ONLY to build the label, never as a feature)
cause_cols = ["AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY", "SECURITY_DELAY"]
df[cause_cols] = df[cause_cols].fillna(0)
airline_fault = df["AIRLINE_DELAY"] + df["LATE_AIRCRAFT_DELAY"]
extraordinary = df["WEATHER_DELAY"] + df["SECURITY_DELAY"]

# Drop rows where only air-traffic delay is recorded (ambiguous cause)
df = df[(airline_fault + extraordinary) > 0].copy()
airline_fault, extraordinary = airline_fault[df.index], extraordinary[df.index]
df["approved"] = (airline_fault >= extraordinary).astype(int)

# 3. Add ~5% label noise to mimic inconsistent agent decisions
flip = rng.random(len(df)) < 0.05
df.loc[flip, "approved"] = 1 - df.loc[flip, "approved"]

# 4. Rules layer: distance in km and EU261-style compensation tier
df["distance_km"] = (df["DISTANCE"] * 1.609).round(0)
df["tier_eur"] = np.select(
    [df["distance_km"] <= 1500, df["distance_km"] <= 3500], [250, 400], default=600
)

# 5. Claim-time features only (no cause columns!)
df["delay_minutes"] = df["ARRIVAL_DELAY"].astype(int)
df["hour"] = df["SCHEDULED_DEPARTURE"] // 100
out = df.rename(columns={
    "MONTH": "month", "DAY_OF_WEEK": "day_of_week", "AIRLINE": "airline",
    "ORIGIN_AIRPORT": "origin", "DESTINATION_AIRPORT": "destination",
})[["month", "day_of_week", "hour", "airline", "origin", "destination",
    "distance_km", "delay_minutes", "tier_eur", "approved"]]

if len(out) > N_ROWS:
    out = out.sample(N_ROWS, random_state=SEED)
out = out.sort_values("month").reset_index(drop=True)
out.to_csv("data/claims.csv", index=False)

print("Rows saved:", len(out))
print("\nApproved share (1 = airline fault):")
print(out["approved"].value_counts(normalize=True).round(3))
print("\nRows per month (needed for the time-based split):")
print(out["month"].value_counts().sort_index())