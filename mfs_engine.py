# mfs_engine.py

def calculate_mfs_score():
    # ŞİMDİLİK SABİT – engine zaten sende var
    scores = {
        "doviz": 70,
        "cds": 80,
        "global": 75,
        "faiz": 60,
        "likidite": 65
    }

    weights = {
        "doviz": 0.30,
        "cds": 0.25,
        "global": 0.25,
        "faiz": 0.15,
        "likidite": 0.05
    }

    total = int(sum(scores[k] * weights[k] for k in scores))

    regime = "ON" if total >= 60 else "OFF"

    return total, regime, scores
