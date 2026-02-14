import math
import numpy as np
import pandas as pd
import gradio as gr

PROFILES = {
    "Tour Pro": {
        "dispersion": {
            "Driver": (14, 14),
            "Fairway/Hybrid": (10, 10),
            "Long Iron": (9, 9),
            "Mid Iron": (7, 7),
            "Wedge": (5, 5),
        },
        "lat_bias_y": 0.0,
        "penalty_multiplier": 0.70,
        "short_game_offset": -0.20,
        "approach_offset": -0.10,
        "putting_skill": 1.00,
    },
    "Elite Am (+2)": {
        "dispersion": {
            "Driver": (20, 18),
            "Fairway/Hybrid": (13, 13),
            "Long Iron": (12, 12),
            "Mid Iron": (10, 10),
            "Wedge": (7, 7),
        },
        "lat_bias_y": -2.0,
        "penalty_multiplier": 1.00,
        "short_game_offset": 0.00,
        "approach_offset": 0.00,
        "putting_skill": 1.12,
    },
    "6 Handicap": {
        "dispersion": {
            "Driver": (30, 24),
            "Fairway/Hybrid": (18, 16),
            "Long Iron": (16, 16),
            "Mid Iron": (14, 14),
            "Wedge": (10, 10),
        },
        "lat_bias_y": -4.0,
        "penalty_multiplier": 1.25,
        "short_game_offset": +0.20,
        "approach_offset": +0.08,
        "putting_skill": 1.30,
    }
}

CLUBS = [
    ("Driver", 300, "Driver"),
    ("3W", 245, "Fairway/Hybrid"),
    ("5W", 230, "Fairway/Hybrid"),
    ("3H", 220, "Fairway/Hybrid"),
    ("4i", 210, "Long Iron"),
    ("5i", 200, "Long Iron"),
    ("6i", 192, "Long Iron"),
    ("7i", 185, "Mid Iron"),
    ("8i", 172, "Mid Iron"),
    ("9i", 160, "Mid Iron"),
    ("PW", 148, "Wedge"),
    ("GW", 136, "Wedge"),
    ("SW", 122, "Wedge"),
    ("LW", 110, "Wedge"),
]

def base_expected_strokes(distance_y, lie, on_green=False, putt_ft=None):
    if on_green:
        d = max(1.0, float(putt_ft or 15.0))
        exp_putts = 1.05 + 0.28 * math.log(d / 3.0 + 1.0)
        return exp_putts
    d = max(1.0, float(distance_y))
    fw = 2.15 + 0.55 * math.log(d / 20.0 + 1.0)
    adj = {
        "Fairway": 0.00,
        "First cut": 0.05,
        "Rough (light)": 0.15,
        "Rough (heavy)": 0.30,
        "Greenside bunker": 0.45,
        "Fairway bunker": 0.20,
        "Recovery (trees)": 0.55
    }.get(lie, 0.15)
    return fw + adj

def profile_expected_strokes(profile, distance_y, lie, on_green=False, putt_ft=None):
    p = PROFILES[profile]
    base = base_expected_strokes(distance_y, lie, on_green=on_green, putt_ft=putt_ft)
    if on_green:
        return base * p["putting_skill"]
    if distance_y <= 50:
        base += p["short_game_offset"]
    else:
        base += p["approach_offset"]
    return base

def classify_outcome(x, y, gd, gw, hz, miss_surface):
    half_d, half_w = gd/2.0, gw/2.0
    if abs(x) <= half_d and abs(y) <= half_w:
        putt_ft = math.sqrt(x*x + y*y) * 3.0
        return ("Green", putt_ft)

    left_edge, right_edge = -half_w, half_w
    front_edge, back_edge = -half_d, half_d

    if y < left_edge - hz["left_buf"]:
        return ("Penalty" if hz["left"]=="Penalty" else "Recovery (trees)" if hz["left"]=="Trees" else "Rough (light)", None)
    if y > right_edge + hz["right_buf"]:
        return ("Penalty" if hz["right"]=="Penalty" else "Recovery (trees)" if hz["right"]=="Trees" else "Rough (light)", None)
    if x < front_edge - hz["short_buf"]:
        return ("Penalty" if hz["short"]=="Penalty" else "Greenside bunker" if hz["short"]=="Bunker" else "Rough (light)", None)
    if x > back_edge + hz["long_buf"]:
        return ("Penalty" if hz["long"]=="Penalty" else "Rough (light)", None)

    return (miss_surface, None)

def candidates(distance_to_pin, wind_adj, elev_adj):
    plays = distance_to_pin + wind_adj + elev_adj
    sorted_by = sorted(CLUBS, key=lambda t: abs(t[1]-plays))
    return sorted_by[:7], plays

def run(distance_to_pin, lie, wind, elevation, green_depth, green_width, pin_side, pin_depth,
        short_side_danger, left_miss, right_miss, short_miss, long_miss,
        left_buf, right_buf, short_buf, long_buf, sims):

    wind_adj = -5.0 if wind=="Helping" else (7.0 if wind=="Hurt" else 0.0)
    elev_adj = -5.0 if elevation=="Downhill" else (5.0 if elevation=="Uphill" else 0.0)

    hz = {
        "left": left_miss, "right": right_miss, "short": short_miss, "long": long_miss,
        "left_buf": float(left_buf), "right_buf": float(right_buf),
        "short_buf": float(short_buf), "long_buf": float(long_buf)
    }

    pin_y = (-green_width*0.25 if pin_side=="Left" else (green_width*0.25 if pin_side=="Right" else 0.0))
    pin_x = (-green_depth*0.25 if pin_depth=="Front" else (green_depth*0.25 if pin_depth=="Back" else 0.0))

    fat_y = (green_width*0.20 if pin_side=="Left" else (-green_width*0.20 if pin_side=="Right" else 0.0))
    fat_x = (green_depth*0.15 if pin_depth=="Front" else (-green_depth*0.10 if pin_depth=="Back" else 0.0))

    aims = [("Pin", pin_x, pin_y), ("Middle", 0.0, 0.0), ("Fat-side", fat_x, fat_y)]

    miss_surface = "Rough (light)" if lie in ["Tee","Fairway","First cut","Fairway bunker"] else lie

    cands, plays = candidates(distance_to_pin, wind_adj, elev_adj)

    rows = []
    rng = np.random.default_rng()

    for club, carry, group in cands:
        for aim_name, ax, ay in aims:
            row = {"Option": f"{club} — {aim_name}", "Club": club, "Aim": aim_name}
            for prof in PROFILES:
                p = PROFILES[prof]
                sig_lat, sig_dist = p["dispersion"][group]
                lat_bias = p["lat_bias_y"]

                xs = rng.normal(loc=ax, scale=sig_dist, size=int(sims))
                ys = rng.normal(loc=ay + lat_bias, scale=sig_lat, size=int(sims))

                total = 0.0
                pen = 0
                green = 0

                for x, y in zip(xs, ys):
                    state, putt_ft = classify_outcome(x, y, green_depth, green_width, hz, miss_surface)
                    if state == "Penalty":
                        pen += 1
                        stroke = 1.0 + profile_expected_strokes(prof, distance_to_pin, "Fairway")
                    elif state == "Green":
                        green += 1
                        stroke = profile_expected_strokes(prof, 0.0, "Green", on_green=True, putt_ft=putt_ft)
                    else:
                        rem = max(1.0, math.sqrt(x*x + y*y))
                        stroke = profile_expected_strokes(prof, rem, state)
                    total += stroke

                pen_rate = pen / sims
                adj_pen = min(1.0, pen_rate * p["penalty_multiplier"])
                replacement = profile_expected_strokes(prof, 35, "Rough (light)")
                pen_cost = 1.0 + profile_expected_strokes(prof, distance_to_pin, "Fairway")
                pen_delta = (pen_cost - replacement) * (pen_rate - adj_pen)

                exp = (total / sims) - pen_delta
                row[f"{prof} E"] = exp
                row[f"{prof} SG"] = 0.0
                row[f"{prof} Pen%"] = adj_pen
                row[f"{prof} Green%"] = green / sims

            rows.append(row)

    df = pd.DataFrame(rows)

    # Reference: nearest club (by plays-like) aiming middle
    nearest = min(CLUBS, key=lambda t: abs(t[1] - plays))[0]
    ref_opt = f"{nearest} — Middle"
    ref = df[df["Option"] == ref_opt].head(1)
    if len(ref)==0:
        ref = df[df["Aim"]=="Middle"].head(1)

    for prof in PROFILES:
        refv = float(ref.iloc[0][f"{prof} E"])
        df[f"{prof} SG"] = refv - df[f"{prof} E"]

    df["Avg E"] = df[[f"{p} E" for p in PROFILES]].mean(axis=1)
    df = df.sort_values("Avg E").reset_index(drop=True)

    # Compact output
    out = df[["Option","Avg E"] +
             [f"{p} SG" for p in PROFILES] +
             [f"{p} E" for p in PROFILES] +
             [f"{p} Pen%" for p in PROFILES] +
             [f"{p} Green%" for p in PROFILES]
            ].head(12)

    return out, f"Reference: {ref.iloc[0]['Option']} (plays like {plays:.0f}y)"

with gr.Blocks(title="Golf Shot Decision Engine (Mobile)") as demo:
    gr.Markdown("## Golf Shot Decision Engine (Mobile)\nTour Pro vs Elite Am (+2) vs 6 Handicap — Expected Strokes + Strokes Gained")

    with gr.Row():
        distance_to_pin = gr.Number(value=165, label="Distance to pin (yards)", precision=0)
        lie = gr.Dropdown(["Tee","Fairway","First cut","Rough (light)","Rough (heavy)","Fairway bunker","Recovery (trees)"], value="Fairway", label="Lie")

    with gr.Row():
        wind = gr.Radio(["Helping","None","Hurt"], value="None", label="Wind")
        elevation = gr.Radio(["Downhill","Flat","Uphill"], value="Flat", label="Elevation")

    with gr.Row():
        green_depth = gr.Slider(18,45, value=30, step=1, label="Green depth (y)")
        green_width = gr.Slider(18,45, value=32, step=1, label="Green width (y)")

    with gr.Row():
        pin_side = gr.Radio(["Left","Center","Right"], value="Center", label="Pin side")
        pin_depth = gr.Radio(["Front","Middle","Back"], value="Middle", label="Pin depth")

    short_side_danger = gr.Checkbox(value=True, label="Short-side dangerous")

    gr.Markdown("### Trouble")
    with gr.Row():
        left_miss = gr.Dropdown(["None","Rough","Trees","Penalty"], value="Penalty", label="Left miss")
        right_miss = gr.Dropdown(["None","Rough","Trees","Penalty"], value="Rough", label="Right miss")
        short_miss = gr.Dropdown(["None","Rough","Bunker","Penalty"], value="Bunker", label="Short miss")
        long_miss = gr.Dropdown(["None","Rough","Penalty"], value="Rough", label="Long miss")

    with gr.Row():
        left_buf = gr.Slider(0,25, value=8, step=1, label="Left buffer (y)")
        right_buf = gr.Slider(0,25, value=10, step=1, label="Right buffer (y)")
        short_buf = gr.Slider(0,25, value=6, step=1, label="Short buffer (y)")
        long_buf = gr.Slider(0,25, value=10, step=1, label="Long buffer (y)")

    sims = gr.Slider(1000, 15000, value=5000, step=1000, label="Simulations per option")

    btn = gr.Button("Rank shot options")
    ref_text = gr.Markdown()
    table = gr.Dataframe(interactive=False, wrap=True)

    btn.click(
        fn=run,
        inputs=[distance_to_pin, lie, wind, elevation, green_depth, green_width, pin_side, pin_depth,
                short_side_danger, left_miss, right_miss, short_miss, long_miss,
                left_buf, right_buf, short_buf, long_buf, sims],
        outputs=[table, ref_text]
    )

demo.launch()
