import os
import logging
import warnings
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from model import load_model, predict as ml_predict, get_known_locations, ensure_folders

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", 
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://localhost:8080"
).split(",")
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

_secret_from_env = os.environ.get("FLASK_SECRET_KEY")
app.secret_key = _secret_from_env if _secret_from_env else os.urandom(24).hex()

bundle = load_model()


def _build_input_dict(data: dict) -> dict:
    location = str(data.get("location", "")).strip()
    if not location:
        raise ValueError("الموقع مطلوب ولا يمكن أن يكون فارغاً.")

    def check_field(key, label, min_val, max_val, default=None, is_int=False):
        val = data.get(key)
        if val in (None, ""):
            if default is None:
                raise ValueError(f"الحقل '{label}' مطلوب.")
            return default
        try:
            val = int(val) if is_int else float(val)
        except (ValueError, TypeError):
            raise ValueError(f"يجب أن يكون '{label}' رقماً صالحاً.")
        if val < min_val or val > max_val:
            raise ValueError(f"قيمة '{label}' يجب أن تكون بين {min_val} و {max_val}.")
        return val

    def check_choice(key, label, allowed_values, default):
        val = str(data.get(key, "") or "").strip()
        if not val:
            return default
        if val not in allowed_values:
            raise ValueError(
                f"قيمة '{label}' غير معروفة: '{val}'. "
                f"القيم المسموحة: {', '.join(sorted(allowed_values))}."
            )
        return val

    total_area      = check_field("total_area", "المساحة الإجمالية", 20, 1000)
    net_area         = check_field("net_area", "المساحة الصافية", 15, 1000, default=total_area * 0.9)
    rooms_count      = check_field("rooms_count", "عدد الغرف", 1, 10, default=3, is_int=True)
    bathrooms_count  = check_field("bathrooms_count", "عدد الحمامات", 1, 5, default=1, is_int=True)
    halls_count      = check_field("halls_count", "عدد الصالات", 0, 5, default=1, is_int=True)
    floor_num        = check_field("floor_num", "رقم الطابق", -3, 15, default=0, is_int=True)
    building_age     = check_field("building_age", "عمر البناء", 0, 100, default=5)

    if net_area > total_area * 1.05:
        raise ValueError("المساحة الصافية لا يمكن أن تكون أكبر من المساحة الإجمالية.")

    heating_type = check_choice(
        "heating_type", "نوع التدفئة",
        {"لا يوجد", "تدفئة ديزل", "مدفئة ديزل", "تدفئة مركزية"},
        default="لا يوجد",
    )
    house_condition = check_choice(
        "house_condition", "حالة الشقة",
        {"مسكون من صاحبه", "فارغ", "مؤجر"},
        default="فارغ",
    )
    legal_status = check_choice(
        "legal_status", "الوضع القانوني",
        {"طابو أخضر/نظامي", "كاتب عدل/وكالة", "حكم محكمة"},
        default="طابو أخضر/نظامي",
    )

    def _flag(key):
        val = data.get(key, 0)
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, int):
            return val
        return 1 if str(val).strip() in {"1", "on", "true", "yes"} else 0

    return {
        "location":           location,
        "total_area":         total_area,
        "net_area":           net_area,
        "rooms_count":        rooms_count,
        "bathrooms_count":    bathrooms_count,
        "halls_count":        halls_count,
        "floor_num":          floor_num,
        "building_age_clean": building_age,
        "heating_type":       heating_type,
        "house_condition":    house_condition,
        "legal_status":       legal_status,
        "near_hospital":      _flag("near_hospital"),
        "near_market":        _flag("near_market"),
        "has_elevator":       _flag("has_elevator"),
        "has_parking":        _flag("has_parking"),
        "is_furnished":       _flag("is_furnished"),
    }


@app.route("/")
def home():
    locations = get_known_locations(bundle) if bundle else []
    return render_template("index.html", locations=locations, model_ready=bundle is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if bundle is None:
        flash("النموذج غير محمّل. يرجى تشغيل ملف النموذج أولاً.", "error")
        return redirect(url_for("home"))

    try:
        input_data = _build_input_dict(request.form)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("home"))

    try:
        price_raw = ml_predict(bundle, input_data)
        total_area = input_data["total_area"]
        price_per_sqm = price_raw / total_area if total_area > 0 else 0

        amenities_list = ["near_hospital", "near_market", "has_elevator", "has_parking", "is_furnished"]
        selected_amenities = [k for k in amenities_list if input_data[k] == 1]

        return render_template(
            "index.html",
            locations=get_known_locations(bundle),
            model_ready=True,
            result=True,
            price=f"{price_raw:,.0f}",
            price_per_sqm=f"{price_per_sqm:,.0f}",
            sel_location=input_data["location"],
            sel_area=int(total_area),
            sel_rooms=input_data["rooms_count"],
            sel_bathrooms=input_data["bathrooms_count"],
            sel_halls=input_data["halls_count"],
            sel_floor=input_data["floor_num"],
            sel_age=int(input_data["building_age_clean"]),
            selected_amenities=selected_amenities,
            sel_net_area=input_data["net_area"],
            sel_heating=input_data["heating_type"],
            sel_condition=input_data["house_condition"],
            sel_legal=input_data["legal_status"],
        )
    except Exception:
        flash("حدث خطأ أثناء حساب التقييم. يرجى التحقق من البيانات.", "error")
        return redirect(url_for("home"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if bundle is None:
        return jsonify({"success": False, "error": "Model not loaded."}), 503

    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "Invalid JSON body."}), 400
        input_data = _build_input_dict(data)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"success": False, "error": "Malformed JSON body."}), 400

    try:
        price_raw = ml_predict(bundle, input_data)
        total_area = input_data["total_area"]
        per_sqm = round(price_raw / total_area) if total_area > 0 else 0

        return jsonify({
            "success":             True,
            "predicted_price_usd": round(price_raw),
            "price_per_sqm_usd":   per_sqm,
            "location":            input_data["location"],
            "total_area_m2":       total_area,
        }), 200
    except Exception:
        return jsonify({
            "success": False,
            "error": "حدث خطأ داخلي أثناء معالجة الطلب. يرجى المحاولة لاحقاً."
        }), 500


@app.route("/api/locations", methods=["GET"])
def api_locations():
    if bundle is None:
        return jsonify({"success": False, "error": "Model not loaded.", "locations": []}), 503
    return jsonify({"success": True, "locations": get_known_locations(bundle)}), 200


@app.route("/api/status", methods=["GET"])
def api_status():
    if bundle is None:
        return jsonify({"status": "not_ready", "error": "No model found."}), 503

    metrics = bundle.get("metrics", {})
    return jsonify({
        "status":          "ready",
        "model_name":      bundle.get("model_name", "unknown"),
        "locations_count": len(bundle.get("target_enc_map", {})),
        "r2_score":        round(metrics.get("r2_score", 0), 4),
        "accuracy_15pct":  round(metrics.get("accuracy", 0), 1),
    }), 200


def print_banner(port):
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
    GRAY    = "\033[90m"

    line = f"{GRAY}──────────────────────────────────────────────────{RESET}"

    print()
    print(f"  🏠 {BOLD}{GREEN}HOMS REAL ESTATE API{RESET} {GRAY}v1.0.0{RESET}")
    print(f"  {line}")
    print(f"  {GREEN}➜{RESET}  {BOLD}Local Web UI:{RESET}       {CYAN}http://127.0.0.1:{port}/{RESET}")
    print(f"  {GREEN}➜{RESET}  {BOLD}API Status Check:{RESET}   {CYAN}http://127.0.0.1:{port}/api/status{RESET}")
    print(f"  {GREEN}➜{RESET}  {BOLD}Predict Endpoint:{RESET}   {MAGENTA}http://localhost:{port}/api/predict{RESET} {GRAY}[POST]{RESET}")
    print(f"  {line}")
    print(f"  {GRAY}press CTRL+C to stop{RESET}")
    print()


if __name__ == "__main__":
    ensure_folders()
    port = 5092
    
    # تجنب التكرار أثناء الـ reload
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print_banner(port)
        
    app.run(debug=True, host="0.0.0.0", port=port)