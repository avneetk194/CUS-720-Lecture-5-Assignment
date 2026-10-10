
from flask import Flask, jsonify, render_template, request

MODEL_NAME = "dslim/bert-base-NER"
ALLOWED_TYPES = {"PER", "LOC", "ORG", "MISC"}


def create_app(redactor=None):
    app = Flask(__name__)

    app.config["MODEL_NAME"] = MODEL_NAME
    app.config["REDACTION_THRESHOLD"] = 0.80

    if redactor is None:
        from redactor import Redactor
        redactor = Redactor()

    app.config["REDACTOR"] = redactor

    def validate_input(data):
        if not isinstance(data, dict):
            return None, None, "Request body must be JSON."

        text = data.get("text")

        if not isinstance(text, str):
            return None, None, "Text must be a string."

        if not 1 <= len(text) <= 10000:
            return None, None, "Text must be 1 to 10,000 characters."

        types = data.get("types")

        if types is not None:
            if not isinstance(types, list):
                return None, None, "Types must be a list."

            if any(not isinstance(t, str) for t in types):
                return None, None, "Invalid entity type."

            if not set(types).issubset(ALLOWED_TYPES):
                return None, None, "Invalid entity type."

        return text, types, None

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "GET":
            return render_template("index.html")

        text = request.form.get("text", "")
        types = request.form.getlist("types")

        text, types, error = validate_input(
            {"text": text, "types": types}
        )

        if error:
            return render_template(
                "index.html",
                error=error,
                text=request.form.get("text", ""),
                selected_types=request.form.getlist("types"),
            ), 400

        result = app.config["REDACTOR"].redact(text, types)

        return render_template(
            "index.html",
            result=result,
            text=text,
            selected_types=types,
        )

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model": app.config["MODEL_NAME"],
        })

    @app.route("/api/redact", methods=["POST"])
    def redact():
        data = request.get_json(silent=True)
        text, types, error = validate_input(data)

        if error:
            return jsonify({"error": error}), 400

        result = app.config["REDACTOR"].redact(text, types)
        return jsonify(result)

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found"}), 404

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
