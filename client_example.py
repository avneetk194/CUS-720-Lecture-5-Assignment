"""Send an invented transcript to the running redaction service."""

import requests


def main():
    payload = {
        "text": "Maria works in Boston. Email maria@example.com.",
        "types": ["PER", "LOC", "ORG", "MISC"],
    }

    try:
        response = requests.post(
            "http://127.0.0.1:5000/api/redact",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.RequestException as error:
        print(f"Request failed: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
