"""Run functional API test scenario from scenario.json."""

import json
import sys
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIO_PATH = BASE_DIR / "scenario.json"


def run_scenario() -> None:
    """Execute API functional test scenario."""
    if not SCENARIO_PATH.exists():
        raise FileNotFoundError(f"Scenario file not found: {SCENARIO_PATH}")

    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    base_url = scenario["base_url"]
    steps = scenario["steps"]

    for step in steps:
        method = step["method"].upper()
        url = f"{base_url}{step['endpoint']}"
        payload = step.get("json")

        response = requests.request(method=method, url=url, json=payload, timeout=30)

        if response.status_code != step["expected_status"]:
            raise AssertionError(
                f"Step '{step['name']}' failed: "
                f"expected status {step['expected_status']}, got {response.status_code}. "
                f"Response body: {response.text}"
            )

        response_json = response.json()

        expected_body_contains = step.get("expected_body_contains", {})
        for key, value in expected_body_contains.items():
            if response_json.get(key) != value:
                raise AssertionError(
                    f"Step '{step['name']}' failed: expected body field "
                    f"'{key}'={value}, got {response_json.get(key)}"
                )

        expected_body_keys = step.get("expected_body_keys", [])
        for key in expected_body_keys:
            if key not in response_json:
                raise AssertionError(
                    f"Step '{step['name']}' failed: missing key '{key}' in response"
                )

        print(f"[OK] {step['name']}")

    print("Scenario completed successfully.")


if __name__ == "__main__":
    try:
        run_scenario()
    except (AssertionError, requests.RequestException, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
