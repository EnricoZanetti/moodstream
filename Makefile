.PHONY: setup run run-headless demo demo-local lint format export-dashboard pipeline-up pipeline-down clean

setup:
	uv sync

run:
	uv run python src/detector.py

run-headless:
	uv run python src/detector.py --no-display

demo:
	uv run python src/detector.py --demo

demo-local:
	uv run python src/detector.py --demo --no-mqtt

lint:
	uv run ruff check .

format:
	uv run ruff format .

export-dashboard:
	@curl -s "http://localhost:3000/api/dashboards/uid/emotion-detection" \
		| python3 -c "import sys,json;d=json.load(sys.stdin)['dashboard'];d.pop('id',None);d.pop('version',None);print(json.dumps(d,indent=2))" \
		> pipeline/grafana/dashboards/emotion-detection.json
	@echo "Dashboard exported to pipeline/grafana/dashboards/emotion-detection.json"

pipeline-up:
	docker compose up -d

pipeline-down:
	docker compose down

clean:
	rm -rf .venv __pycache__ .ruff_cache
