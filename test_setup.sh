#!/usr/bin/env bash
# Gen AI Framework - test setup script (uses uv)
# Run from project root: ./test_setup.sh [pytest args...]
# Syncs deps with uv, creates test fixtures, runs pytest.

set -e
cd "$(dirname "$0")"

echo "Setting up test environment (uv)..."

if ! command -v uv &> /dev/null; then
    echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Syncing dependencies..."
uv sync

# Create test fixture and output directories
mkdir -p tests/fixtures/data/batch/bills output/batch
echo "Created tests/fixtures/data/batch/, tests/fixtures/data/batch/bills/, output/batch/."

# Seed minimal fixtures if missing (so batch test can run without cloning fixtures)
BATCH_DIR="tests/fixtures/data/batch"
if [ ! -f "$BATCH_DIR/policy.txt" ]; then
    echo "Creating minimal $BATCH_DIR/policy.txt..."
    cat > "$BATCH_DIR/policy.txt" << 'EOF'
Admin expense policy for cab and meals:
- Cab: Max 50 USD per trip. Only Uber, Lyft, or licensed taxi.
- Meals: Max 30 USD per meal. Must have date and receipt.
- Reject if amount is missing or over limit.
EOF
fi
if [ ! -f "$BATCH_DIR/bills/cab_trip_1.txt" ] && [ ! -f "$BATCH_DIR/bills/meal_lunch.txt" ]; then
    echo "Creating minimal sample bill in $BATCH_DIR/bills/..."
    echo "UBER receipt
Date: 2024-01-15
Trip: Downtown to Airport
Amount: 45.00 USD
Vendor: Uber" > "$BATCH_DIR/bills/cab_trip_1.txt"
fi

echo ""
echo "Test setup complete. Running pytest..."
echo ""

# Run pytest via uv; pass any extra args (e.g. -v, -k test_rag, --tb=short)
exec uv run pytest tests/ "$@"
