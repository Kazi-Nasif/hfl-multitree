#!/bin/bash
echo "========================================"
echo "Generating Final Project Report"
echo "========================================"

# Run all tests and save outputs
echo ""
echo "Running comprehensive tests..."

python test_complete_system.py > results/final_test_output.txt 2>&1
python experiments/visualize_results.py >> results/final_test_output.txt 2>&1

echo "✓ Tests completed"
echo ""
echo "Generated files:"
echo "  - results/final_test_output.txt"
echo "  - results/plots/*.png (4 figures)"
echo "  - RESULTS_SUMMARY.md"
echo "  - README.md"
echo ""
echo "📊 View your results:"
echo "   cat results/final_test_output.txt"
echo "   ls -lh results/plots/"
echo ""
echo "🎉 Project complete!"
