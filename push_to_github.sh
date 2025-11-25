#!/bin/bash
echo "=========================================="
echo "Pushing HFL-MultiTree to GitHub"
echo "=========================================="
echo ""
echo "GitHub Username: Kazi-Nasif"
echo "Repository: hfl-multitree"
echo ""

# Check if already committed
if git rev-parse HEAD >/dev/null 2>&1; then
    echo "✓ Git repository already initialized"
else
    echo "Initializing git repository..."
    git init
fi

# Add all files
echo ""
echo "Adding files..."
git add .

# Create commit
echo ""
echo "Creating commit..."
git commit -m "Initial commit: Complete HFL+MultiTree implementation

Project Achievements:
- Implemented MultiTree all-reduce with O(log n) complexity
- Integrated AHFLP adaptive aggregation optimization  
- Achieved 80% reduction in training time and energy (exceeded 40-60% target)
- Comprehensive simulation framework with SimPy
- Complete documentation and visualizations
- 4 publication-quality figures

Technical Details:
- 64 spanning trees with BFS construction
- Discrete-event simulation with link contention
- Resource-constrained optimization (time, energy, bandwidth)
- Baseline comparison with Ring all-reduce

Course: CS 8125 Advanced Cloud Computing
Institution: Georgia Institute of Technology"

# Set main branch
echo ""
echo "Setting main branch..."
git branch -M main

# Add remote
echo ""
echo "Adding remote repository..."
if git remote | grep -q origin; then
    echo "Remote 'origin' already exists, updating URL..."
    git remote set-url origin https://github.com/Kazi-Nasif/hfl-multitree.git
else
    git remote add origin https://github.com/Kazi-Nasif/hfl-multitree.git
fi

echo ""
echo "=========================================="
echo "Ready to push!"
echo "=========================================="
echo ""
echo "IMPORTANT: Before running 'git push', make sure you have:"
echo ""
echo "1. Created the repository on GitHub:"
echo "   → Go to: https://github.com/new"
echo "   → Repository name: hfl-multitree"
echo "   → Description: Hierarchical Federated Learning with MultiTree All-Reduce - CS 8125 Project"
echo "   → Make it Public (recommended for portfolio)"
echo "   → DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. After creating the repository, run:"
echo "   git push -u origin main"
echo ""
echo "If you need to authenticate, GitHub will prompt you."
echo "Use a Personal Access Token (not password) if prompted."
echo ""
echo "To create a token:"
echo "   → https://github.com/settings/tokens"
echo "   → Generate new token (classic)"
echo "   → Select 'repo' scope"
echo "   → Copy and use as password when pushing"
echo ""
echo "=========================================="
