#!/bin/bash

# Settings
REPORT_DIR="results/tests"
METRICS_FILE="$REPORT_DIR/metrics_history.dat"
REPORT_FILE="$REPORT_DIR/metrics_report.txt"
mkdir -p "$REPORT_DIR"

# Function to safely get metrics considering Prometheus format
get_metric() {
    local metric_name=$1
    curl -s localhost:8000 | 
        grep -E "${metric_name} [0-9eE.+-]+" | 
        awk '{print $2}' | 
        sed 's/[eE]+\{0,1\}/*10^/g' | 
        bc -l || echo "0"
}

# Initialize data file
echo "# generation phase best_score avg_score pool_size rl_usage" > "$METRICS_FILE"

# Function to determine evolution phase
get_evolution_phase() {
    local gen=$1
    local best=$2
    local rl=$3

    if (( $(echo "$gen < 5" | bc -l) )); then
        echo "initial_exploration"
    elif (( $(echo "$best < 0.3" | bc -l) )); then
        echo "stagnation_recovery"
    elif (( $(echo "$rl > 0.7" | bc -l) )); then
        echo "rl_domination"
    else
        echo "balanced_evolution"
    fi
}

# Function to generate recommendations
get_recommendations() {
    local phase=$1
    local best=$2
    local avg=$3
    local rl=$4

    case $phase in
        "initial_exploration")
            echo "Recommendations: Increase mutation diversity (add_prob +0.1)"
            echo "Action: Increase probability of add and invert mutations"
            ;;
        "stagnation_recovery")
            echo "Recommendations: Add 10 random strategies to the pool"
            echo "Action: Increase drop_prob by 0.15"
            ;;
        "rl_domination")
            if (( $(echo "$best > 1.0" | bc -l) )); then
                echo "Recommendations: RL agents performing excellently! Export best strategy"
                echo "Action: Save best_strategy.json"
            else
                echo "Recommendations: Balance RL and classical approaches"
                echo "Action: Decrease rl_mutation_prob by 0.2"
            fi
            ;;
        *)
            echo "Recommendations: Continue current evolution process"
            echo "Action: Monitoring without parameter changes"
            ;;
    esac
    
    # Additional analytical recommendations
    if (( $(echo "$avg < 0" | bc -l) )); then
        echo "Critical: Most strategies are unprofitable! Reset pool"
        echo "Emergency action: Initialize 50 new random strategies"
    fi
    
    if (( $(echo "$rl == 1.0" | bc -l) )); then
        echo "Warning: All strategies use RL - diversity reduction"
        echo "Action: Introduce max_rl_usage=0.8 limit"
    fi
}

# Main loop
last_gen=-1
while true; do
    # Get metrics with validation
    GEN=$(get_metric "evolution_generation")
    BEST=$(get_metric "evolution_best_score")
    AVG=$(get_metric "evolution_avg_score")
    POOL=$(get_metric "evolution_pool_size")
    RL=$(get_metric "evolution_rl_usage")
    
    # Data validation
    if ! [[ "$GEN" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Error getting data. Retrying in 10 sec..."
        sleep 10
        continue
    fi
    
    # Update only when generation changes
    if (( $(echo "$GEN > $last_gen" | bc -l) )); then
        last_gen=$GEN
        
        # Calculations
        RL_PERCENT=$(echo "scale=0; $RL * 100" | bc -l)
        PHASE=$(get_evolution_phase "$GEN" "$BEST" "$RL")
        RECOMMENDATIONS=$(get_recommendations "$PHASE" "$BEST" "$AVG" "$RL")
        
        # Write data
        echo "$GEN $PHASE $BEST $AVG $POOL $RL" >> "$METRICS_FILE"
        
        # Generate report
        {
            # Color codes
            RED='\033[0;31m'
            GREEN='\033[0;32m'
            YELLOW='\033[1;33m'
            BLUE='\033[0;34m'
            CYAN='\033[0;36m'
            NC='\033[0m'
            
            # Evolution phase with color
            case $PHASE in
                "initial_exploration") COLOR=$BLUE ;;
                "stagnation_recovery") COLOR=$RED ;;
                "rl_domination") COLOR=$GREEN ;;
                *) COLOR=$YELLOW ;;
            esac
            
            echo -e "=== ${CYAN}Evolution Monitoring${NC} ==="
            echo -e "Time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo -e "Current phase: ${COLOR}$PHASE${NC}"
            echo -e "Generation: ${YELLOW}$GEN${NC}"
            echo ""
            
            # Key metrics
            echo -e "Best score: ${GREEN}$BEST${NC}"
            if (( $(echo "$AVG >= 0" | bc -l) )); then
                echo -e "Average score: ${GREEN}$AVG${NC}"
            else
                echo -e "Average score: ${RED}$AVG${NC}"
            fi
            echo -e "Pool size: $POOL"
            echo -e "RL usage: $RL_PERCENT%"
            echo ""
            
            # Statistics for recent values
            if (( $(wc -l < "$METRICS_FILE") > 2 )); then
                echo -e "${CYAN}Change history:${NC}"
                tail -n 5 "$METRICS_FILE" | awk '{
                    printf "Gen %3d: Best=%-8.2f Avg=%-8.2f RL=%-3.0f%% Phase=%-20s\n", 
                    $1, $3, $4, $6*100, $2
                }'
                echo ""
            fi
            
            # Analysis and recommendations
            echo -e "${CYAN}Analysis and recommendations:${NC}"
            echo -e "$RECOMMENDATIONS"
            echo ""
            
            # Plots
            if (( $(wc -l < "$METRICS_FILE") > 5 )); then
                echo "Metrics trend:"
                gnuplot <<-EOF
                    set terminal dumb size 100,30
                    set multiplot layout 2,1
                    
                    set title "Scores Dynamics"
                    set ylabel "Score"
                    set xlabel "Generation"
                    plot "$METRICS_FILE" using 1:3 with lines title "Best", \
                         "$METRICS_FILE" using 1:4 with lines title "Avg"
                    
                    set title "RL Usage"
                    set ylabel "RL Usage (%)"
                    set yrange [0:100]
                    plot "$METRICS_FILE" using 1:(\$6*100) with lines title "RL %"
                    
                    unset multiplot
EOF
            else
                echo "Collecting data... (need >5 generations)"
            fi
        } > "$REPORT_FILE"
    fi
    
    # Display report
    clear
    cat "$REPORT_FILE"
    
    # Progress waiting
    for i in {1..10}; do
        echo -n "."
        sleep 1
    done
done