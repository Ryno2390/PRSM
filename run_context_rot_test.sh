#!/bin/bash
# NWTN Context Rot Test - Background Execution Script
# Runs the full 175+ minute NWTN pipeline with caffeinate to prevent sleep

echo "🚀 Starting NWTN Context Rot Test - Background Execution"
echo "========================================================"

# Set environment variables
export PYTHONPATH="/Users/ryneschultz/Documents/GitHub/PRSM"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Create logs directory if it doesn't exist
mkdir -p logs

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/context_rot_test_${TIMESTAMP}.log"
ERROR_LOG="logs/context_rot_test_${TIMESTAMP}_errors.log"

echo "📝 Logging to: $LOG_FILE"
echo "❌ Errors to: $ERROR_LOG"
echo ""

# The context rot prompt that previously caused freeze
CONTEXT_ROT_PROMPT="What are the most significant advances in AI safety research in the past 2 years, and what key gaps remain? Please provide a comprehensive analysis with detailed technical explanations, covering alignment research, robustness methods, interpretability advances, governance frameworks, and emerging risks. I want an academic paper-length response with extensive technical depth."

echo "🧠 Query: $CONTEXT_ROT_PROMPT"
echo ""
echo "⚡ Expected: 5,040 candidates → 177M+ operations → Academic paper synthesis"
echo "⏱️  Estimated runtime: 175+ minutes"
echo "🛡️  Caffeinate: Prevents system sleep during execution"
echo ""

# Run with caffeinate to prevent system sleep
# -d: prevent display sleep
# -i: prevent idle sleep  
# -m: prevent disk sleep
# -s: prevent system sleep
echo "🚀 Starting NWTN pipeline with full system wake protection..."

caffeinate -dims python3 -c "
import sys
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

import asyncio
from datetime import datetime, timezone
from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.nwtn.config.breakthrough_modes import get_breakthrough_mode, BreakthroughMode

async def run_context_rot_test():
    print('🎯 NWTN Context Rot Test Starting')
    print('=' * 50)
    
    start_time = datetime.now(timezone.utc)
    query = '''$CONTEXT_ROT_PROMPT'''
    
    # Initialize enhanced orchestrator
    orchestrator = EnhancedNWTNOrchestrator()
    await orchestrator.initialize()
    
    print(f'🧠 Query: {query[:100]}...')
    print(f'🚀 Mode: REVOLUTIONARY (5,040 candidates)')
    print(f'⏱️  Start time: {start_time.isoformat()}')
    print('')
    
    try:
        # Run full REVOLUTIONARY pipeline
        breakthrough_config = get_breakthrough_mode(BreakthroughMode.REVOLUTIONARY)
        
        result = await orchestrator.enhanced_process_query(
            query=query,
            breakthrough_config=breakthrough_config,
            include_world_model=True,
            max_candidates=5040  # Full revolutionary scale
        )
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        print('')
        print('🎉 NWTN Context Rot Test COMPLETED')
        print('=' * 50)
        print(f'⏱️  Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)')
        print(f'📊 Result type: {type(result).__name__}')
        
        if hasattr(result, 'final_response') and result.final_response:
            response_length = len(result.final_response)
            print(f'📝 Response length: {response_length:,} characters')
            print(f'📄 Academic paper length: {\"YES\" if response_length > 15000 else \"NO\"} ({response_length} chars)')
            
            # Save the response to file
            output_file = f'context_rot_response_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.txt'
            with open(output_file, 'w') as f:
                f.write(f'NWTN Context Rot Test Response\\n')
                f.write(f'Query: {query}\\n')
                f.write(f'Generated: {end_time.isoformat()}\\n')
                f.write(f'Duration: {duration:.1f} seconds\\n')
                f.write(f'Length: {response_length:,} characters\\n')
                f.write('=' * 80 + '\\n\\n')
                f.write(result.final_response)
            
            print(f'💾 Response saved to: {output_file}')
        
        print('')
        print('✅ Context rot test SUCCESSFUL - no pipeline freeze!')
        return True
        
    except Exception as e:
        print(f'❌ Context rot test FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False

# Run the test
result = asyncio.run(run_context_rot_test())
exit(0 if result else 1)
" 2>&1 | tee "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ NWTN Context Rot Test COMPLETED SUCCESSFULLY"
else
    echo "❌ NWTN Context Rot Test FAILED (exit code: $EXIT_CODE)"
fi

echo "📝 Full log: $LOG_FILE"
echo "⏱️  Completed at: $(date)"
echo "========================================================"

exit $EXIT_CODE