/**
 * Coverage Threshold Checker for PRSM Smart Contracts
 * 
 * Enforces 80% minimum coverage for security-critical smart contracts
 * Addresses cold developer audit feedback on consistent test coverage measurement
 */

const fs = require('fs');
const path = require('path');

// Coverage thresholds for different contract types
const COVERAGE_THRESHOLDS = {
  // Critical Security Components: 80%
  'FTNSTokenSimple.sol': 80,
  'FTNSToken.sol': 80,
  'FTNSGovernance.sol': 80,
  'FTNSMarketplace.sol': 80,
  
  // Default threshold for other contracts
  'default': 60
};

function checkCoverage() {
  console.log('🔍 Checking Smart Contract Test Coverage...');
  console.log('=' * 50);
  
  try {
    // Read coverage report from solidity-coverage
    const coverageReportPath = path.join(__dirname, '..', 'coverage', 'coverage.json');
    
    if (!fs.existsSync(coverageReportPath)) {
      console.error('❌ Coverage report not found. Run `npm run test:coverage` first.');
      process.exit(1);
    }
    
    const coverageData = JSON.parse(fs.readFileSync(coverageReportPath, 'utf8'));
    
    let overallPassed = true;
    let totalContracts = 0;
    let passedContracts = 0;
    
    console.log('\\n📊 Contract Coverage Results:');
    console.log('-'.repeat(60));
    
    // Check coverage for each contract
    for (const [contractPath, contractData] of Object.entries(coverageData)) {
      const contractName = path.basename(contractPath);
      const threshold = COVERAGE_THRESHOLDS[contractName] || COVERAGE_THRESHOLDS.default;
      
      // Calculate line coverage percentage
      const lineData = contractData.l || {};
      const totalLines = Object.keys(lineData).length;
      const coveredLines = Object.values(lineData).filter(count => count > 0).length;
      const coveragePercentage = totalLines > 0 ? (coveredLines / totalLines) * 100 : 0;
      
      totalContracts++;
      const passed = coveragePercentage >= threshold;
      
      if (passed) {
        passedContracts++;
      } else {
        overallPassed = false;
      }
      
      const statusEmoji = passed ? '✅' : '❌';
      const gap = passed ? '' : ` (${(threshold - coveragePercentage).toFixed(1)}% gap)`;
      
      console.log(`${statusEmoji} ${contractName.padEnd(25)} ${coveragePercentage.toFixed(1)}%/${threshold}%${gap}`);
      
      // Show detailed metrics for failed contracts
      if (!passed) {
        console.log(`   📍 Lines: ${coveredLines}/${totalLines} covered`);
        
        // Calculate other metrics if available
        if (contractData.f) {
          const functions = contractData.f;
          const totalFunctions = Object.keys(functions).length;
          const coveredFunctions = Object.values(functions).filter(count => count > 0).length;
          const functionCoverage = totalFunctions > 0 ? (coveredFunctions / totalFunctions) * 100 : 0;
          console.log(`   📍 Functions: ${coveredFunctions}/${totalFunctions} (${functionCoverage.toFixed(1)}%)`);
        }
        
        if (contractData.b) {
          const branches = contractData.b;
          const totalBranches = Object.keys(branches).length;
          const coveredBranches = Object.values(branches).filter(branch => 
            Array.isArray(branch) && branch.some(count => count > 0)
          ).length;
          const branchCoverage = totalBranches > 0 ? (coveredBranches / totalBranches) * 100 : 0;
          console.log(`   📍 Branches: ${coveredBranches}/${totalBranches} (${branchCoverage.toFixed(1)}%)`);
        }
      }
    }
    
    console.log('-'.repeat(60));
    console.log(`📈 Overall Results: ${passedContracts}/${totalContracts} contracts passed`);
    
    if (overallPassed) {
      console.log('\\n🎉 All contracts meet coverage requirements!');
      console.log('✅ Smart contract testing meets production security standards');
    } else {
      console.log('\\n❌ Some contracts do not meet coverage requirements');
      console.log('💡 Add more test cases to increase coverage');
      
      // Generate specific recommendations
      console.log('\\n📋 Recommendations:');
      for (const [contractPath, contractData] of Object.entries(coverageData)) {
        const contractName = path.basename(contractPath);
        const threshold = COVERAGE_THRESHOLDS[contractName] || COVERAGE_THRESHOLDS.default;
        
        const lineData = contractData.l || {};
        const totalLines = Object.keys(lineData).length;
        const coveredLines = Object.values(lineData).filter(count => count > 0).length;
        const coveragePercentage = totalLines > 0 ? (coveredLines / totalLines) * 100 : 0;
        
        if (coveragePercentage < threshold) {
          const missingLines = Math.ceil((threshold - coveragePercentage) * totalLines / 100);
          console.log(`  • ${contractName}: Add ~${missingLines} more lines of test coverage`);
          
          // Find uncovered lines
          const uncoveredLines = [];
          for (const [lineNum, count] of Object.entries(lineData)) {
            if (count === 0) {
              uncoveredLines.push(lineNum);
            }
          }
          
          if (uncoveredLines.length > 0) {
            const sampleUncovered = uncoveredLines.slice(0, 5);
            console.log(`    Uncovered lines: ${sampleUncovered.join(', ')}${uncoveredLines.length > 5 ? '...' : ''}`);
          }
        }
      }
    }
    
    // Save coverage summary for CI
    const coverageSummary = {
      timestamp: new Date().toISOString(),
      totalContracts,
      passedContracts,
      overallPassed,
      thresholds: COVERAGE_THRESHOLDS
    };
    
    const summaryPath = path.join(__dirname, '..', 'coverage-summary.json');
    fs.writeFileSync(summaryPath, JSON.stringify(coverageSummary, null, 2));
    
    console.log(`\\n💾 Coverage summary saved to: coverage-summary.json`);
    
    // Exit with appropriate code
    process.exit(overallPassed ? 0 : 1);
    
  } catch (error) {
    console.error('❌ Error checking coverage:', error.message);
    process.exit(1);
  }
}

// Helper function to format percentage with color
function formatPercentage(percentage, threshold) {
  const passed = percentage >= threshold;
  const color = passed ? '\\x1b[32m' : '\\x1b[31m'; // Green or red
  const reset = '\\x1b[0m';
  return `${color}${percentage.toFixed(1)}%${reset}`;
}

// Run coverage check
if (require.main === module) {
  checkCoverage();
}

module.exports = { checkCoverage, COVERAGE_THRESHOLDS };