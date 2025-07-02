package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/PRSM-AI/prsm-go-sdk/client"
	"github.com/PRSM-AI/prsm-go-sdk/types"
)

func main() {
	// Get API key from environment variable
	apiKey := os.Getenv("PRSM_API_KEY")
	if apiKey == "" {
		log.Fatal("PRSM_API_KEY environment variable is required")
	}

	// Create PRSM client with custom configuration
	config := client.DefaultConfig()
	config.APIKey = apiKey
	config.Timeout = 120 // 2 minutes timeout for complex queries
	prsmClient := client.NewWithConfig(config)

	// Create context
	ctx := context.Background()

	fmt.Println("PRSM Advanced Features Examples")
	fmt.Println("===============================")

	// 1. Safety monitoring
	fmt.Println("1. Safety Status Check:")
	safetyStatus, err := prsmClient.GetSafetyStatus(ctx)
	if err != nil {
		log.Printf("Failed to get safety status: %v", err)
	} else {
		fmt.Printf("Overall Safety Status: %s\n", safetyStatus.OverallStatus)
		fmt.Printf("Active Monitors: %d\n", safetyStatus.ActiveMonitors)
		fmt.Printf("Threats Detected: %d\n", safetyStatus.ThreatsDetected)
		fmt.Printf("Circuit Breakers Triggered: %d\n", safetyStatus.CircuitBreakersTriggered)
		fmt.Printf("Network Health: %.2f%%\n", safetyStatus.NetworkHealth*100)
		fmt.Printf("Last Assessment: %s\n", safetyStatus.LastAssessment.Format("2006-01-02 15:04:05"))
	}

	// 2. Complex query with system prompt and tools
	fmt.Println("\n2. Advanced Query with System Prompt:")
	systemPrompt := "You are a helpful research assistant specializing in AI and machine learning. Always provide evidence-based responses and cite sources when possible."
	
	advancedQuery := &types.QueryRequest{
		Prompt:       "Compare the effectiveness of transformer architectures versus CNNs for image classification tasks.",
		MaxTokens:    800,
		Temperature:  0.3, // Lower temperature for more focused response
		SystemPrompt: &systemPrompt,
		Context: map[string]interface{}{
			"domain":       "computer_vision",
			"academic_level": "graduate",
			"require_citations": true,
		},
		Tools:       []string{"web_search", "arxiv_search"},
		SafetyLevel: types.SafetyLevelHigh,
	}

	response, err := prsmClient.Query(ctx, advancedQuery)
	if err != nil {
		log.Printf("Advanced query failed: %v", err)
	} else {
		fmt.Printf("Response: %s\n", response.Content)
		fmt.Printf("Safety validated: %s\n", response.SafetyStatus)
		
		if len(response.ReasoningTrace) > 0 {
			fmt.Println("\nReasoning Steps:")
			for i, step := range response.ReasoningTrace {
				fmt.Printf("  %d. %s\n", i+1, step)
			}
		}
	}

	// 3. Tool execution example
	fmt.Println("\n3. Direct Tool Execution:")
	toolReq := &types.ToolExecutionRequest{
		ToolName: "web_search",
		Parameters: map[string]interface{}{
			"query":     "latest developments in quantum computing 2024",
			"max_results": 5,
		},
		Context: map[string]interface{}{
			"search_type": "academic",
		},
		SafetyLevel: types.SafetyLevelModerate,
	}

	toolResponse, err := prsmClient.Tools.Execute(ctx, toolReq)
	if err != nil {
		log.Printf("Tool execution failed: %v", err)
	} else {
		fmt.Printf("Tool execution successful: %t\n", toolResponse.Success)
		fmt.Printf("Execution time: %.2f seconds\n", toolResponse.ExecutionTime)
		fmt.Printf("FTNS cost: %.4f\n", toolResponse.FTNSCost)
		fmt.Printf("Result: %v\n", toolResponse.Result)
	}

	// 4. Health check and diagnostics
	fmt.Println("\n4. System Health Check:")
	health, err := prsmClient.HealthCheck(ctx)
	if err != nil {
		log.Printf("Health check failed: %v", err)
	} else {
		fmt.Printf("API Status: %v\n", health["status"])
		fmt.Printf("Response Time: %v ms\n", health["response_time"])
		fmt.Printf("Active Models: %v\n", health["active_models"])
		fmt.Printf("System Load: %v\n", health["system_load"])
	}

	// 5. Model performance analysis
	fmt.Println("\n5. Model Performance Analysis:")
	models, err := prsmClient.ListAvailableModels(ctx)
	if err != nil {
		log.Printf("Failed to list models: %v", err)
	} else if len(models) > 0 {
		// Analyze first available model
		model := models[0]
		analysis, err := prsmClient.Marketplace.AnalyzeModelPerformance(ctx, model.ID, "last_30_days")
		if err != nil {
			log.Printf("Failed to analyze model performance: %v", err)
		} else {
			fmt.Printf("Performance analysis for %s:\n", model.Name)
			fmt.Printf("- Average accuracy: %.2f%%\n", analysis.AverageAccuracy*100)
			fmt.Printf("- Average latency: %.2f ms\n", analysis.AverageLatency*1000)
			fmt.Printf("- Reliability score: %.2f/5.0\n", analysis.ReliabilityScore)
			fmt.Printf("- Cost efficiency: %.2f/5.0\n", analysis.CostEfficiency)
		}
	}

	// 6. Error handling demonstration
	fmt.Println("\n6. Error Handling Examples:")
	
	// Example of handling different error types
	invalidQuery := &types.QueryRequest{
		Prompt:      "", // Empty prompt should fail
		MaxTokens:   0,
		Temperature: -1, // Invalid temperature
	}

	_, err = prsmClient.Query(ctx, invalidQuery)
	if err != nil {
		switch e := err.(type) {
		case *types.ValidationError:
			fmt.Printf("Validation error: %s\n", e.Error())
		case *types.AuthenticationError:
			fmt.Printf("Authentication error: %s\n", e.Error())
		case *types.InsufficientFundsError:
			fmt.Printf("Insufficient funds: %s\n", e.Error())
		case *types.RateLimitError:
			fmt.Printf("Rate limit error: %s\n", e.Error())
		default:
			fmt.Printf("Other error: %s\n", err.Error())
		}
	}

	fmt.Println("\n--- Advanced Features Demo Complete ---")
}