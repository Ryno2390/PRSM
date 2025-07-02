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

	// Create PRSM client
	prsmClient := client.New(apiKey)

	// Create context with timeout
	ctx := context.Background()

	// Prepare query request
	queryReq := &types.QueryRequest{
		Prompt:      "What are the key principles of machine learning?",
		MaxTokens:   500,
		Temperature: 0.7,
		SafetyLevel: types.SafetyLevelModerate,
	}

	fmt.Println("Executing query...")

	// Execute query
	response, err := prsmClient.Query(ctx, queryReq)
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}

	// Display results
	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Model ID: %s\n", response.ModelID)
	fmt.Printf("Provider: %s\n", response.Provider)
	fmt.Printf("Execution Time: %.2f seconds\n", response.ExecutionTime)
	fmt.Printf("FTNS Cost: %.4f\n", response.FTNSCost)
	fmt.Printf("Safety Status: %s\n", response.SafetyStatus)
	fmt.Printf("Request ID: %s\n", response.RequestID)

	if len(response.ReasoningTrace) > 0 {
		fmt.Println("\nReasoning Trace:")
		for i, step := range response.ReasoningTrace {
			fmt.Printf("  %d. %s\n", i+1, step)
		}
	}
}