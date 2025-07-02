package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/PRSM-AI/prsm-go-sdk/client"
)

func main() {
	// Get API key from environment variable
	apiKey := os.Getenv("PRSM_API_KEY")
	if apiKey == "" {
		log.Fatal("PRSM_API_KEY environment variable is required")
	}

	// Create PRSM client
	prsmClient := client.New(apiKey)

	// Create context
	ctx := context.Background()

	// Example prompts of varying complexity
	prompts := []string{
		"What is 2 + 2?",
		"Explain the concept of neural networks in detail.",
		"Write a comprehensive analysis of quantum computing and its potential impact on cryptography, including mathematical foundations and practical applications.",
	}

	fmt.Println("Cost Estimation Examples")
	fmt.Println("========================")

	for i, prompt := range prompts {
		fmt.Printf("\nExample %d: %s\n", i+1, prompt)
		
		// Estimate cost without specifying model (uses default)
		cost, err := prsmClient.EstimateCost(ctx, prompt, nil)
		if err != nil {
			log.Printf("Failed to estimate cost for example %d: %v", i+1, err)
			continue
		}

		fmt.Printf("Estimated FTNS Cost: %.4f\n", cost)
		
		// Check if we have enough balance before executing
		balance, err := prsmClient.FTNS.GetBalance(ctx)
		if err != nil {
			log.Printf("Failed to get balance: %v", err)
			continue
		}

		if balance.AvailableBalance >= cost {
			fmt.Printf("✓ Sufficient balance (Available: %.4f)\n", balance.AvailableBalance)
		} else {
			fmt.Printf("✗ Insufficient balance (Available: %.4f, Required: %.4f)\n", 
				balance.AvailableBalance, cost)
		}
	}

	// Example with specific model
	fmt.Printf("\n--- Model-Specific Cost Estimation ---\n")
	
	// List available models
	models, err := prsmClient.ListAvailableModels(ctx)
	if err != nil {
		log.Printf("Failed to list models: %v", err)
		return
	}

	if len(models) > 0 {
		model := models[0]
		fmt.Printf("Using model: %s (%s)\n", model.Name, model.ID)
		
		cost, err := prsmClient.EstimateCost(ctx, "Explain artificial intelligence", &model.ID)
		if err != nil {
			log.Printf("Failed to estimate cost with specific model: %v", err)
			return
		}

		fmt.Printf("Estimated cost with %s: %.4f FTNS\n", model.Name, cost)
		fmt.Printf("Cost per token: %.6f FTNS\n", model.CostPerToken)
	}
}