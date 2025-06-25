package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/prsm-ai/prsm-go"
)

func main() {
	// Initialize client with API key from environment
	apiKey := os.Getenv("PRSM_API_KEY")
	if apiKey == "" {
		log.Fatal("PRSM_API_KEY environment variable is required")
	}

	client := prsm.NewClient(apiKey)
	ctx := context.Background()

	// Example 1: Basic model inference
	fmt.Println("=== Basic Model Inference ===")
	result, err := client.Models.Infer(ctx, prsm.InferRequest{
		Model:     "gpt-4",
		Prompt:    "Explain distributed AI in one sentence.",
		MaxTokens: 100,
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Response: %s\n\n", result.Content)
	}

	// Example 2: Cost-optimized routing
	fmt.Println("=== Cost-Optimized Routing ===")
	optimized, err := client.CostOptimization.OptimizeRequest(ctx, prsm.OptimizeRequest{
		Prompt: "Generate a Python function to calculate fibonacci numbers",
		Constraints: prsm.OptimizeConstraints{
			MaxCost:    0.02,
			MinQuality: 0.8,
		},
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Selected Model: %s\n", optimized.SelectedModel)
		fmt.Printf("Estimated Cost: $%.4f\n", optimized.EstimatedCost)
		fmt.Printf("Cost Savings: $%.4f\n\n", optimized.CostSavings)
	}

	// Example 3: List available models
	fmt.Println("=== Available Models ===")
	models, err := client.Models.List(ctx, prsm.ListModelsRequest{
		Limit: 5,
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		for _, model := range models.Models {
			fmt.Printf("- %s: %s\n", model.ID, model.Description)
		}
		fmt.Println()
	}

	// Example 4: Get budget status
	fmt.Println("=== Budget Status ===")
	budget, err := client.CostOptimization.GetBudget(ctx)
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Total Budget: $%.2f\n", budget.TotalBudget)
		fmt.Printf("Spent: $%.2f\n", budget.Spent)
		fmt.Printf("Remaining: $%.2f\n", budget.Remaining)
	}
}