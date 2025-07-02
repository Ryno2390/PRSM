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

	// Create context
	ctx := context.Background()

	fmt.Println("PRSM Model Marketplace Examples")
	fmt.Println("===============================")

	// 1. List all available models
	fmt.Println("\n1. Available Models:")
	models, err := prsmClient.ListAvailableModels(ctx)
	if err != nil {
		log.Fatalf("Failed to list models: %v", err)
	}

	for _, model := range models {
		fmt.Printf("- %s (%s)\n", model.Name, model.ID)
		fmt.Printf("  Provider: %s\n", model.Provider)
		fmt.Printf("  Description: %s\n", model.Description)
		fmt.Printf("  Cost per token: %.6f FTNS\n", model.CostPerToken)
		fmt.Printf("  Max tokens: %d\n", model.MaxTokens)
		fmt.Printf("  Performance: %.2f/5.0\n", model.PerformanceRating)
		fmt.Printf("  Safety: %.2f/5.0\n", model.SafetyRating)
		fmt.Printf("  Available: %t\n\n", model.IsAvailable)
	}

	// 2. Search for specific models
	fmt.Println("2. Searching for Language Models:")
	maxCost := 0.01
	minPerformance := 4.0
	
	searchQuery := &types.MarketplaceQuery{
		Query:          "language model",
		MaxCost:        &maxCost,
		MinPerformance: &minPerformance,
		Capabilities:   []string{"text-generation", "reasoning"},
		Limit:          5,
	}

	searchResults, err := prsmClient.Marketplace.SearchModels(ctx, searchQuery)
	if err != nil {
		log.Printf("Failed to search models: %v", err)
	} else {
		for _, model := range searchResults {
			fmt.Printf("- %s (Performance: %.2f, Cost: %.6f)\n", 
				model.Name, model.PerformanceRating, model.CostPerToken)
		}
	}

	// 3. Find best model for specific task
	fmt.Println("\n3. Finding Best Model for Code Generation:")
	codeGenModels, err := prsmClient.Marketplace.FindBestModel(ctx, "code generation", []string{"python", "javascript"})
	if err != nil {
		log.Printf("Failed to find best model: %v", err)
	} else if codeGenModels != nil {
		fmt.Printf("Recommended: %s\n", codeGenModels.Name)
		fmt.Printf("Capabilities: %v\n", codeGenModels.Capabilities)
	}

	// 4. Compare models
	fmt.Println("\n4. Model Comparison:")
	if len(models) >= 2 {
		comparison, err := prsmClient.Marketplace.CompareModels(ctx, models[0].ID, models[1].ID)
		if err != nil {
			log.Printf("Failed to compare models: %v", err)
		} else {
			fmt.Printf("Comparing %s vs %s:\n", models[0].Name, models[1].Name)
			fmt.Printf("Performance: %.2f vs %.2f\n", 
				comparison.Model1Performance, comparison.Model2Performance)
			fmt.Printf("Cost efficiency: %.2f vs %.2f\n", 
				comparison.Model1CostEfficiency, comparison.Model2CostEfficiency)
			fmt.Printf("Recommendation: %s\n", comparison.Recommendation)
		}
	}

	// 5. Get model performance metrics
	fmt.Println("\n5. Model Performance Metrics:")
	if len(models) > 0 {
		metrics, err := prsmClient.Marketplace.GetModelMetrics(ctx, models[0].ID)
		if err != nil {
			log.Printf("Failed to get model metrics: %v", err)
		} else {
			fmt.Printf("Metrics for %s:\n", models[0].Name)
			fmt.Printf("- Average response time: %.2f seconds\n", metrics.AverageResponseTime)
			fmt.Printf("- Success rate: %.2f%%\n", metrics.SuccessRate*100)
			fmt.Printf("- User satisfaction: %.2f/5.0\n", metrics.UserSatisfaction)
		}
	}
}