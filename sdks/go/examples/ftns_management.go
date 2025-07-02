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

	fmt.Println("FTNS Token Management Examples")
	fmt.Println("==============================")

	// 1. Check current balance
	fmt.Println("1. Current FTNS Balance:")
	balance, err := prsmClient.FTNS.GetBalance(ctx)
	if err != nil {
		log.Fatalf("Failed to get balance: %v", err)
	}

	fmt.Printf("Total Balance: %.4f FTNS\n", balance.TotalBalance)
	fmt.Printf("Available Balance: %.4f FTNS\n", balance.AvailableBalance)
	fmt.Printf("Reserved Balance: %.4f FTNS\n", balance.ReservedBalance)
	fmt.Printf("Earned Today: %.4f FTNS\n", balance.EarnedToday)
	fmt.Printf("Spent Today: %.4f FTNS\n", balance.SpentToday)
	fmt.Printf("Last Updated: %s\n", balance.LastUpdated.Format("2006-01-02 15:04:05"))

	// 2. Get transaction history
	fmt.Println("\n2. Recent Transaction History:")
	transactions, err := prsmClient.FTNS.GetTransactionHistory(ctx, 10)
	if err != nil {
		log.Printf("Failed to get transaction history: %v", err)
	} else {
		for _, tx := range transactions {
			fmt.Printf("- %s: %+.4f FTNS (%s) - %s\n", 
				tx.Timestamp.Format("Jan 02 15:04"), 
				tx.Amount, 
				tx.Type, 
				tx.Description)
		}
	}

	// 3. Transfer tokens (example - requires recipient)
	fmt.Println("\n3. Token Transfer Example:")
	// Uncomment and modify the recipient address for actual transfer
	/*
	recipient := "user123"  // Replace with actual recipient
	transferAmount := 1.0
	
	transferResult, err := prsmClient.FTNS.Transfer(ctx, recipient, transferAmount, "Test transfer")
	if err != nil {
		log.Printf("Failed to transfer tokens: %v", err)
	} else {
		fmt.Printf("Transfer successful! Transaction ID: %s\n", transferResult.TransactionID)
	}
	*/
	fmt.Println("(Transfer example commented out - modify recipient address to test)")

	// 4. Estimate earnings for content upload
	fmt.Println("\n4. Estimated Earnings for Content Upload:")
	contentSize := int64(1024 * 1024) // 1MB
	contentType := "research_paper"
	
	estimatedEarnings, err := prsmClient.FTNS.EstimateContentEarnings(ctx, contentSize, contentType)
	if err != nil {
		log.Printf("Failed to estimate earnings: %v", err)
	} else {
		fmt.Printf("Estimated earnings for 1MB %s: %.4f FTNS\n", contentType, estimatedEarnings)
	}

	// 5. Check staking opportunities
	fmt.Println("\n5. Staking Opportunities:")
	stakingOptions, err := prsmClient.FTNS.GetStakingOptions(ctx)
	if err != nil {
		log.Printf("Failed to get staking options: %v", err)
	} else {
		for _, option := range stakingOptions {
			fmt.Printf("- %s: %.2f%% APY (Min: %.2f FTNS, Lock: %d days)\n",
				option.Name, option.APY*100, option.MinimumStake, option.LockPeriodDays)
		}
	}

	// 6. Get governance voting power
	fmt.Println("\n6. Governance Voting Power:")
	votingPower, err := prsmClient.FTNS.GetVotingPower(ctx)
	if err != nil {
		log.Printf("Failed to get voting power: %v", err)
	} else {
		fmt.Printf("Current voting power: %.2f\n", votingPower.Power)
		fmt.Printf("Based on holdings: %.4f FTNS\n", votingPower.BaseHoldings)
		fmt.Printf("Staking multiplier: %.2fx\n", votingPower.StakingMultiplier)
	}

	// 7. Monitor network revenue
	fmt.Println("\n7. Network Revenue Statistics:")
	revenueStats, err := prsmClient.FTNS.GetNetworkRevenue(ctx)
	if err != nil {
		log.Printf("Failed to get network revenue: %v", err)
	} else {
		fmt.Printf("Total network revenue (24h): %.2f FTNS\n", revenueStats.Revenue24h)
		fmt.Printf("Your share (estimated): %.4f FTNS\n", revenueStats.YourShare)
		fmt.Printf("Next dividend date: %s\n", revenueStats.NextDividendDate.Format("Jan 02, 2006"))
	}
}