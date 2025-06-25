/**
 * PRSM React Web3 Integration Hooks
 * =================================
 * 
 * Custom React hooks for integrating PRSM Web3 functionality
 * into React applications with minimal setup.
 */

import { useState, useEffect, useCallback, useContext, createContext } from 'react';
import { ethers } from 'ethers';

// Contract ABIs (simplified for example)
const FTNS_TOKEN_ABI = [
  "function balanceOf(address owner) view returns (uint256)",
  "function transfer(address to, uint256 amount) returns (bool)",
  "function approve(address spender, uint256 amount) returns (bool)",
  "function getLiquidBalance(address user) view returns (uint256)",
  "function getContextAllocation(address user, string context) view returns (uint256)",
  "function stakeTokens(uint256 amount, string context)",
  "function unstakeTokens(uint256 stakeIndex)",
  "function lockTokens(uint256 amount, uint256 duration, string context)",
  "function unlockTokens(uint256 lockIndex)",
  "event Transfer(address indexed from, address indexed to, uint256 value)",
  "event TokensStaked(address indexed user, uint256 amount, string context)",
  "event TokensLocked(address indexed user, uint256 amount, uint256 unlockTime, string context)"
];

const FTNS_MARKETPLACE_ABI = [
  "function listService(string context, string name, string description, uint8 serviceType, uint256 price, uint256 minStake) returns (uint256)",
  "function purchaseService(uint256 serviceId) returns (uint256)",
  "function getService(uint256 serviceId) view returns (tuple)",
  "function getServicesByContext(string context) view returns (uint256[])",
  "function submitReview(uint256 serviceId, uint8 rating, string comment)",
  "event ServiceListed(uint256 indexed serviceId, address indexed provider, string context, uint256 price, string serviceType)",
  "event ServicePurchased(uint256 indexed serviceId, address indexed buyer, address indexed provider, uint256 amount, string context)"
];

// Contract addresses (replace with deployed addresses)
const CONTRACT_ADDRESSES = {
  polygon: {
    ftnsToken: "0x...", // Replace with actual address
    marketplace: "0x...", // Replace with actual address
    governance: "0x...", // Replace with actual address
  },
  mumbai: {
    ftnsToken: "0x...", // Replace with testnet address
    marketplace: "0x...", // Replace with testnet address
    governance: "0x...", // Replace with testnet address
  }
};

// Web3 Context
const Web3Context = createContext();

export const Web3Provider = ({ children }) => {
  const [provider, setProvider] = useState(null);
  const [signer, setSigner] = useState(null);
  const [account, setAccount] = useState(null);
  const [network, setNetwork] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);

  const connectWallet = useCallback(async () => {
    try {
      if (!window.ethereum) {
        throw new Error('MetaMask not installed');
      }

      const provider = new ethers.BrowserProvider(window.ethereum);
      await provider.send("eth_requestAccounts", []);
      
      const signer = await provider.getSigner();
      const account = await signer.getAddress();
      const network = await provider.getNetwork();

      setProvider(provider);
      setSigner(signer);
      setAccount(account);
      setNetwork(network);
      setIsConnected(true);
      setError(null);

      // Store connection state
      localStorage.setItem('web3Connected', 'true');

    } catch (err) {
      setError(err.message);
      console.error('Failed to connect wallet:', err);
    }
  }, []);

  const disconnectWallet = useCallback(() => {
    setProvider(null);
    setSigner(null);
    setAccount(null);
    setNetwork(null);
    setIsConnected(false);
    setError(null);
    localStorage.removeItem('web3Connected');
  }, []);

  // Auto-connect on load if previously connected
  useEffect(() => {
    if (localStorage.getItem('web3Connected') === 'true') {
      connectWallet();
    }
  }, [connectWallet]);

  // Listen for account changes
  useEffect(() => {
    if (window.ethereum) {
      window.ethereum.on('accountsChanged', (accounts) => {
        if (accounts.length === 0) {
          disconnectWallet();
        } else {
          connectWallet();
        }
      });

      window.ethereum.on('chainChanged', () => {
        window.location.reload();
      });
    }

    return () => {
      if (window.ethereum) {
        window.ethereum.removeAllListeners('accountsChanged');
        window.ethereum.removeAllListeners('chainChanged');
      }
    };
  }, [connectWallet, disconnectWallet]);

  const value = {
    provider,
    signer,
    account,
    network,
    isConnected,
    error,
    connectWallet,
    disconnectWallet
  };

  return (
    <Web3Context.Provider value={value}>
      {children}
    </Web3Context.Provider>
  );
};

export const useWeb3 = () => {
  const context = useContext(Web3Context);
  if (!context) {
    throw new Error('useWeb3 must be used within a Web3Provider');
  }
  return context;
};

// FTNS Token Hook
export const useFTNSToken = () => {
  const { signer, network } = useWeb3();
  const [contract, setContract] = useState(null);
  const [balance, setBalance] = useState('0');
  const [liquidBalance, setLiquidBalance] = useState('0');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (signer && network) {
      const networkName = network.chainId === 137 ? 'polygon' : 'mumbai';
      const address = CONTRACT_ADDRESSES[networkName]?.ftnsToken;
      
      if (address) {
        const tokenContract = new ethers.Contract(address, FTNS_TOKEN_ABI, signer);
        setContract(tokenContract);
      }
    }
  }, [signer, network]);

  const loadBalance = useCallback(async (userAddress) => {
    if (!contract || !userAddress) return;

    try {
      setLoading(true);
      const [totalBalance, liquidBal] = await Promise.all([
        contract.balanceOf(userAddress),
        contract.getLiquidBalance(userAddress)
      ]);

      setBalance(ethers.formatEther(totalBalance));
      setLiquidBalance(ethers.formatEther(liquidBal));
    } catch (err) {
      console.error('Failed to load balance:', err);
    } finally {
      setLoading(false);
    }
  }, [contract]);

  const transfer = useCallback(async (to, amount) => {
    if (!contract) throw new Error('Contract not loaded');

    const tx = await contract.transfer(to, ethers.parseEther(amount));
    return tx.wait();
  }, [contract]);

  const stakeTokens = useCallback(async (amount, context) => {
    if (!contract) throw new Error('Contract not loaded');

    const tx = await contract.stakeTokens(ethers.parseEther(amount), context);
    return tx.wait();
  }, [contract]);

  const lockTokens = useCallback(async (amount, duration, context) => {
    if (!contract) throw new Error('Contract not loaded');

    const tx = await contract.lockTokens(ethers.parseEther(amount), duration, context);
    return tx.wait();
  }, [contract]);

  const getContextAllocation = useCallback(async (userAddress, context) => {
    if (!contract || !userAddress) return '0';

    const allocation = await contract.getContextAllocation(userAddress, context);
    return ethers.formatEther(allocation);
  }, [contract]);

  return {
    contract,
    balance,
    liquidBalance,
    loading,
    loadBalance,
    transfer,
    stakeTokens,
    lockTokens,
    getContextAllocation
  };
};

// Marketplace Hook
export const useFTNSMarketplace = () => {
  const { signer, network } = useWeb3();
  const [contract, setContract] = useState(null);
  const [services, setServices] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (signer && network) {
      const networkName = network.chainId === 137 ? 'polygon' : 'mumbai';
      const address = CONTRACT_ADDRESSES[networkName]?.marketplace;
      
      if (address) {
        const marketplaceContract = new ethers.Contract(address, FTNS_MARKETPLACE_ABI, signer);
        setContract(marketplaceContract);
      }
    }
  }, [signer, network]);

  const listService = useCallback(async (serviceData) => {
    if (!contract) throw new Error('Contract not loaded');

    const { context, name, description, serviceType, price, minStake } = serviceData;
    const tx = await contract.listService(
      context,
      name,
      description,
      serviceType,
      ethers.parseEther(price),
      ethers.parseEther(minStake || '0')
    );
    return tx.wait();
  }, [contract]);

  const purchaseService = useCallback(async (serviceId) => {
    if (!contract) throw new Error('Contract not loaded');

    const tx = await contract.purchaseService(serviceId);
    return tx.wait();
  }, [contract]);

  const getServicesByContext = useCallback(async (context) => {
    if (!contract) return [];

    setLoading(true);
    try {
      const serviceIds = await contract.getServicesByContext(context);
      const servicePromises = serviceIds.map(id => contract.getService(id));
      const servicesData = await Promise.all(servicePromises);
      
      setServices(servicesData);
      return servicesData;
    } catch (err) {
      console.error('Failed to load services:', err);
      return [];
    } finally {
      setLoading(false);
    }
  }, [contract]);

  const submitReview = useCallback(async (serviceId, rating, comment) => {
    if (!contract) throw new Error('Contract not loaded');

    const tx = await contract.submitReview(serviceId, rating, comment);
    return tx.wait();
  }, [contract]);

  return {
    contract,
    services,
    loading,
    listService,
    purchaseService,
    getServicesByContext,
    submitReview
  };
};

// Network utilities
export const useNetworkHelper = () => {
  const { provider, network } = useWeb3();

  const switchToPolygon = useCallback(async () => {
    if (!window.ethereum) return;

    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: '0x89' }], // Polygon Mainnet
      });
    } catch (switchError) {
      // If network doesn't exist, add it
      if (switchError.code === 4902) {
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [{
            chainId: '0x89',
            chainName: 'Polygon Mainnet',
            nativeCurrency: {
              name: 'MATIC',
              symbol: 'MATIC',
              decimals: 18,
            },
            rpcUrls: ['https://polygon-rpc.com'],
            blockExplorerUrls: ['https://polygonscan.com'],
          }],
        });
      }
    }
  }, []);

  const switchToMumbai = useCallback(async () => {
    if (!window.ethereum) return;

    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: '0x13881' }], // Mumbai Testnet
      });
    } catch (switchError) {
      if (switchError.code === 4902) {
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [{
            chainId: '0x13881',
            chainName: 'Polygon Mumbai',
            nativeCurrency: {
              name: 'MATIC',
              symbol: 'MATIC',
              decimals: 18,
            },
            rpcUrls: ['https://rpc-mumbai.maticvigil.com'],
            blockExplorerUrls: ['https://mumbai.polygonscan.com'],
          }],
        });
      }
    }
  }, []);

  const addFTNSToken = useCallback(async () => {
    if (!window.ethereum || !network) return;

    const networkName = network.chainId === 137 ? 'polygon' : 'mumbai';
    const tokenAddress = CONTRACT_ADDRESSES[networkName]?.ftnsToken;

    if (tokenAddress) {
      await window.ethereum.request({
        method: 'wallet_watchAsset',
        params: {
          type: 'ERC20',
          options: {
            address: tokenAddress,
            symbol: 'FTNS',
            decimals: 18,
            image: 'https://prsm.ai/logo.png', // Replace with actual logo
          },
        },
      });
    }
  }, [network]);

  return {
    switchToPolygon,
    switchToMumbai,
    addFTNSToken,
    isPolygon: network?.chainId === 137,
    isMumbai: network?.chainId === 80001,
  };
};

// Transaction helper hook
export const useTransactions = () => {
  const [pending, setPending] = useState(new Set());
  const [completed, setCompleted] = useState([]);

  const addTransaction = useCallback((txHash, description) => {
    setPending(prev => new Set([...prev, txHash]));
    
    // You could also add to a more persistent storage here
    const tx = {
      hash: txHash,
      description,
      timestamp: Date.now(),
      status: 'pending'
    };
    
    setCompleted(prev => [tx, ...prev]);
  }, []);

  const markCompleted = useCallback((txHash, status = 'success') => {
    setPending(prev => {
      const newPending = new Set(prev);
      newPending.delete(txHash);
      return newPending;
    });

    setCompleted(prev => 
      prev.map(tx => 
        tx.hash === txHash 
          ? { ...tx, status, completedAt: Date.now() }
          : tx
      )
    );
  }, []);

  return {
    pendingTxs: Array.from(pending),
    completedTxs: completed,
    addTransaction,
    markCompleted,
    hasPending: pending.size > 0
  };
};

export default {
  Web3Provider,
  useWeb3,
  useFTNSToken,
  useFTNSMarketplace,
  useNetworkHelper,
  useTransactions
};