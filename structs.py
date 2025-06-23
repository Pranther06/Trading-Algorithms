from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Trade:
    """Trade in the market"""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = None
    trader_id: Optional[str] = None  # Who made the trade
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass 
class Holding:
    """Holding in a security"""
    symbol: str
    quantity: float
    avg_cost: float

class TradingHistory:
    """Trades in the market"""
    
    def __init__(self):
        self.market_trades: List[Trade] = []
    
    def add_market_trade(self, trade: Trade):
        self.market_trades.append(trade)
    
    def get_market_trades(self, symbol: str = None) -> List[Trade]:
        if symbol is None:
            return self.market_trades.copy()
        return [trade for trade in self.market_trades if trade.symbol == symbol]
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        symbol_trades = self.get_market_trades(symbol)
        if not symbol_trades:
            return None
        return symbol_trades[-1].price
    
    def get_volume(self, symbol: str) -> float:
        symbol_trades = self.get_market_trades(symbol)
        return sum(trade.quantity for trade in symbol_trades)
    
    def get_price_history(self, symbol: str) -> List[float]:
        symbol_trades = self.get_market_trades(symbol)
        return [trade.price for trade in symbol_trades]

class Portfolio:
    """Personal trades and holdings"""
    
    def __init__(self, initial_cash: float = 100000.0, trader_id: str = "me"):
        self.cash = initial_cash
        self.trader_id = trader_id
        self.holdings: Dict[str, Holding] = {}
        self.my_trades: List[Trade] = []  # Only your trades
    
    def execute_trade(self, symbol: str, side: OrderSide, quantity: float, price: float, 
                     market_history: TradingHistory) -> bool:
        
        if not self._validate_trade(symbol, side, quantity, price):
            return False
        
        trade = Trade(symbol, side, quantity, price, trader_id=self.trader_id)
        
        self.my_trades.append(trade)
        
        market_history.add_market_trade(trade)
        
        # Update your holdings and cash
        self._update_holdings(trade)
        
        return True
    
    def get_holding(self, symbol: str) -> Optional[Holding]:
        return self.holdings.get(symbol)
    
    def get_all_holdings(self) -> Dict[str, Holding]:
        return self.holdings.copy()
    
    def get_cash(self) -> float:
        return self.cash
    
    def get_my_trades(self, symbol: str = None) -> List[Trade]:
        if symbol is None:
            return self.my_trades.copy()
        return [trade for trade in self.my_trades if trade.symbol == symbol]
    
    def _validate_trade(self, symbol: str, side: OrderSide, quantity: float, price: float) -> bool:
        if side == OrderSide.BUY:
            # Check if enough cash
            cost = quantity * price
            return cost <= self.cash
        
        elif side == OrderSide.SELL:
            # Check if enough shares
            holding = self.get_holding(symbol)
            return holding is not None and holding.quantity >= quantity
        
        return False
    
    def _update_holdings(self, trade: Trade):
        if trade.side == OrderSide.BUY:
            self.cash -= trade.quantity * trade.price
            
            if trade.symbol in self.holdings:
                holding = self.holdings[trade.symbol]
                total_cost = (holding.quantity * holding.avg_cost) + (trade.quantity * trade.price)
                total_quantity = holding.quantity + trade.quantity
                holding.avg_cost = total_cost / total_quantity
                holding.quantity = total_quantity
            else:
                self.holdings[trade.symbol] = Holding(trade.symbol, trade.quantity, trade.price)
        
        elif trade.side == OrderSide.SELL:
            self.cash += trade.quantity * trade.price
            
            holding = self.holdings[trade.symbol]
            holding.quantity -= trade.quantity
            
            if holding.quantity <= 0:
                del self.holdings[trade.symbol]

class TradingStrategy(ABC):
    """Trading strategy"""
    
    def __init__(self, name: str, trader_id: str = None):
        self.name = name
        self.portfolio = Portfolio(trader_id=trader_id or name)
    
    @abstractmethod
    def generate_signals(self, market_history: TradingHistory) -> List[Dict]:
        """
        Generate trading signals based on market trading history
        
        Args:
            market_history: TradingHistory containing all market trades
            
        Returns:
            List of signal dictionaries with keys: symbol, side, quantity, price
        """
        pass
    
    @abstractmethod
    def should_execute_signal(self, signal: Dict, market_history: TradingHistory) -> bool:
        """
        Determine if a signal should be executed
        
        Args:
            signal: Signal dictionary from generate_signals
            market_history: Current market trading history
            
        Returns:
            Boolean indicating whether to execute the signal
        """
        pass
    
    def execute_signals(self, signals: List[Dict], market_history: TradingHistory) -> List[bool]:
        """Execute a list of signals"""
        results = []
        for signal in signals:
            if self.should_execute_signal(signal, market_history):
                success = self.portfolio.execute_trade(
                    signal['symbol'],
                    signal['side'],
                    signal['quantity'],
                    signal['price'],
                    market_history
                )
                results.append(success)
            else:
                results.append(False)
        return results
    
    def run_strategy(self, market_history: TradingHistory) -> List[bool]:
        """Complete strategy execution pipeline"""
        signals = self.generate_signals(market_history)
        return self.execute_signals(signals, market_history)