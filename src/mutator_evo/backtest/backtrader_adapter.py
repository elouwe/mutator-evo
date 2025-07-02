import backtrader as bt

class BacktraderAdapter:
    def __init__(self, data_feed):
        self.data_feed = data_feed
        
    def evaluate(self, strategy_embedding) -> dict:
        """Runs a strategy backtest through Backtrader"""
        class CustomStrategy(bt.Strategy):
            params = strategy_embedding.features
            
            def __init__(self):
                # Initializing indicators
                pass
                
            def next(self):
                # The logic of strategy
                pass
                
        cerebro = bt.Cerebro()
        cerebro.adddata(self.data_feed)
        cerebro.addstrategy(CustomStrategy)
        
        # Add analytics
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        results = cerebro.run()
        strat = results[0]
        
        return {
            "sharpe": strat.analyzers.sharpe.get_analysis()['sharperatio'],
            "max_drawdown": strat.analyzers.drawdown.get_analysis()['max']['drawdown'],
            "win_rate": strat.analyzers.trades.get_analysis()['won']['total'] / 
                        strat.analyzers.trades.get_analysis()['total']['total']
        }