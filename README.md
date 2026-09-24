# CS50 Analysis Module
Module to implement the pattern detections for long term data through harmonic pattern detection, medium term data through classical chart pattern detection and short term data through 
candlestick pattern detection. 
 - The data for long-term data is 1 year of OHLCV data for 1 day candles
 - The data for medium-term data is 1 month of OHLCV data for 1 hr candles
 - The data for short-term data is 1 week of OHLCV data for 1min candles.
   
### Testing Analysis Module
* Please run the kafka/finance_ohlc_producer.py in the terminal to obtained the data-> do not interrupt this
  - command to run: python kafka/finance_ohlc_producer.py
* Please run the kafka/producer2.py in another terminal to obtain the analysis data -> do not interrupt this
  - command to run: python -m kafka.producer.py
    
**Note: run both of these program in the Analysis root directory and not in the kafka folder**
