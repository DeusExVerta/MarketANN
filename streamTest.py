import os
from alpaca_trade_api import StreamConn
from alpaca_trade_api.common import URL


ALPACA_API_KEY = os.environ['APCA_API_KEY_ID']
ALPACA_SECRET_KEY = os.environ['APCA_API_SECRET_KEY']
mbars = []

if __name__ == '__main__':
    import logging

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    conn = StreamConn(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            base_url=URL('https://paper-api.alpaca.markets'),
            data_url=URL('https://data.alpaca.markets'),
            data_stream='alpacadatav1',
            debug = True
        )
    

    @conn.on(r'^AM\..+$')
    async def on_minute_bars(conn, channel, bar):
        if len(mbars)>480:
            mbars.pop(0)
        mbars.append(bar)
        print('bars', bar)

    quote_count = 0  # don't print too much quotes
    @conn.on(r'Q\..+', ['AAPL'])
    async def on_quotes(conn, channel, quote):
        global quote_count
        if quote_count % 10 == 0:
            print('quote', quote)
        quote_count += 1


    @conn.on(r'T\..+', ['AAPL'])
    async def on_trades(conn, channel, trade):
        print('trade', trade)

    conn.run(['alpacadatav1/AM.AAPL'])