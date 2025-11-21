from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime
import time
import threading


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.pair = None
        self.current_bars = []

    # def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="", errorTime=""):
    #     if errorCode in [2104, 2106, 2158]:
    #         return
    #     print(f"Error: {errorCode} - {errorString}")

    def error(self, reqId, errorCode, errorString):
        print("IB ERROR", reqId, errorCode, errorString)

    def historicalData(self, reqId, bar):
        self.current_bars.append(bar)

    def historicalDataEnd(self, reqId, start, end):
        if self.current_bars:
            bar = self.current_bars[-1]
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"{current_time} - {self.pair} @ {bar.close:.6f}")
        self.current_bars = []


def monitor_forex(symbol, currency, update_interval=15, port=4001):
    app = IBapi()
    app.connect('127.0.0.1', port, 123)
    app.pair = f"{symbol}/{currency}"

    api_thread = threading.Thread(target=lambda: app.run(), daemon=True)
    api_thread.start()

    time.sleep(2)

    contract = Contract()
    contract.symbol = symbol
    contract.secType = "CASH"
    contract.exchange = "IDEALPRO"
    contract.currency = currency

    print(f"\n{symbol}/{currency} (updates every {update_interval}s)")

    try:
        req_id = 1
        while True:
            app.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime='',
                durationStr='60 S',
                barSizeSetting='30 secs',
                whatToShow='MIDPOINT',
                useRTH=0,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )

            req_id += 1
            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\nStopping...")
        app.disconnect()


monitor_forex('EUR', 'USD', update_interval=15)

